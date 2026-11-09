"""
A Categorizer folder written by LabGym looks like
	<models_dir>/<name>/
		model_parameters.txt   (CSV; one row per behavior class)
		saved_model.pb         (Keras SavedModel)
		variables/             (Keras SavedModel weights)
		training_metrics.csv   (optional)
		training_history.png   (optional)
"""

# Allow use of newer syntax Python 3.10 type hints in Python 3.9.
from __future__ import annotations

import csv
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import tempfile

# Log the load of this module (by the module loader, on first import).
logger = logging.getLogger(__name__)
logger.debug('loading %s', __file__)


PARAMETERS_FILENAME = 'model_parameters.txt'
SAVED_MODEL_FILENAME = 'saved_model.pb'
VARIABLES_DIRNAME = 'variables'
TRAINING_METRICS_FILENAME = 'training_metrics.csv'
TRAINING_HISTORY_FILENAME = 'training_history.png'

# Folder names that are never Categorizers.  Hidden folders (names
# starting with '.') are also skipped.
EXCLUDED_DIRNAMES = frozenset({
	'__pycache__',
	'__init__',
	'__MACOSX',
	'.ipynb_checkpoints',
})

STATUS_AVAILABLE = 'Available'
STATUS_INCOMPLETE = 'Incomplete'

NETWORK_PATTERN = 0
NETWORK_ANIMATION = 1
NETWORK_COMBINED = 2

NETWORK_TYPES = {
	NETWORK_PATTERN: 'Pattern',
	NETWORK_ANIMATION: 'Animation',
	NETWORK_COMBINED: 'Combined',
}

BEHAVIOR_MODES = {
	0: 'Non-interactive',
	1: 'Interactive basic',
	2: 'Interactive advanced',
	3: 'Static images',
}

# Architecture columns that apply to each network type.
ARCHITECTURE_FIELDS = {
	NETWORK_PATTERN: ('dim_conv', 'level_conv', 'channel'),
	NETWORK_ANIMATION: ('dim_tconv', 'level_tconv', 'channel'),
	NETWORK_COMBINED: ('dim_conv', 'level_conv', 'dim_tconv', 'level_tconv', 'channel'),
}

# Columns that LabGym's own testing/analysis code reads unconditionally.
REQUIRED_FIELDS = ('classnames', 'network', 'time_step', 'inner_code', 'std', 'background_free')


@dataclasses.dataclass(frozen=True)
class CategorizerInfo:
	"""What the library knows about one folder in the models directory.

	Fields that were not recorded, or do not apply to this model's
	network type or behavior mode, are None.
	"""

	name: str
	path: str
	status: str
	problems: tuple[str, ...] = ()
	notes: tuple[str, ...] = ()

	network: int | None = None
	classnames: tuple[str, ...] = ()
	behavior_mode: int | None = None
	behavior_mode_recorded: bool = False

	time_step: int | None = None
	channel: int | None = None
	dim_conv: int | None = None
	level_conv: int | None = None
	dim_tconv: int | None = None
	level_tconv: int | None = None

	inner_code: int | None = None
	std: int | None = None
	background_free: int | None = None
	black_background: int | None = None
	social_distance: int | None = None
	color_code: int | None = None

	has_training_metrics: bool = False
	has_training_history: bool = False

	@property
	def is_available(self) -> bool:
		return self.status == STATUS_AVAILABLE

	@property
	def network_type(self) -> str | None:
		#'Pattern', 'Animation', 'Combined', or None if unknown.
		return NETWORK_TYPES.get(self.network)

	@property
	def behavior_mode_name(self) -> str | None:
		return BEHAVIOR_MODES.get(self.behavior_mode)

	def behavior_summary(self, preview: int = 3) -> str:
		#Class count plus a short preview, e.g. '5 classes: a, b, c, ...'.
		n = len(self.classnames)
		if n == 0:
			return 'No classes recorded'
		head = ', '.join(self.classnames[:preview])
		more = ', ...' if n > preview else ''
		return f"{n} {'class' if n == 1 else 'classes'}: {head}{more}"


class ModelParametersError(ValueError):
	"""model_parameters.txt is missing, unreadable, or malformed."""


def is_excluded_dirname(name: str) -> bool:
	return name in EXCLUDED_DIRNAMES or name.startswith('.')


def discover_categorizers(models_dir) -> list[CategorizerInfo]:
	"""Return a CategorizerInfo for each candidate folder in models_dir.

	Only immediate subfolders are considered; files and excluded
	folders are skipped.  Results are sorted case-insensitively by name.
	Raises OSError if models_dir itself cannot be listed.
	"""
	models_dir = os.path.abspath(models_dir)
	results = []
	with os.scandir(models_dir) as entries:
		for entry in entries:
			if is_excluded_dirname(entry.name):
				continue
			try:
				if not entry.is_dir():
					continue
			except OSError:
				continue
			results.append(read_categorizer(entry.path))
	results.sort(key=lambda info: (info.name.casefold(), info.name))
	return results


def read_categorizer(path) -> CategorizerInfo:
	"""Read one model folder.

	Never raises for a bad folder; problems are reported through
	CategorizerInfo.status and CategorizerInfo.problems.
	"""
	path = os.path.abspath(path)
	name = os.path.basename(path)
	problems = []
	notes = []
	fields = {}

	try:
		rows = read_model_parameters(os.path.join(path, PARAMETERS_FILENAME))
	except ModelParametersError as e:
		problems.append(str(e))
	else:
		fields = _interpret_parameters(rows, problems, notes)

	if not os.path.isfile(os.path.join(path, SAVED_MODEL_FILENAME)):
		problems.append(f'{SAVED_MODEL_FILENAME} not found')
	if not os.path.isdir(os.path.join(path, VARIABLES_DIRNAME)):
		problems.append(f'{VARIABLES_DIRNAME}/ folder not found')

	return CategorizerInfo(
		name=name,
		path=path,
		status=STATUS_INCOMPLETE if problems else STATUS_AVAILABLE,
		problems=tuple(problems),
		notes=tuple(notes),
		has_training_metrics=os.path.isfile(os.path.join(path, TRAINING_METRICS_FILENAME)),
		has_training_history=os.path.isfile(os.path.join(path, TRAINING_HISTORY_FILENAME)),
		**fields,
	)


def read_model_parameters(params_path) -> list[dict[str, str]]:
	"""Read model_parameters.txt as a list of row dicts.

	The file is the CSV that categorizer.py writes with
	pandas.DataFrame.to_csv: a header row, then one row per class.
	Keys and values are stripped strings; values are not converted.
	"""
	try:
		with open(params_path, newline='', encoding='utf-8-sig') as f:
			reader = csv.DictReader(f)
			rows = list(reader)
			header = reader.fieldnames
	except FileNotFoundError:
		raise ModelParametersError(f'{PARAMETERS_FILENAME} not found') from None
	except (OSError, UnicodeDecodeError, csv.Error) as e:
		raise ModelParametersError(f'{PARAMETERS_FILENAME} could not be read ({e})') from None

	if not header:
		raise ModelParametersError(f'{PARAMETERS_FILENAME} is empty')
	if not rows:
		raise ModelParametersError(f'{PARAMETERS_FILENAME} has no behavior classes')
	for row in rows:
		# DictReader puts surplus values under the key None and fills
		# missing values with None; either means a ragged row.
		if None in row or None in row.values():
			raise ModelParametersError(f'{PARAMETERS_FILENAME} has rows with the wrong number of columns')
	return [{k.strip(): v.strip() for k, v in row.items()} for row in rows]


def _interpret_parameters(rows, problems, notes) -> dict:
	"""Convert parameter rows into CategorizerInfo fields.

	Appends to problems (which make the model Incomplete) and to notes
	(informational only).  Settings are read from the first row, as
	LabGym does; class names are read from every row.
	"""
	first = rows[0]
	out = {}

	for field in REQUIRED_FIELDS:
		if field not in first:
			problems.append(f'{field} not recorded in {PARAMETERS_FILENAME}')

	def as_int(field, required=False):
		"""Integer value of field, or None if absent, blank, or invalid.

		Blank values are a problem only for required fields; invalid
		values are always a problem.
		"""
		raw = first.get(field)
		if raw is None:
			return None
		if raw == '':
			if required:
				problems.append(f'{field} is blank in {PARAMETERS_FILENAME}')
			return None
		try:
			value = float(raw)
		except ValueError:
			value = None
		if value is None or not value.is_integer():
			problems.append(f'{field} has an invalid value {raw!r}')
			return None
		return int(value)

	# Class names: one per row, kept as strings exactly as recorded.
	if 'classnames' in first:
		classnames = tuple(row['classnames'] for row in rows)
		if any(c == '' for c in classnames):
			problems.append('one or more behavior class names are blank')
		if len(set(classnames)) != len(classnames):
			problems.append('behavior class names are duplicated')
		out['classnames'] = classnames

	network = as_int('network', required=True)
	if network is not None and network not in NETWORK_TYPES:
		problems.append(f'network has an unknown value {network}')
		network = None
	out['network'] = network

	if 'behavior_kind' in first:
		behavior_mode = as_int('behavior_kind', required=True)
		if behavior_mode is not None and behavior_mode not in BEHAVIOR_MODES:
			problems.append(f'behavior_kind has an unknown value {behavior_mode}')
			behavior_mode = None
		out['behavior_mode'] = behavior_mode
		out['behavior_mode_recorded'] = True
	else:
		# Same default LabGym applies when loading such a model.
		out['behavior_mode'] = 0
		notes.append('behavior mode not recorded; LabGym treats this model as Non-interactive')

	for field in ('time_step', 'inner_code', 'std', 'background_free'):
		out[field] = as_int(field, required=True)
	out['black_background'] = as_int('black_background')

	# Architecture settings only for the network type that uses them.
	if network is not None:
		for field in ARCHITECTURE_FIELDS[network]:
			if field not in first:
				problems.append(f'{field} not recorded (required for {NETWORK_TYPES[network]} models)')
			out[field] = as_int(field, required=True)

	# Settings that only apply to Interactive advanced models.
	if out['behavior_mode'] == 2:
		if 'social_distance' not in first:
			problems.append('social_distance not recorded for an Interactive advanced model')
		out['social_distance'] = as_int('social_distance', required=True)
		# LabGym has a default when color_code is missing, so it is optional.
		if network in (NETWORK_ANIMATION, NETWORK_COMBINED):
			out['color_code'] = as_int('color_code')

	return out


# Display text.  Kept here, not in the dialog, so it can be tested
# without wx and reused by any entry point.

UNKNOWN = 'Unknown'

AVAILABLE_MEANING = (
	'Available means the model files and required settings are present. '
	'It does not confirm compatibility with this version of LabGym.')


def list_row(info: CategorizerInfo) -> tuple[str, str, str, str, str]:
	"""Main-list columns: name, network type, behaviors, behavior mode, status."""
	return (
		info.name,
		info.network_type or UNKNOWN,
		info.behavior_summary(),
		info.behavior_mode_name or UNKNOWN,
		info.status,
	)


def detail_rows(info: CategorizerInfo) -> list[tuple[str, str]]:
	"""(label, value) pairs for the selected-model details.

	Only settings that apply to the model's network type and behavior
	mode are included.  Values come from the model folder only.
	"""
	rows = [('Name', info.name)]

	if info.is_available:
		rows.append(('Status', f'{STATUS_AVAILABLE} (files present; compatibility not checked)'))
	else:
		rows.append(('Status', STATUS_INCOMPLETE))
		rows.append(('Problems', '\n'.join(info.problems)))

	rows.append(('Network type', info.network_type or UNKNOWN))
	mode = info.behavior_mode_name or UNKNOWN
	if info.behavior_mode_name and not info.behavior_mode_recorded:
		mode += ' (not recorded; LabGym default)'
	rows.append(('Behavior mode', mode))

	if info.classnames:
		rows.append((f'Behavior classes ({len(info.classnames)})', '\n'.join(info.classnames)))
	else:
		rows.append(('Behavior classes', 'None recorded'))

	def add(label, value, fmt='{}'):
		if value is not None:
			rows.append((label, fmt.format(value)))

	if info.network in (NETWORK_PATTERN, NETWORK_COMBINED):
		add('Pattern Recognizer dimension', info.dim_conv)
		add('Pattern Recognizer level', info.level_conv)
	if info.network in (NETWORK_ANIMATION, NETWORK_COMBINED):
		add('Animation Analyzer dimension', info.dim_tconv)
		add('Animation Analyzer level', info.level_tconv)
	add('Time step', info.time_step, '{} frames')
	add('Channel', info.channel)

	if info.inner_code is not None:
		if info.inner_code == 0:
			body = 'Included' if info.std is None else f'Included (STD {info.std})'
		else:
			body = 'Not included'
		rows.append(('Body parts', body))

	if info.background_free is not None:
		if info.background_free == 0:
			background = 'Excluded'
			if info.black_background == 1:
				background += ' (white background)'
			elif info.black_background == 0:
				background += ' (black background)'
		else:
			background = 'Included'
		rows.append(('Background', background))

	if info.behavior_mode == 2:
		if info.social_distance == 0:
			rows.append(('Social distance', '0 (no limit)'))
		else:
			add('Social distance', info.social_distance)
		if info.network in (NETWORK_ANIMATION, NETWORK_COMBINED):
			if info.color_code is None:
				costar = 'Grayscale (not recorded; LabGym default)'
			else:
				costar = 'RGB' if info.color_code == 0 else 'Grayscale'
			rows.append(('Supporting characters', costar))

	rows.append(('Location', info.path))
	rows.append(('Training metrics', TRAINING_METRICS_FILENAME if info.has_training_metrics else 'Not found'))
	rows.append(('Training history', TRAINING_HISTORY_FILENAME if info.has_training_history else 'Not found'))
	rows.append(('LabGym version', 'Not recorded'))

	if info.notes:
		rows.append(('Notes', '\n'.join(info.notes)))
	return rows


def format_details(info: CategorizerInfo) -> str:
	"""detail_rows as plain text, one 'Label: value' block per row."""
	blocks = []
	for label, value in detail_rows(info):
		if '\n' in value:
			lines = '\n'.join('    ' + v for v in value.split('\n'))
			blocks.append(f'{label}:\n{lines}')
		else:
			blocks.append(f'{label}: {value}')
	return '\n'.join(blocks)


# File actions: reveal, export, rename.

class LibraryActionError(Exception):
	"""A file action failed or was refused; str(e) is shown to the user."""


# Characters Windows does not allow in file names.  They are rejected
# on every platform so a renamed model can be shared across platforms.
INVALID_NAME_CHARS = frozenset('<>:"/\\|?*')

WINDOWS_RESERVED_NAMES = frozenset(
	['CON', 'PRN', 'AUX', 'NUL']
	+ [f'COM{i}' for i in range(1, 10)]
	+ [f'LPT{i}' for i in range(1, 10)])

MAX_NAME_BYTES = 255


def _find_casefold(folder, name):
	"""Name of an entry in folder matching name case-insensitively, or None.

	Case-insensitive because the default macOS and Windows file systems
	treat 'Mouse' and 'mouse' as the same folder.
	"""
	key = name.casefold()
	for entry in os.listdir(folder):
		if entry.casefold() == key:
			return entry
	return None


def validate_new_name(models_dir, current_name, new_name) -> str | None:
	"""Return why new_name cannot be used to rename current_name, or None."""
	if not new_name or not new_name.strip():
		return 'Enter a name.'
	if new_name != new_name.strip():
		return 'The name cannot start or end with a space.'
	if new_name == current_name:
		return 'The new name is the same as the current name.'
	if '/' in new_name or '\\' in new_name:
		return 'The name cannot contain / or \\.'
	if any(c in INVALID_NAME_CHARS or ord(c) < 32 for c in new_name):
		return 'The name cannot contain any of these characters: < > : " | ? *'
	if new_name.startswith('.'):
		return 'The name cannot start with a period; such folders are hidden from the library.'
	if new_name.endswith('.'):
		return 'The name cannot end with a period.'
	if is_excluded_dirname(new_name):
		return f'"{new_name}" is reserved and cannot be used as a Categorizer name.'
	if new_name.split('.')[0].strip().upper() in WINDOWS_RESERVED_NAMES:
		return f'"{new_name}" is a reserved name on Windows.'
	if len(new_name.encode('utf-8')) > MAX_NAME_BYTES:
		return 'The name is too long.'
	try:
		existing = [e for e in os.listdir(models_dir) if e != current_name]
	except OSError as e:
		return f'The models folder could not be read ({e}).'
	key = new_name.casefold()
	clash = next((e for e in existing if e.casefold() == key), None)
	if clash is not None:
		return f'A folder named "{clash}" already exists in the models folder.'
	return None


def rename_categorizer(models_dir, current_name, new_name) -> str:
	"""Rename a model folder inside models_dir; return the new path.

	Raises LibraryActionError if the name is invalid, collides with an
	existing folder, or the rename fails.  Never replaces another folder.
	"""
	models_dir = os.path.abspath(models_dir)
	src = os.path.join(models_dir, current_name)
	if not os.path.isdir(src):
		raise LibraryActionError(f'"{current_name}" no longer exists. Refresh the library.')
	problem = validate_new_name(models_dir, current_name, new_name)
	if problem:
		raise LibraryActionError(problem)
	dst = os.path.join(models_dir, new_name)
	try:
		os.rename(src, dst)
	except OSError as e:
		raise LibraryActionError(f'"{current_name}" could not be renamed ({e}).') from None
	return dst


def export_categorizer(src_path, dest_dir) -> str:
	"""Copy a model folder into dest_dir; return the new folder's path.

	The copy is made in a hidden temporary folder inside dest_dir and
	moved into place only after it finishes, so a failed export never
	leaves a folder that looks like a complete model.  An existing
	folder of the same name is never overwritten or merged into.
	Raises LibraryActionError on any failure.
	"""
	src_path = os.path.abspath(src_path)
	dest_dir = os.path.abspath(dest_dir)
	name = os.path.basename(src_path)
	if not os.path.isdir(src_path):
		raise LibraryActionError(f'"{name}" no longer exists. Refresh the library.')
	if not os.path.isdir(dest_dir):
		raise LibraryActionError(f'The destination folder does not exist: {dest_dir}')
	if os.path.commonpath([src_path, dest_dir]) == src_path:
		raise LibraryActionError(f'"{name}" cannot be exported into its own folder.')

	def check_collision():
		try:
			clash = _find_casefold(dest_dir, name)
		except OSError as e:
			raise LibraryActionError(f'The destination folder could not be read ({e}).') from None
		if clash is not None:
			raise LibraryActionError(
				f'The destination already contains a folder named "{clash}". '
				'Nothing was copied. Choose another destination, or move or rename that folder first.')

	check_collision()
	target = os.path.join(dest_dir, name)
	try:
		staging = tempfile.mkdtemp(prefix=f'.{name}.export-incomplete-', dir=dest_dir)
	except OSError as e:
		raise LibraryActionError(f'Could not write to the destination folder ({e}).') from None
	try:
		staged = os.path.join(staging, name)
		shutil.copytree(src_path, staged)
		# Check again in case the folder appeared while copying.  os.rename
		# then refuses a non-empty existing folder on every platform.
		check_collision()
		os.rename(staged, target)
	except LibraryActionError:
		raise
	except (OSError, shutil.Error) as e:
		raise LibraryActionError(f'"{name}" could not be exported; nothing was kept ({e}).') from None
	finally:
		shutil.rmtree(staging, ignore_errors=True)
	return target


def reveal_command(path, platform=None) -> list[str]:
	"""Command that shows path in the system file manager."""
	platform = sys.platform if platform is None else platform
	if platform == 'darwin':
		return ['open', '-R', path]
	if platform.startswith('win'):
		return ['explorer', '/select,', os.path.normpath(path)]
	# Most Linux file managers cannot select an item, so open its parent.
	return ['xdg-open', os.path.dirname(path)]


def reveal_location(path) -> None:
	"""Show path in the system file manager without waiting for it."""
	if not os.path.exists(path):
		raise LibraryActionError(f'{path} no longer exists. Refresh the library.')
	try:
		subprocess.Popen(reveal_command(path))
	except OSError as e:
		raise LibraryActionError(f'The file manager could not be opened ({e}). The location is:\n{path}') from None

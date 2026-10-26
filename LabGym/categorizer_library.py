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
	NETWORK_PATTERN: ('dim_conv', 'level_conv'),
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
				problems.append(f'{field} not recorded for a {NETWORK_TYPES[network]} model')
			out[field] = as_int(field, required=True)
	# Static-image Pattern Recognizers also record their channel count.
	if network == NETWORK_PATTERN and out['behavior_mode'] == 3:
		out['channel'] = as_int('channel')

	# Settings that only apply to Interactive advanced models.
	if out['behavior_mode'] == 2:
		out['social_distance'] = as_int('social_distance')
		if network in (NETWORK_ANIMATION, NETWORK_COMBINED):
			out['color_code'] = as_int('color_code')

	return out

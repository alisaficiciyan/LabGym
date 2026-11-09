"""Categorizer Library core: discovery and metadata parsing (no GUI)."""

import os
from pathlib import Path

from LabGym import categorizer_library as cl


# Header/values in the column order categorizer.py writes for each network.
PATTERN_PARAMS = {
	'dim_conv': '32', 'channel': '1', 'time_step': '15', 'network': '0',
	'level_conv': '3', 'inner_code': '1', 'std': '0', 'background_free': '1',
	'black_background': '0', 'behavior_kind': '0', 'social_distance': '0',
}
ANIMATION_PARAMS = {
	'dim_tconv': '16', 'channel': '3', 'time_step': '10', 'network': '1',
	'level_tconv': '2', 'inner_code': '1', 'std': '0', 'background_free': '0',
	'black_background': '1', 'behavior_kind': '1', 'social_distance': '0',
	'color_code': '0',
}
COMBINED_PARAMS = {
	'dim_tconv': '64', 'dim_conv': '64', 'channel': '1', 'time_step': '14',
	'network': '2', 'level_tconv': '4', 'level_conv': '4', 'inner_code': '0',
	'std': '50', 'background_free': '0', 'black_background': '0',
	'behavior_kind': '2', 'social_distance': '3', 'color_code': '1',
}


def _params_text(classnames, params):
	header = ['classnames'] + list(params)
	lines = [','.join(header)]
	for name in classnames:
		lines.append(','.join([name] + list(params.values())))
	return '\n'.join(lines) + '\n'


def _make_model(root: Path, name, classnames=('a', 'b'), params=None,
                params_text=None, weights=True, metrics=False, history=False):
	"""Create a fake model folder like the ones categorizer.py writes."""
	folder = root / name
	folder.mkdir(parents=True)
	if params_text is None and params is not None:
		params_text = _params_text(classnames, params)
	if params_text is not None:
		(folder / cl.PARAMETERS_FILENAME).write_text(params_text, encoding='utf-8')
	if weights:
		(folder / cl.SAVED_MODEL_FILENAME).write_bytes(b'')
		(folder / cl.VARIABLES_DIRNAME).mkdir()
	if metrics:
		(folder / cl.TRAINING_METRICS_FILENAME).write_text('', encoding='utf-8')
	if history:
		(folder / cl.TRAINING_HISTORY_FILENAME).write_bytes(b'')
	return folder


def _only(models_dir):
	infos = cl.discover_categorizers(models_dir)
	assert len(infos) == 1
	return infos[0]


# Valid models.

def test_pattern_model(tmp_path):
	folder = _make_model(tmp_path, 'pat', ('groom', 'rear', 'walk', 'sit'), PATTERN_PARAMS)
	info = _only(tmp_path)
	assert info.status == cl.STATUS_AVAILABLE
	assert info.is_available
	assert info.problems == ()
	assert info.name == 'pat'
	assert info.path == os.path.abspath(folder)
	assert info.network_type == 'Pattern'
	assert info.classnames == ('groom', 'rear', 'walk', 'sit')
	assert info.behavior_mode_name == 'Non-interactive'
	assert info.behavior_mode_recorded
	assert (info.dim_conv, info.level_conv, info.channel) == (32, 3, 1)
	assert info.time_step == 15
	assert info.background_free == 1
	assert info.black_background == 0


def test_pattern_model_omits_animation_fields(tmp_path):
	_make_model(tmp_path, 'pat', params=PATTERN_PARAMS)
	info = _only(tmp_path)
	assert info.dim_tconv is None
	assert info.level_tconv is None
	assert info.social_distance is None
	assert info.color_code is None


def test_static_image_pattern_model(tmp_path):
	_make_model(tmp_path, 'img', params={**PATTERN_PARAMS, 'behavior_kind': '3'})
	info = _only(tmp_path)
	assert info.is_available
	assert info.behavior_mode_name == 'Static images'
	assert info.channel == 1


def test_animation_model(tmp_path):
	_make_model(tmp_path, 'anim', ('x', 'y'), ANIMATION_PARAMS)
	info = _only(tmp_path)
	assert info.is_available
	assert info.network_type == 'Animation'
	assert info.behavior_mode_name == 'Interactive basic'
	assert (info.dim_tconv, info.level_tconv, info.channel) == (16, 2, 3)
	assert info.dim_conv is None
	assert info.level_conv is None
	assert info.color_code is None  # only for Interactive advanced


def test_combined_model(tmp_path):
	_make_model(tmp_path, 'combo', ('p', 'q', 'r'), COMBINED_PARAMS)
	info = _only(tmp_path)
	assert info.is_available
	assert info.network_type == 'Combined'
	assert info.behavior_mode_name == 'Interactive advanced'
	assert (info.dim_conv, info.level_conv) == (64, 4)
	assert (info.dim_tconv, info.level_tconv, info.channel) == (64, 4, 1)
	assert info.inner_code == 0
	assert info.std == 50
	assert info.social_distance == 3
	assert info.color_code == 1


def test_non_applicable_columns_ignored(tmp_path):
	# Stray conv columns in an Animation model are not reported.
	params = {**ANIMATION_PARAMS, 'dim_conv': '99', 'level_conv': '9'}
	_make_model(tmp_path, 'anim', params=params)
	info = _only(tmp_path)
	assert info.is_available
	assert info.dim_conv is None
	assert info.level_conv is None


def test_missing_behavior_kind_defaults_with_note(tmp_path):
	params = {k: v for k, v in PATTERN_PARAMS.items() if k != 'behavior_kind'}
	_make_model(tmp_path, 'old', params=params)
	info = _only(tmp_path)
	assert info.is_available
	assert info.behavior_mode == 0
	assert not info.behavior_mode_recorded
	assert any('behavior mode not recorded' in n for n in info.notes)


def test_classnames_kept_as_strings(tmp_path):
	_make_model(tmp_path, 'nums', ('1', '02', '3.0'), PATTERN_PARAMS)
	info = _only(tmp_path)
	assert info.classnames == ('1', '02', '3.0')


def test_quoted_and_non_ascii_classnames(tmp_path):
	text = (
		'classnames,dim_conv,channel,time_step,network,level_conv,inner_code,std,background_free\n'
		'"left, then right",32,1,15,0,3,1,0,1\n'
		'café,32,1,15,0,3,1,0,1\n'
	)
	_make_model(tmp_path, 'q', params_text=text)
	info = _only(tmp_path)
	assert info.is_available
	assert info.classnames == ('left, then right', 'café')


def test_training_artifacts_detected(tmp_path):
	_make_model(tmp_path, 'm1', params=PATTERN_PARAMS, metrics=True, history=True)
	_make_model(tmp_path, 'm2', params=PATTERN_PARAMS)
	m1, m2 = cl.discover_categorizers(tmp_path)
	assert m1.has_training_metrics and m1.has_training_history
	assert not m2.has_training_metrics and not m2.has_training_history


def test_behavior_summary():
	info = cl.CategorizerInfo(name='m', path='/m', status=cl.STATUS_AVAILABLE,
	                          classnames=('a', 'b', 'c', 'd'))
	assert info.behavior_summary() == '4 classes: a, b, c, ...'
	assert info.behavior_summary(preview=4) == '4 classes: a, b, c, d'
	one = cl.CategorizerInfo(name='m', path='/m', status=cl.STATUS_AVAILABLE,
	                         classnames=('a',))
	assert one.behavior_summary() == '1 class: a'
	none = cl.CategorizerInfo(name='m', path='/m', status=cl.STATUS_INCOMPLETE)
	assert none.behavior_summary() == 'No classes recorded'


# Discovery and filtering.

def test_excluded_and_non_directory_entries_skipped(tmp_path):
	_make_model(tmp_path, 'real', params=PATTERN_PARAMS)
	for junk in ('__pycache__', '__init__', '__MACOSX', '.ipynb_checkpoints', '.hidden'):
		(tmp_path / junk).mkdir()
	(tmp_path / '__init__.py').write_text('', encoding='utf-8')
	(tmp_path / '.DS_Store').write_bytes(b'')
	(tmp_path / 'notes.txt').write_text('', encoding='utf-8')
	assert [i.name for i in cl.discover_categorizers(tmp_path)] == ['real']


def test_only_immediate_subfolders_scanned(tmp_path):
	outer = _make_model(tmp_path, 'outer', params=PATTERN_PARAMS)
	_make_model(outer, 'nested', params=PATTERN_PARAMS)
	assert [i.name for i in cl.discover_categorizers(tmp_path)] == ['outer']


def test_sorted_case_insensitively(tmp_path):
	for name in ('beta', 'Alpha', 'gamma'):
		_make_model(tmp_path, name, params=PATTERN_PARAMS)
	assert [i.name for i in cl.discover_categorizers(tmp_path)] == ['Alpha', 'beta', 'gamma']


def test_empty_models_dir(tmp_path):
	assert cl.discover_categorizers(tmp_path) == []


def test_missing_models_dir_raises(tmp_path):
	try:
		cl.discover_categorizers(tmp_path / 'nope')
	except OSError:
		pass
	else:
		raise AssertionError('expected OSError')


# Incomplete and malformed folders.

def _assert_incomplete(info, fragment):
	assert info.status == cl.STATUS_INCOMPLETE
	assert not info.is_available
	assert any(fragment in p for p in info.problems), info.problems


def test_unrelated_folder_is_incomplete(tmp_path):
	(tmp_path / 'notes').mkdir()
	info = _only(tmp_path)
	_assert_incomplete(info, 'model_parameters.txt not found')
	_assert_incomplete(info, 'saved_model.pb not found')
	_assert_incomplete(info, 'variables/ folder not found')
	assert info.network_type is None
	assert info.classnames == ()


def test_missing_parameters_file(tmp_path):
	_make_model(tmp_path, 'm')
	_assert_incomplete(_only(tmp_path), 'model_parameters.txt not found')


def test_missing_weights(tmp_path):
	_make_model(tmp_path, 'm', params=COMBINED_PARAMS, weights=False)
	info = _only(tmp_path)
	_assert_incomplete(info, 'saved_model.pb not found')
	# metadata is still read so the user can identify the model
	assert info.network_type == 'Combined'
	assert info.classnames == ('a', 'b')


def test_empty_parameters_file(tmp_path):
	_make_model(tmp_path, 'm', params_text='')
	_assert_incomplete(_only(tmp_path), 'is empty')


def test_header_only_parameters_file(tmp_path):
	_make_model(tmp_path, 'm', params_text='classnames,network\n')
	_assert_incomplete(_only(tmp_path), 'no behavior classes')


def test_ragged_parameters_file(tmp_path):
	text = _params_text(('a',), PATTERN_PARAMS) + 'b,1,2\n'
	_make_model(tmp_path, 'm', params_text=text)
	_assert_incomplete(_only(tmp_path), 'wrong number of columns')


def test_non_utf8_parameters_file(tmp_path):
	folder = _make_model(tmp_path, 'm')
	(folder / cl.PARAMETERS_FILENAME).write_bytes(b'classnames\n\xff\xfe\x00\n')
	_assert_incomplete(_only(tmp_path), 'could not be read')


def test_missing_required_field(tmp_path):
	params = {k: v for k, v in PATTERN_PARAMS.items() if k != 'time_step'}
	_make_model(tmp_path, 'm', params=params)
	info = _only(tmp_path)
	_assert_incomplete(info, 'time_step not recorded')
	assert info.network_type == 'Pattern'


def test_missing_network_field(tmp_path):
	params = {k: v for k, v in PATTERN_PARAMS.items() if k != 'network'}
	_make_model(tmp_path, 'm', params=params)
	info = _only(tmp_path)
	_assert_incomplete(info, 'network not recorded')
	assert info.network_type is None
	assert info.dim_conv is None


def test_missing_architecture_field(tmp_path):
	params = {k: v for k, v in COMBINED_PARAMS.items() if k != 'level_tconv'}
	_make_model(tmp_path, 'm', params=params)
	_assert_incomplete(_only(tmp_path), 'level_tconv not recorded (required for Combined models)')


def _without(params, field):
	return {k: v for k, v in params.items() if k != field}


def test_pattern_model_without_channel(tmp_path):
	_make_model(tmp_path, 'm', params=_without(PATTERN_PARAMS, 'channel'))
	info = _only(tmp_path)
	_assert_incomplete(info, 'channel not recorded (required for Pattern models)')
	assert info.channel is None


def test_animation_and_combined_models_without_channel(tmp_path):
	_make_model(tmp_path, 'anim', params=_without(ANIMATION_PARAMS, 'channel'))
	_make_model(tmp_path, 'combo', params=_without(COMBINED_PARAMS, 'channel'))
	anim, combo = cl.discover_categorizers(tmp_path)
	_assert_incomplete(anim, 'channel not recorded (required for Animation models)')
	_assert_incomplete(combo, 'channel not recorded (required for Combined models)')


def test_blank_channel(tmp_path):
	_make_model(tmp_path, 'm', params={**PATTERN_PARAMS, 'channel': ''})
	_assert_incomplete(_only(tmp_path), 'channel is blank')


def test_interactive_advanced_without_social_distance(tmp_path):
	# Same rule for every network type.
	pattern = {**PATTERN_PARAMS, 'behavior_kind': '2'}
	_make_model(tmp_path, 'a_pat', params=_without(pattern, 'social_distance'))
	_make_model(tmp_path, 'b_anim', params=_without({**ANIMATION_PARAMS, 'behavior_kind': '2'}, 'social_distance'))
	_make_model(tmp_path, 'c_combo', params=_without(COMBINED_PARAMS, 'social_distance'))
	for info in cl.discover_categorizers(tmp_path):
		_assert_incomplete(info, 'social_distance not recorded for an Interactive advanced model')
		assert info.social_distance is None


def test_blank_social_distance(tmp_path):
	_make_model(tmp_path, 'm', params={**COMBINED_PARAMS, 'social_distance': ''})
	_assert_incomplete(_only(tmp_path), 'social_distance is blank')


def test_social_distance_not_required_for_other_modes(tmp_path):
	for mode in ('0', '1', '3'):
		params = _without({**PATTERN_PARAMS, 'behavior_kind': mode}, 'social_distance')
		_make_model(tmp_path, f'mode{mode}', params=params)
	for info in cl.discover_categorizers(tmp_path):
		assert info.is_available, info.problems
		assert info.social_distance is None


def test_interactive_advanced_without_color_code_is_available(tmp_path):
	# LabGym falls back to a default color mode when color_code is missing.
	_make_model(tmp_path, 'm', params=_without(COMBINED_PARAMS, 'color_code'))
	info = _only(tmp_path)
	assert info.is_available, info.problems
	assert info.color_code is None


def test_unknown_network_value(tmp_path):
	_make_model(tmp_path, 'm', params={**PATTERN_PARAMS, 'network': '7'})
	info = _only(tmp_path)
	_assert_incomplete(info, 'network has an unknown value 7')
	assert info.network_type is None


def test_unknown_behavior_mode(tmp_path):
	_make_model(tmp_path, 'm', params={**PATTERN_PARAMS, 'behavior_kind': '9'})
	info = _only(tmp_path)
	_assert_incomplete(info, 'behavior_kind has an unknown value 9')
	assert info.behavior_mode is None


def test_blank_behavior_mode(tmp_path):
	_make_model(tmp_path, 'm', params={**PATTERN_PARAMS, 'behavior_kind': ''})
	info = _only(tmp_path)
	_assert_incomplete(info, 'behavior_kind is blank')
	assert info.behavior_mode is None


def test_non_numeric_value(tmp_path):
	_make_model(tmp_path, 'm', params={**PATTERN_PARAMS, 'dim_conv': 'big'})
	info = _only(tmp_path)
	_assert_incomplete(info, "dim_conv has an invalid value 'big'")
	assert info.dim_conv is None


def test_float_values(tmp_path):
	_make_model(tmp_path, 'ok', params={**PATTERN_PARAMS, 'dim_conv': '32.0'})
	_make_model(tmp_path, 'bad', params={**PATTERN_PARAMS, 'dim_conv': '32.5'})
	bad, ok = cl.discover_categorizers(tmp_path)
	assert ok.is_available and ok.dim_conv == 32
	_assert_incomplete(bad, "dim_conv has an invalid value '32.5'")


def test_blank_and_duplicate_classnames(tmp_path):
	_make_model(tmp_path, 'blank', ('a', ''), PATTERN_PARAMS)
	_make_model(tmp_path, 'dup', ('a', 'a'), PATTERN_PARAMS)
	blank, dup = cl.discover_categorizers(tmp_path)
	_assert_incomplete(blank, 'class names are blank')
	_assert_incomplete(dup, 'class names are duplicated')


def test_incomplete_folder_does_not_hide_others(tmp_path):
	_make_model(tmp_path, 'broken', params_text='garbage')
	_make_model(tmp_path, 'good', params=ANIMATION_PARAMS)
	broken, good = cl.discover_categorizers(tmp_path)
	assert broken.status == cl.STATUS_INCOMPLETE
	assert good.status == cl.STATUS_AVAILABLE


# Display text for the main list and the details.

def _details(info):
	return dict(cl.detail_rows(info))


def test_list_row_has_no_architecture_columns(tmp_path):
	_make_model(tmp_path, 'combo', ('p', 'q', 'r', 's'), COMBINED_PARAMS)
	assert cl.list_row(_only(tmp_path)) == (
		'combo', 'Combined', '4 classes: p, q, r, ...', 'Interactive advanced', 'Available')


def test_list_row_for_unrelated_folder(tmp_path):
	(tmp_path / 'notes').mkdir()
	assert cl.list_row(_only(tmp_path)) == (
		'notes', 'Unknown', 'No classes recorded', 'Unknown', 'Incomplete')


def test_pattern_details_show_only_pattern_architecture(tmp_path):
	_make_model(tmp_path, 'pat', ('groom', 'rear'), PATTERN_PARAMS)
	d = _details(_only(tmp_path))
	assert d['Pattern Recognizer dimension'] == '32'
	assert d['Pattern Recognizer level'] == '3'
	assert 'Animation Analyzer dimension' not in d
	assert 'Animation Analyzer level' not in d
	assert d['Behavior classes (2)'] == 'groom\nrear'
	assert d['Time step'] == '15 frames'
	assert d['Channel'] == '1'
	assert d['Body parts'] == 'Not included'
	assert d['Background'] == 'Included'
	assert 'Social distance' not in d


def test_animation_details_show_only_animation_architecture(tmp_path):
	_make_model(tmp_path, 'anim', params=ANIMATION_PARAMS)
	d = _details(_only(tmp_path))
	assert d['Animation Analyzer dimension'] == '16'
	assert d['Animation Analyzer level'] == '2'
	assert 'Pattern Recognizer dimension' not in d
	assert d['Background'] == 'Excluded (white background)'


def test_combined_details(tmp_path):
	folder = _make_model(tmp_path, 'combo', params=COMBINED_PARAMS, metrics=True)
	d = _details(_only(tmp_path))
	assert d['Pattern Recognizer dimension'] == '64'
	assert d['Pattern Recognizer level'] == '4'
	assert d['Animation Analyzer dimension'] == '64'
	assert d['Animation Analyzer level'] == '4'
	assert d['Body parts'] == 'Included (STD 50)'
	assert d['Background'] == 'Excluded (black background)'
	assert d['Social distance'] == '3'
	assert d['Supporting characters'] == 'Grayscale'
	assert d['Location'] == os.path.abspath(folder)
	assert d['Training metrics'] == cl.TRAINING_METRICS_FILENAME
	assert d['Training history'] == 'Not found'


def test_details_never_claim_compatibility_or_version(tmp_path):
	_make_model(tmp_path, 'm', params=PATTERN_PARAMS)
	d = _details(_only(tmp_path))
	assert d['Status'] == 'Available (files present; compatibility not checked)'
	assert d['LabGym version'] == 'Not recorded'
	assert 'compatib' in cl.AVAILABLE_MEANING


def test_incomplete_details_list_problems(tmp_path):
	_make_model(tmp_path, 'm', params=PATTERN_PARAMS, weights=False)
	d = _details(_only(tmp_path))
	assert d['Status'] == 'Incomplete'
	assert 'saved_model.pb not found' in d['Problems']
	assert 'variables/ folder not found' in d['Problems']


def test_unrecorded_defaults_are_labelled(tmp_path):
	params = _without(_without(COMBINED_PARAMS, 'color_code'), 'black_background')
	_make_model(tmp_path, 'adv', params=params)
	_make_model(tmp_path, 'old', params=_without(PATTERN_PARAMS, 'behavior_kind'))
	adv, old = cl.discover_categorizers(tmp_path)
	assert _details(adv)['Supporting characters'] == 'Grayscale (not recorded; LabGym default)'
	assert _details(adv)['Background'] == 'Excluded'
	assert _details(old)['Behavior mode'] == 'Non-interactive (not recorded; LabGym default)'


def test_social_distance_zero_means_no_limit(tmp_path):
	_make_model(tmp_path, 'm', params={**COMBINED_PARAMS, 'social_distance': '0'})
	assert _details(_only(tmp_path))['Social distance'] == '0 (no limit)'


def test_format_details_indents_multiline_values(tmp_path):
	_make_model(tmp_path, 'm', ('a', 'b'), PATTERN_PARAMS)
	text = cl.format_details(_only(tmp_path))
	assert 'Behavior classes (2):\n    a\n    b\n' in text
	assert 'Name: m\n' in text


# File actions.

import shutil
import subprocess

import pytest


@pytest.mark.parametrize('name, fragment', [
	('', 'Enter a name'),
	('   ', 'Enter a name'),
	(' lead', 'start or end with a space'),
	('trail ', 'start or end with a space'),
	('m', 'same as the current name'),
	('a/b', 'cannot contain / or \\'),
	('a\\b', 'cannot contain / or \\'),
	('a:b', 'cannot contain any of these characters'),
	('a*b', 'cannot contain any of these characters'),
	('a?b', 'cannot contain any of these characters'),
	('tab\there', 'cannot contain any of these characters'),
	('.', 'cannot start with a period'),
	('..', 'cannot start with a period'),
	('.hidden', 'cannot start with a period'),
	('dot.', 'cannot end with a period'),
	('__pycache__', 'reserved and cannot be used'),
	('CON', 'reserved name on Windows'),
	('nul', 'reserved name on Windows'),
	('com1.txt', 'reserved name on Windows'),
	('LPT9', 'reserved name on Windows'),
	('x' * 256, 'too long'),
])
def test_rename_rejects_invalid_names(tmp_path, name, fragment):
	_make_model(tmp_path, 'm', params=PATTERN_PARAMS)
	assert fragment in cl.validate_new_name(tmp_path, 'm', name)
	with pytest.raises(cl.LibraryActionError, match='.'):
		cl.rename_categorizer(tmp_path, 'm', name)
	assert (tmp_path / 'm').is_dir()


def test_rename_accepts_ordinary_names(tmp_path):
	_make_model(tmp_path, 'm', params=PATTERN_PARAMS)
	for name in ('mouse grooming v2', 'café_model', 'CONSOLE', 'a.b'):
		assert cl.validate_new_name(tmp_path, 'm', name) is None


def test_rename_success(tmp_path):
	_make_model(tmp_path, 'old', params=PATTERN_PARAMS)
	new_path = cl.rename_categorizer(tmp_path, 'old', 'new')
	assert new_path == os.path.join(os.path.abspath(tmp_path), 'new')
	assert [i.name for i in cl.discover_categorizers(tmp_path)] == ['new']
	assert _only(tmp_path).is_available


def test_rename_collision_leaves_both_folders(tmp_path):
	_make_model(tmp_path, 'a', params=PATTERN_PARAMS)
	_make_model(tmp_path, 'b', params=ANIMATION_PARAMS)
	with pytest.raises(cl.LibraryActionError, match='"b" already exists'):
		cl.rename_categorizer(tmp_path, 'a', 'b')
	a, b = cl.discover_categorizers(tmp_path)
	assert (a.network_type, b.network_type) == ('Pattern', 'Animation')


def test_rename_collision_is_case_insensitive(tmp_path):
	_make_model(tmp_path, 'a', params=PATTERN_PARAMS)
	_make_model(tmp_path, 'Mouse', params=PATTERN_PARAMS)
	assert 'already exists' in cl.validate_new_name(tmp_path, 'a', 'mouse')


def test_rename_collision_with_file_or_junk_folder(tmp_path):
	_make_model(tmp_path, 'a', params=PATTERN_PARAMS)
	(tmp_path / 'notes.txt').write_text('', encoding='utf-8')
	assert 'already exists' in cl.validate_new_name(tmp_path, 'a', 'notes.txt')


def test_rename_case_only_change_allowed(tmp_path):
	_make_model(tmp_path, 'mouse', params=PATTERN_PARAMS)
	assert cl.validate_new_name(tmp_path, 'mouse', 'Mouse') is None
	cl.rename_categorizer(tmp_path, 'mouse', 'Mouse')
	assert [i.name for i in cl.discover_categorizers(tmp_path)] == ['Mouse']


def test_rename_missing_source(tmp_path):
	with pytest.raises(cl.LibraryActionError, match='no longer exists'):
		cl.rename_categorizer(tmp_path, 'gone', 'new')


def _tree(folder):
	return sorted(str(p.relative_to(folder)) for p in Path(folder).rglob('*'))


def test_export_copies_model(tmp_path):
	src = _make_model(tmp_path / 'models', 'm', params=COMBINED_PARAMS, metrics=True)
	dest = tmp_path / 'dest'
	dest.mkdir()
	target = cl.export_categorizer(src, dest)
	assert target == str(dest / 'm')
	assert _tree(target) == _tree(src)
	assert cl.read_categorizer(target).is_available
	# no staging folder left behind
	assert os.listdir(dest) == ['m']


def test_export_never_overwrites_or_merges(tmp_path):
	src = _make_model(tmp_path / 'models', 'm', params=COMBINED_PARAMS)
	dest = tmp_path / 'dest'
	existing = dest / 'm'
	existing.mkdir(parents=True)
	(existing / 'keep.txt').write_text('mine', encoding='utf-8')
	with pytest.raises(cl.LibraryActionError, match='already contains a folder named "m"'):
		cl.export_categorizer(src, dest)
	assert _tree(existing) == ['keep.txt']
	assert (existing / 'keep.txt').read_text(encoding='utf-8') == 'mine'
	assert os.listdir(dest) == ['m']


def test_export_refuses_empty_existing_folder_and_file(tmp_path):
	src = _make_model(tmp_path / 'models', 'm', params=PATTERN_PARAMS)
	for i, make in enumerate((lambda p: p.mkdir(), lambda p: p.write_text(''))):
		dest = tmp_path / f'dest{i}'
		dest.mkdir()
		make(dest / 'm')
		with pytest.raises(cl.LibraryActionError, match='already contains'):
			cl.export_categorizer(src, dest)


def test_export_collision_is_case_insensitive(tmp_path):
	src = _make_model(tmp_path / 'models', 'Mouse', params=PATTERN_PARAMS)
	dest = tmp_path / 'dest'
	(dest / 'mouse').mkdir(parents=True)
	with pytest.raises(cl.LibraryActionError, match='already contains a folder named "mouse"'):
		cl.export_categorizer(src, dest)


def test_failed_export_leaves_nothing(tmp_path, monkeypatch):
	src = _make_model(tmp_path / 'models', 'm', params=COMBINED_PARAMS)
	dest = tmp_path / 'dest'
	dest.mkdir()
	real_copytree = shutil.copytree

	def copy_then_fail(s, d, *args, **kwargs):
		real_copytree(s, d, *args, **kwargs)  # a full copy exists in staging...
		raise OSError(28, 'No space left on device')  # ...then the copy fails

	monkeypatch.setattr(cl.shutil, 'copytree', copy_then_fail)
	with pytest.raises(cl.LibraryActionError, match='could not be exported; nothing was kept'):
		cl.export_categorizer(src, dest)
	assert os.listdir(dest) == []


def test_export_collision_appearing_during_copy(tmp_path, monkeypatch):
	src = _make_model(tmp_path / 'models', 'm', params=PATTERN_PARAMS)
	dest = tmp_path / 'dest'
	dest.mkdir()
	real_copytree = shutil.copytree

	def copy_and_race(s, d, *args, **kwargs):
		real_copytree(s, d, *args, **kwargs)
		if not (dest / 'm').exists():  # copytree recurses through this patch
			(dest / 'm').mkdir()
		(dest / 'm' / 'other.txt').write_text('other', encoding='utf-8')

	monkeypatch.setattr(cl.shutil, 'copytree', copy_and_race)
	with pytest.raises(cl.LibraryActionError, match='already contains'):
		cl.export_categorizer(src, dest)
	assert _tree(dest) == ['m', os.path.join('m', 'other.txt')]


def test_export_incomplete_model_copies_as_is(tmp_path):
	# The dialog warns first; the copy itself does not repair anything.
	src = _make_model(tmp_path / 'models', 'm', params=PATTERN_PARAMS, weights=False)
	dest = tmp_path / 'dest'
	dest.mkdir()
	target = cl.export_categorizer(src, dest)
	assert cl.read_categorizer(target).status == cl.STATUS_INCOMPLETE


def test_export_into_own_folder_refused(tmp_path):
	src = _make_model(tmp_path, 'm', params=PATTERN_PARAMS)
	for dest in (src, src / 'variables'):
		with pytest.raises(cl.LibraryActionError, match='into its own folder'):
			cl.export_categorizer(src, dest)


def test_export_missing_source_or_destination(tmp_path):
	src = _make_model(tmp_path, 'm', params=PATTERN_PARAMS)
	with pytest.raises(cl.LibraryActionError, match='no longer exists'):
		cl.export_categorizer(tmp_path / 'gone', tmp_path)
	with pytest.raises(cl.LibraryActionError, match='destination folder does not exist'):
		cl.export_categorizer(src, tmp_path / 'nowhere')


def test_reveal_command_per_platform():
	assert cl.reveal_command('/m/a', 'darwin') == ['open', '-R', '/m/a']
	assert cl.reveal_command('/m/a', 'win32')[:2] == ['explorer', '/select,']
	assert cl.reveal_command('/m/a', 'linux') == ['xdg-open', '/m']


def test_reveal_location(tmp_path, monkeypatch):
	# Never launch a real file manager from the tests.
	calls = []
	monkeypatch.setattr(cl.subprocess, 'Popen', lambda cmd: calls.append(cmd))
	cl.reveal_location(str(tmp_path))
	assert calls == [cl.reveal_command(str(tmp_path))]
	with pytest.raises(cl.LibraryActionError, match='no longer exists'):
		cl.reveal_location(str(tmp_path / 'gone'))


def test_reveal_location_without_file_manager(tmp_path, monkeypatch):
	def missing(cmd):
		raise FileNotFoundError(2, 'No such file', cmd[0])
	monkeypatch.setattr(cl.subprocess, 'Popen', missing)
	with pytest.raises(cl.LibraryActionError, match='The location is:'):
		cl.reveal_location(str(tmp_path))

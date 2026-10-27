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

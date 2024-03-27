import re

import pandas as pd
import dataset.logic2text.logical_form.APIs as legacy_api
from datasource.logic2text.model_utils import get_tokenizer

# Thresholds
ALL = 1  # 100%
MOST = 3  # 30%

# Extrema types
MAX = 'max'
MIN = 'min'

api = {
	'count': {
		'argument': ['row'],
		'output': 'num',
		'function': lambda t: len(t),
		'tostr': lambda t: 'count {{ {} }}'.format(t),
		'append': True
	},
	'only': {
		'argument': ['row'],
		'output': 'bool',
		'function': lambda t: len(t) == 1,
		'tostr': lambda t: 'only {{ {} }}'.format(t),
		'append': None
	},
	'hop': {
		'argument': ['row', 'header'],
		'output': 'obj',
		'function': lambda t, col: fn_hop(t, col),
		'tostr': lambda t, col: 'hop {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'hop_first': {
		'argument': ['row', 'header'],
		'output': 'obj',
		'function': lambda t, col: fn_hop(t, col),
		'tostr': lambda t, col: 'hop_first {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'avg': {
		'argument': ['row', 'header'],
		'output': 'num',
		'function': lambda t, col: fn_agg(t, col, 'avg'),
		'tostr': lambda t, col: 'avg {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'sum': {
		'argument': ['row', 'header'],
		'output': 'num',
		'function': lambda t, col: fn_agg(t, col, 'sum'),
		'tostr': lambda t, col: 'sum {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'max': {
		'argument': ['row', 'header'],
		'output': 'obj',
		'function': lambda t, col: fn_extrema_val(t, col, extrema_type=MAX),
		'tostr': lambda t, col: 'max {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'min': {
		'argument': ['row', 'header'],
		'output': 'obj',
		'function': lambda t, col: fn_extrema_val(t, col, extrema_type=MIN),
		'tostr': lambda t, col: 'min {{ {} ; {} }}'.format(t, col),
		'append': True
	},
	'argmax': {
		'argument': ['row', 'header'],
		'output': 'row',
		'function': lambda t, col: fn_extrema_row(t, col, extrema_type=MAX),
		'tostr': lambda t, col: 'argmax {{ {} ; {} }}'.format(t, col),
		'append': False
	},
	'argmin': {
		'argument': ['row', 'header'],
		'output': 'row',
		'function': lambda t, col: fn_extrema_row(t, col, extrema_type=MIN),
		'tostr': lambda t, col: 'argmin {{ {} ; {} }}'.format(t, col),
		'append': False
	},
	'nth_argmax': {
		'argument': ['row', 'header', 'num'],
		'output': 'row',
		'function': lambda t, col, ind: fn_extrema_row(t, col, order=ind, extrema_type=MAX),
		'tostr': lambda t, col, ind: 'nth_argmax {{ {} ; {} ; {} }}'.format(t, col, ind),
		'append': False
	},
	'nth_argmin': {
		'argument': ['row', 'header', 'num'],
		'output': 'row',
		'function': lambda t, col, ind: fn_extrema_row(t, col, order=ind, extrema_type=MIN),
		'tostr': lambda t, col, ind: 'nth_argmin {{ {} ; {} ; {} }}'.format(t, col, ind),
		'append': False
	},
	'nth_max': {
		'argument': ['row', 'header', 'num'],
		'output': 'num',
		'function': lambda t, col, ind: fn_extrema_val(t, col, order=ind, extrema_type=MAX),
		'tostr': lambda t, col, ind: 'nth_max {{ {} ; {} ; {} }}'.format(t, col, ind),
		'append': True
	},
	'nth_min': {
		'argument': ['row', 'header', 'num'],
		'output': 'num',
		'function': lambda t, col, ind: fn_extrema_val(t, col, order=ind, extrema_type=MIN),
		'tostr': lambda t, col, ind: 'nth_min {{ {} ; {} ; {} }}'.format(t, col, ind),
		'append': True
	},
	'diff': {
		'argument': ['obj', 'obj'],
		'output': 'str',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="diff"),
		'tostr': lambda t1, t2: 'diff {{ {} ; {} }}'.format(t1, t2),
		'append': True
	},
	'greater': {
		'argument': ['obj', 'obj'],
		'output': 'bool',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="greater"),
		'tostr': lambda t1, t2: 'greater {{ {} ; {} }}'.format(t1, t2),
		'append': False
	},
	'less': {
		'argument': ['obj', 'obj'],
		'output': 'bool',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="less"),
		'tostr': lambda t1, t2: 'less {{ {} ; {} }}'.format(t1, t2),
		'append': True
	},
	'eq': {
		'argument': ['obj', 'obj'],
		'output': 'bool',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="eq"),
		'tostr': lambda t1, t2: 'eq {{ {} ; {} }}'.format(t1, t2),
		'append': None
	},
	'not_eq': {
		'argument': ['obj', 'obj'],
		'output': 'bool',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="eq", negated=True),
		'tostr': lambda t1, t2: 'not_eq {{ {} ; {} }}'.format(t1, t2),
		'append': None
	},
	'round_eq': {
		'argument': ['obj', 'obj'],
		'output': 'bool',
		'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="round_eq"),
		'tostr': lambda t1, t2: 'round_eq {{ {} ; {} }}'.format(t1, t2),
		'append': None
	},
	'and': {
		'argument': ['bool', 'bool'],
		'output': 'bool',
		'function': lambda t1, t2: t1 and t2,
		'tostr': lambda t1, t2: 'and {{ {} ; {} }}'.format(t1, t2),
		'append': None
	},
	'filter_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='eq'),
		'tostr': lambda t, col, value: 'filter_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_not_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='not_eq'),
		'tostr': lambda t, col, value: 'filter_not_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_less': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='less'),
		'tostr': lambda t, col, value: 'filter_less {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_greater': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='greater'),
		'tostr': lambda t, col, value: 'filter_greater {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_greater_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='greater_eq'),
		'tostr': lambda t, col, value: 'filter_greater_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_less_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='less_eq'),
		'tostr': lambda t, col, value: 'filter_less_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_all': {
		'argument': ['row', 'header'],
		'output': 'row',
		'function': lambda t, col: t,
		'tostr': lambda t, col: 'filter_all {{ {} ; {} }}'.format(t, col),
		'append': False
	},
	'all_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='eq'),
		'tostr': lambda t, col, value: 'all_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'all_not_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='not_eq'),
		'tostr': lambda t, col, value: 'all_not_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'all_less': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='less'),
		'tostr': lambda t, col, value: 'all_less {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'all_less_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='less_eq'),
		'tostr': lambda t, col, value: 'all_less_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'all_greater': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='greater'),
		'tostr': lambda t, col, value: 'all_greater {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'all_greater_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='greater_eq'),
		'tostr': lambda t, col, value: 'all_greater_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='eq'),
		'tostr': lambda t, col, value: 'most_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_not_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='not_eq'),
		'tostr': lambda t, col, value: 'most_not_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_less': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='less'),
		'tostr': lambda t, col, value: 'most_less {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_less_eq': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='less_eq'),
		'tostr': lambda t, col, value: 'most_less_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_greater': {
		'argument': ['row', 'header', 'obj'],
		'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='greater'),
		'tostr': lambda t, col, value: 'most_greater {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	'most_greater_eq': {
		'argument': ['row', 'header', 'obj'], 'output': 'bool',
		'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='greater_eq'),
		'tostr': lambda t, col, value: 'most_greater_eq {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': None
	},
	# New
	'contains': {
		'argument': ['obj', 'obj'],
			'output': 'bool',
			'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="contains"),
			'tostr': lambda t1, t2: 'contains {{ {} ; {} }}'.format(t1, t2),
			'append': None
	},
	'not_contains': {
		'argument': ['obj', 'obj'],
			'output': 'bool',
			'function': lambda t1, t2: fn_compare(t1, t2, comparison_type="contains", negated=True),
			'tostr': lambda t1, t2: 'not_contains {{ {} ; {} }}'.format(t1, t2),
			'append': None
	},
	'all_contains': {
		'argument': ['row', 'header', 'obj'],
			'output': 'bool',
			'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='contains'),
			'tostr': lambda t, col, value: 'all_contains {{ {} ; {} ; {} }}'.format(t, col, value),
			'append': None
	},
	'all_not_contains': {
		'argument': ['row', 'header', 'obj'],
			'output': 'bool',
			'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=ALL, filter_type='not_contains'),
			'tostr': lambda t, col, value: 'all_not_contains {{ {} ; {} ; {} }}'.format(t, col, value),
			'append': None
	},
	'most_contains': {
		'argument': ['row', 'header', 'obj'],
			'output': 'bool',
			'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='contains'),
			'tostr': lambda t, col, value: 'most_contains {{ {} ; {} ; {} }}'.format(t, col, value),
			'append': None
	},
	'most_not_contains': {
		'argument': ['row', 'header', 'obj'],
			'output': 'bool',
			'function': lambda t, col, value: filter_meets_amount(t, col, value, threshold=MOST, filter_type='not_contains'),
			'tostr': lambda t, col, value: 'most_not_contains {{ {} ; {} ; {} }}'.format(t, col, value),
			'append': None
	},
	'filter_contains': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='contains'),
		'tostr': lambda t, col, value: 'filter_contains {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'filter_not_contains': {
		'argument': ['row', 'header', 'obj'],
		'output': 'row',
		'function': lambda t, col, value: fn_filter(t, col, value, filter_type='not_contains'),
		'tostr': lambda t, col, value: 'filter_not_contains {{ {} ; {} ; {} }}'.format(t, col, value),
		'append': False
	},
	'substring': {
		'argument': ['obj', 'num'],
		'output': 'str',
		'function': lambda obj, i1: fn_substring(obj, i1),
		'tostr': lambda obj, i1: 'substring {{ {} ; {} }}'.format(obj, i1),
		'append': True
	},
	'substring_range': {
		'argument': ['obj', 'num', 'num'],
		'output': 'str',
		'function': lambda obj, i1, i2: fn_substring_range(obj, i1, i2),
		'tostr': lambda obj, i1, i2: 'substring {{ {} ; {} ; {} }}'.format(obj, i1, i2),
		'append': True
	},
}

class LFExecutionError(Exception):
	def __init__(self, message="LF execution error"):
		self.message = message

def fn_substring(obj:str, i1:int) -> str:
	#return obj[int(i1):int(i2)]
	return fn_substring_range(obj, i1, int(i1))

def fn_substring_range(obj:str, i1:int, i2:int) -> str:
	#return obj[int(i1):int(i2)]
	tokenizer = get_tokenizer()
	obj_tokens = tokenizer.tokenize(obj)
	# i2 +1 because grammar uses two inclusive boundaries [] while substring range in python uses [)
	return tokenizer.convert_tokens_to_string(obj_tokens[int(i1):int(i2)+1])

def fn_filter(df:pd.DataFrame, col:str, value:str, filter_type:str) -> pd.DataFrame:
	"""
	Returns the table filtered with the specified criteria
	:param df:
	:param col:
	:param value:
	:param filter_type:
	:return:
	"""
	res = None
	if filter_type == "greater":
		# Not yet reimplemented, use original Logic2Text APIs
		res = legacy_api.fuzzy_compare_filter(df, col, value, type=filter_type)
	elif filter_type == "greater_eq":
		# Not yet reimplemented, use original Logic2Text APIs
		res = legacy_api.fuzzy_compare_filter(df, col, value, type=filter_type)
	elif filter_type == "less":
		# Not yet reimplemented, use original Logic2Text APIs
		res = legacy_api.fuzzy_compare_filter(df, col, value, type=filter_type)
	elif filter_type == "less_eq":
		# Not yet reimplemented, use original Logic2Text APIs
		res = legacy_api.fuzzy_compare_filter(df, col, value, type=filter_type)
	elif filter_type == "eq":
		res = df[df[col]==value]
	elif filter_type == "not_eq":
		res = df[df[col]!=value]
	elif filter_type == "contains":
		res = df[df[col].str.contains(re.escape(value))]
	elif filter_type == "not_contains":
		res = df[~df[col].str.contains(re.escape(value))]

	if res is None:
		raise LFExecutionError("Filter returned a None DataFrame")

	return res.reset_index(drop=True)

def fn_compare(obj1:str, obj2:str, comparison_type:str, negated:bool=False, tolerance:float=0.0):
	"""
	Returns the comparison of obj1 comparison_type obj2
	:param obj1:
	:param obj2:
	:param comparison_type:
	:param negated whether this comparison is negated or not
	:param tolerance: during round comparisons, the specified tolerance for ==
	:return: bool whether obj1 is comparison_type obj2, str when diff
	"""
	if comparison_type == "eq":
		result = str(obj1) == str(obj2)
	elif comparison_type == "contains":
		result = str(obj2) in str(obj1)
	elif comparison_type == "round_eq":
		# Deprecated, this should be removed after certifying that it is no longer needed
		result = str(obj1) == str(obj2)
	elif comparison_type == "greater":
		# Not yet reimplemented, use original Logic2Text APIs
		result = legacy_api.obj_compare(obj1, obj2, round=tolerance!=0.0, type='greater')
	elif comparison_type == "less":
		# Not yet reimplemented, use original Logic2Text APIs
		result = legacy_api.obj_compare(obj1, obj2, round=tolerance!=0.0, type='less')
	elif comparison_type == "diff":
		# Not yet reimplemented, use original Logic2Text APIs
		result = legacy_api.obj_compare(obj1, obj2, round=tolerance!=0.0, type='diff')
	else:
		raise LFExecutionError("Compare function '{}' not implemented".format(comparison_type))

	return not result if negated else result

def fn_extrema_row(df:pd.DataFrame, col:str, extrema_type:str, order:int=1) -> pd.DataFrame:
	"""
	Returns the row with the nth max/min value for a certain col
	:param df:
	:param col: the values of this column will be ranked
	:param order: The nth value, being 1 the most extreme
	:param extrema_type: max | min
	:return: the row with the value that meets this extrema
	"""
	if extrema_type == MAX:
		# Not yet reimplemented, use original Logic2Text APIs
		return legacy_api.nth_maxmin(df, col, order=order, max_or_min="max", arg=True)
	elif extrema_type == MIN:
		# Not yet reimplemented, use original Logic2Text APIs
		return legacy_api.nth_maxmin(df, col, order=order, max_or_min="min", arg=True)
	else:
		raise LFExecutionError("Extrema type '{}' not implemented".format(extrema_type))

def fn_extrema_val(df:pd.DataFrame, col:str, extrema_type:str, order:int=1) -> str:
	"""
	Returns the actual value for the col of the row with the nth max/min row for that given col
	:param df:
	:param col: the values of this column will be ranked and returned
	:param order: The nth value, being 1 the most extreme
	:param extrema_type: max | min
	:return: the value that meets this extrema
	"""
	return fn_extrema_row(df, col, extrema_type, order)[col].values[0]

def fn_agg(df:pd.DataFrame, col:str, agg_type:str) -> str:
	"""
	performs an aggregation over the values of a given column
	:param df:
	:param col:
	:param agg_type: sum | avg
	:return: the result of the aggregation
	"""
	if agg_type == 'sum':
		# Not yet reimplemented, use original Logic2Text APIs
		return legacy_api.agg(df, col, 'sum')
	elif agg_type == 'avg':
		# Not yet reimplemented, use original Logic2Text APIs
		return legacy_api.agg(df, col, 'mean')
	else:
		raise LFExecutionError("Aggregation operation '{}' not implemented".format(agg_type))

def fn_hop(df:pd.DataFrame, col:str) -> str:
	"""
	Returns the value of the col for the first row of a given DataFrame
	:param df:
	:param col:
	:return:
	"""
	if len(df) == 0:
		raise LFExecutionError("Attempting to performa hop operation over an empty DataFrame")

	return df[col].values[0]


def filter_meets_amount(df:pd.DataFrame, col:str, value:str, threshold:int, filter_type:str) -> bool:
	if len(df) == 0:
		# If the original table is empty the statements is false
		return False

	return (len(df) // threshold) <= len(fn_filter(df, col, value, filter_type))

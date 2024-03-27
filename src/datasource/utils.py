import pandas as pd

# Table2Logic table separators
TABLE_VALUE_SEP = ";"
TABLE_IDX_SEP = ":"
SEGMENT_ID_TABLE_TOPIC = 0
SEGMENT_ID_SCHEMA = 1

def split_into_tokens(text, tokenizer):
	"""
	This is a more sophisticated way of splitting a string into tokens than using split by ' '. It mimics the token split
	that the subword tokenizer would do. We merge part '##' tokens into the same token. This is commonly used for
	spliting the topìc into tokens
	:param text:
	:return:
	"""
	all_sub_token = tokenizer.tokenize(text)

	no_subword_tokens = []

	for sub_token in all_sub_token:
		if len(sub_token) > 2 and sub_token[0:2] == '##':
			no_subword_tokens[-1] += sub_token[2:]
		else:
			no_subword_tokens.append(sub_token)
	return no_subword_tokens


def build_table(table_header, table_content, contains_row_index=False):
	"""
	Builds a pandas DataFrame out of table information
	:param table_header: columns of the table
	:param table_content: table in a list of lists format [['row 0', 'cell11', 'cell12'],['row 1', 'cell21', 'cell22'], ..]
	:param contains_row_index: whether the table_content has or not the 'row n' cell before each column. This won't be in the resulting DataFrame
	:return:
	"""

	"""
	# Legacy

	for ind, header in enumerate(table_header):
		for inr, row in enumerate(table_cont):

			# remove last summarization row
			if inr == len(table_cont) - 1 \
					and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
						 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
				continue
			pd_in[header].append(row[ind])

	pd_table = pd.DataFrame(pd_in)
	"""
	if contains_row_index and table_content is not None:
		table_header = ["logic2text_row_index"] + table_header if table_header is not None else table_header
		df = pd.DataFrame(table_content, columns=table_header)
		df = df.drop(df.columns[[0]], axis=1)
	else:
		df = pd.DataFrame(table_content, columns=table_header)
	# Logic2Text dataset was originally thought to drop any last row representing any kind of summarization.
	# I'm not much of a fan of doing this but if it's not done some of the dataset's LFs won't execute to True
	drop_last_if_sum(df)
	return df


def build_table_from_df(df: pd.DataFrame, add_row_idx: bool = True):
	"""
	Opposite of build_table. Builds table information out of a pandas DataFrame
	:param df:
	:param add_row_idx:
	:return: table in a list of lists format [['row 0', 'cell11', 'cell12'],['row 1', 'cell21', 'cell22'], ..]
	"""
	table = df.values.tolist()
	if add_row_idx:
		table = [['row {}'.format(n)] + row for n, row in enumerate(table)]
	return table


def table2logic_table_linearization(table, tokenizer, has_row_idx: bool = True) -> str:
	"""
	Linearize table in the same way as it's done in Table2Logic
	:param table:
	:param tokenizer:
	:param has_row_idx:
	:return:
	"""
	linearized_table = ""

	for row in table:
		for i, cell in enumerate(row):
			linearized_cell = cell
			# if row n add : if add ; to separate between values or SEP_TOKEN at the end of the row
			if i == 0 and has_row_idx:
				# row 0 :
				linearized_cell += TABLE_IDX_SEP
			elif i+1 < len(row) or (i == 0 and not has_row_idx):
				# value ;
				linearized_cell += TABLE_VALUE_SEP
			else:
				# last_value_of_row SEP
				# get str version of sep not actual id
				linearized_cell += tokenizer.sep_token if tokenizer.sep_token is not None else "<row>"

			linearized_table += linearized_cell

	return linearized_table

def tokenize_column_names(column_names, tokenizer):
	linearized_columns = ""

	for column in column_names:
		column_sub_tokens = column

		column_sub_tokens += tokenizer.sep_token

		linearized_columns += column_sub_tokens

	return linearized_columns

def drop_last_if_sum(df: pd.DataFrame):
	"""
	Drops the last row if it is som kind of summarization result of the previous rows
	Logic2Text dataset was originally thought to drop any last row representing any kind of summarization.
	I'm not much of a fan of doing this but if it's not done some of the dataset's LFs won't execute to True
	:param df:
	:return:
	"""
	if len(df) == 0:
		return
	cell = df.iloc[-1, 0]
	if "all" in cell or "total" in cell or "sum" in cell or "a l l" in cell or "t o t a l" in cell or "s u m" in cell:
		df.drop(df.tail(1).index, inplace=True)

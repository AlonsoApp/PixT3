import re
from typing import List, Dict

import pandas as pd

from datasource.logic2text.model_utils import get_tokenizer
from datasource.logic2text.logical_form.lf_engine import api
from datasource.logic2text.logical_form.lf_grammar_versions import changes
from datasource.logic2text.logical_form.lf_grammar import *
from treelib import Tree as TreePlt
from zss import Node as ZSSNode

MASKED_TOKEN = "[MASKED]"

class ASTNode(object):
	def __init__(self, body: str, args: [], action: Action = None, masked: bool = False):
		"""
		:param body:
		:param args:
		:param action:
		"""
		self.body:str = body
		self.action:Action = action
		self.args:List = args
		self.masked:bool = masked

	@classmethod
	def from_logic_str(cls, func: str, args: []):
		if len(args) == 0:
			# This protects the parser against grammar keywords in leaf nodes
			return ASTNode(func, [])
		action = cls._action(func)
		node_args: List[ASTNode] = []
		for arg in args:
			if type(arg) is dict:
				arg_fn = next(iter(arg))
				node_args.append(ASTNode.from_logic_str(arg_fn, arg[arg_fn]))
			else:
				node_args.append(ASTNode.from_logic_str(arg, []))

		return ASTNode(func, node_args, action)

	@classmethod
	def from_action_list(cls, actions: List[Action], columns: List[str]=None, ordinals: List[str]=None, values: List[str]=None, padding=False):
		actions = actions.copy()
		return ASTNode._req_from_action_list(actions, columns, ordinals, values, padding=padding)

	@staticmethod
	def _req_from_action_list(actions: List[Action], columns: List[str]=None, ordinals: List[str]=None, values: List[str]=None, padding=False):
		"""
		:param actions:
		:param columns:
		:param ordinals:
		:param values:
		:param padding: if True we expect only sketch actions and we add the corresponding terminal action based on the
		expected next action. We don't propagate the recursion further because it is a terminal action
		:return:
		"""
		action = actions.pop(0)
		# Padding
		next_actions = action.get_next_action()
		args = []
		for next_action in next_actions:
			if padding and next_action in Grammar.terminal_actions():
				args.append(ASTNode(body='', args=[], action=next_action(0)))
			else:
				node = ASTNode._req_from_action_list(actions, columns, ordinals, values, padding)
				args.append(node)
		# body
		body = ''
		masked = False
		if type(action) in Grammar.terminal_actions():
			if type(action) == C and columns is not None:
				body = columns[action.id_c]
			elif type(action) == O and ordinals is not None:
				body = ordinals[action.id_c]
			elif type(action) == I:
				body = str(action.id_c)
			elif type(action) == V and values is not None and action.id_c < len(values):
				# If id_c computed the Value is a result of some other kind of computation and is not listed in values
				body = values[action.id_c]
		else:
			for x in action.production.split(' ')[1:]:
				if x in keywords:
					body = x
					break
		return ASTNode(body, args, action, masked=masked)

	def set_action(self, action):
		self.action = action

	def set_args(self, args: []):
		self.args = args

	def validate(self):
		"""
		When parsing from logic_str
		Validates and defines undefined actions of leaf nodes. And expands the nodes of Obj that consist only on N or V
		:return:
		"""
		result = True
		next_actions = self.action.get_next_action()
		for i, (arg, next_action) in enumerate(zip(self.args, next_actions)):
			if arg.action is None and arg.body == "all_rows" and type(next_action) is type(View):
				# This is an edge case were we have skipped the action assignation on init to protect against using
				# keywords as column, or value names
				arg.set_action(View(View.id_all_rows))
			elif next_action is Obj and type(arg.action) != Obj:
				# Obj(4) gets direct V or N in str parsing but in graph we need an extra node Obj->V
				if arg.action is None:
					v_node = ASTNode(arg.body, args=[], action=V(0))
					obj_node = ASTNode('', args=[v_node], action=Obj(Obj.id_V))  # V
					self.args[i] = obj_node
				elif type(arg.action) == N:
					obj_node = ASTNode('', args=[arg], action=Obj(Obj.id_N))  # N
					self.args[i] = obj_node
					result = result and self.args[i].validate()
				else:
					print("ERROR: Tree action: Obj should be N or V not {}".format(arg.action))
					return False
			elif arg.action is None and next_action in Grammar.terminal_actions():
				arg.set_action(next_action(0))
			elif type(arg.action) == next_action:
				result = arg.validate()
				if not result:
					return False
			elif (self.body == 'hop' or self.body == 'str_hop' or self.body == 'num_hop') and type(arg.action) == View:
				# The original LF grammar presented in Logic2Text has an inconsistency where
				# hop (which returns a unique value of a column given a row) gets a View instead of a Row. This is
				# grammatically incorrect but is present in the 25% of the dataset samples. To fix this we change the
				# 'hop' keyword of all 'hop View C' rules to 'hop_first'. Making it coherent throughout the grammar.
				self.body = "{}_first".format(self.body)
				self.action = Obj(Obj.id_hop_first)
				if self.body == 'str_hop_first':
					self.action = Obj(Obj.id_str_hop_first)
				elif self.body == 'num_hop_first':
					self.action = Obj(Obj.id_num_hop_first)
				result = arg.validate()
				if not result:
					return False
			else:
				print("ERROR: {} expected {} but got {}".format(self.body, next_action.__name__, arg.action))
				return False
		return result

	def get_values(self, values: List[str]):
		if type(self.action) == V:
			values.append(self.body)
		for arg in self.args:
			arg.get_values(values)

	def get_columns(self, columns: List[str]):
		if type(self.action) == C:
			columns.append(self.body)
		for arg in self.args:
			arg.get_columns(columns)

	def get_cased_values(self, values: Dict = None, avoid_duplicates=True):
		"""
		Returns a Dict with all the values TAB, INF and AUX within this node and its children
		:param values: the dict we are currently filling, we propagate it down the graph to fill it
		:param avoid_duplicates: for a reason that I cannot recall, the original func avoided duplicated values in each
		case list. This is now optional and can be set to False to return all values
		:return:
		"""
		values = {V.TAB: [], V.INF: [], V.AUX: []} if values is None else values
		if type(self.action) == V:
			if self.action.case not in values:
				# First time we encounter a case like this, initialize empty list
				values[self.action.case] = []
			if not avoid_duplicates or (avoid_duplicates and self.body not in values[self.action.case]):
				values[self.action.case].append(self.body)
		for arg in self.args:
			arg.get_cased_values(values, avoid_duplicates)
		return values

	def compute_values(self, table:pd.DataFrame, values_in_extra_to_point:List[str]):
		"""
		Fills the V node body of the tree with id_c = compute (usually INF) with the result of the computation of the graph
		:param table:
		:param values_in_extra_to_point: ['AUX'] which of the value_cases_in_extra should point the model to. If an
		extra value is not here its index will correspond to; in case of TAB to the value in the table. In the case of
		INF to TBC token.
		:return:
		"""
		for arg in self.args:
			if type(arg.action) == Obj and arg.action.id_c == Obj.id_V:
				v_node = arg.args[0]
				if v_node.action.case in V.INF and V.INF not in values_in_extra_to_point:
					# Only two siblings come next to INF values: N and Obj
					sibling = self.get_sibling(Obj, exclude_id_c=Obj.id_V)
					if sibling is None:
						sibling = self.get_sibling(N)
					v_node.body = str(sibling.execute(table))
				elif v_node.action.case in V.AUX and V.AUX not in values_in_extra_to_point:
					# We leave AUX as it is as we don't know hoe to compote them yet
					None
			else:
				arg.compute_values(table, values_in_extra_to_point)

	def get_sibling(self, action_rule_type, exclude_id_c=None):
		"""
		Returns a sibling node of the given action type excluding certain id_c
		(in case there are two Obj and we want the other one)
		:param action_rule_type:
		:param exclude_id_c:
		:return:
		"""
		for arg in self.args:
			if type(arg.action) == action_rule_type and arg.action.id_c != exclude_id_c:
				return arg
		return None

	def mask_values(self, masked_cases: List[str]):
		"""
		Sets mask as True for every value whose case is in masked_cases list
		:param masked_cases:
		:return:
		"""
		if type(self.action) == V and self.action.case in masked_cases:
			self.masked = True
		for arg in self.args:
			arg.mask_values(masked_cases)

	def assign_id_to_terminal_nodes(self, columns: List[str], ordinals: List[str], flat_table: List[str],
									cs_values: Dict[str, List[str]] = None, tbc_id_c:int=None,
									values_in_extra_to_point:List[str]=None):
		"""
		Given a list of columns, ordinals, flat_table and optionally cs_values, assigns the index of the matching
		value of each list to its corresponding node type. Sets id_c of C, O and V
		:param columns: ['col1', 'col2', 'col3']
		:param ordinals: ['0','1','2',n]
		:param flat_table: ['row 0', 'val11', 'val12', 'row 1', etc]
		:param cs_values: {'AUX':['50']} Include only the elements in CS values
		:param tbc_id_c: id_c for the TBC (To Be Computed) value. Aka INF or AUX when they can't be found in cs_values
		:param tbc_id_c: id_c for the TBC (To Be Computed) value. Aka INF or AUX when they can't be found in cs_values
		:param values_in_extra_to_point: ['AUX'] which of the value_cases_in_extra should point the model to. If an
		extra value is not here its index will correspond to; in case of TAB to the value in the table. In the case of
		INF to TBC token.
		:return:
		"""
		if type(self.action) in Grammar.terminal_actions():
			self.action.id_c = self._find_id_c(columns, ordinals, flat_table, cs_values, tbc_id_c, values_in_extra_to_point)
		for arg in self.args:
			arg.assign_id_to_terminal_nodes(columns, ordinals, flat_table, cs_values, tbc_id_c, values_in_extra_to_point)

	def assign_case_to_values(self, cntxt=None):
		"""
		Considering the graph structure in which V values are, assigns its corresponding case (TAB,INF,AUX)
		:param values:
		:param cntxt:
		:return:
		"""
		# if we are in a pre-value Object
		if type(self.action) == Obj and self.action.id_c == Obj.id_V:
			found = False
			if cntxt.action.potential_case_value() == V.AUX:
				# the parent action is considered a potential case AUX action "less, greater than..."
				self.args[0].action.case = V.AUX
				found = True
			if cntxt.action.potential_case_value() == V.TAB:
				# the parent action is considered a potential TAB2 action "filter"
				self.args[0].action.case = V.TAB
				found = True
			elif type(cntxt.action) == Stat and cntxt.action.potential_case_value() == "go_deeper":
				# the parent action is eq and we have to check its siblings to see if TAB or INF
				siblings: List = cntxt.args.copy()
				siblings.remove(self)
				for arg in siblings:
					if arg.action.potential_case_value() == V.TAB or \
							(arg.action.potential_case_value() == "go_deeper" and arg.args[
								0].action.potential_case_value() == V.TAB):
						# parent is eq and sibling is TAB action (like hop). Thus, this value is TAB
						# or this is an Obj N and its N child is TAB (like max)
						self.args[0].action.case = V.TAB
						found = True
						break
					if arg.action.potential_case_value() == V.INF or \
							(arg.action.potential_case_value() == "go_deeper" and arg.args[
								0].action.potential_case_value() == V.INF):
						# parent is eq and sibling is INF action (like diff). Thus, this value is INF
						# or this is an Obj N and its N child is INF
						self.args[0].action.case = V.INF
						found = True
						break
			if not found:
				# none of the above conditions apply, it must be a AUX
				self.args[0].action.case = V.AUX
		elif type(self.action) == Obj and self.action.id_c in [Obj.id_substring, Obj.id_substring_range]:
			# All Vs in a substring are TAB
			self.get_sibling(V).action.case = V.TAB

		for arg in self.args:
			arg.assign_case_to_values(cntxt=self)

	def _find_id_c(self, columns: List[str], ordinals: List[str], flat_table: List[str],
				   cs_values: Dict[str, List[str]] = None, tbc_id_c:int=None, values_in_extra_to_point:List[str]=None):
		"""
		Given a list of columns, ordinals, flat_table and (optionally) cs_values, assigns the index of the matching
		value of each list to its corresponding node type. Sets id_c of C, O and V
		:param columns: ['col1', 'col2', 'col3']
		:param ordinals: ['0','1','2',n]
		:param flat_table: ['row 0', 'val11', 'val12', 'row 1', etc]
		:param cs_values: {'AUX':['50']} Include only the elements in CS values
		:param tbc_id_c: id_c for the TBC (To Be Computed) value. Aka INF or AUX when they can't be found in cs_values
		values_in_extra_to_point: ['AUX'] which of the value_cases_in_extra should point the model to. If an
		extra value is not here its index will correspond to; in case of TAB to the value in the table. In the case of
		INF to TBC token.
		:return:
		"""
		if values_in_extra_to_point is None and cs_values is not None:
			values_in_extra_to_point = cs_values.keys()
		if type(self.action) is C:
			return columns.index(self.body)
		elif type(self.action) is O:
			return ordinals.index(self.body)
		elif type(self.action) is I:
			return int(self.body)
		elif type(self.action) is V:
			if self.action.case == V.TAB:
				if (cs_values is not None and V.TAB in cs_values) and V.TAB in values_in_extra_to_point:
					# TAB values are fed as cs extra values, the index corresponds to the value in that list
					i = cs_values[V.TAB].index(self.body)
					offset = len(flat_table)
					return offset + i
				else:
					# The usual outcome for TAB values, we return the index of the matching cell in the table
					return flat_table.index(self.body)
			elif (self.action.case == V.INF and cs_values is not None and V.INF in cs_values) and V.INF in values_in_extra_to_point:
				i = cs_values[V.INF].index(self.body)
				offset = len(flat_table)
				offset += 0 if V.TAB not in cs_values else len(cs_values[V.TAB])
				return offset + i
			elif self.action.case == V.AUX:
				if (cs_values is not None and V.AUX in cs_values) and V.AUX in values_in_extra_to_point:
					# As an experiment, if we don't add or point to AUX in extra we find its value in the table our
					# analysis shows that 439/1083 AUX values are in the table in a literal (319) or as substring (120)
					# form. We removed all nonexistent AUX values from train/dev
					i = cs_values[V.AUX].index(self.body)
					offset = len(flat_table)
					offset += 0 if V.TAB not in cs_values else len(cs_values[V.TAB])
					offset += 0 if V.INF not in cs_values else len(cs_values[V.INF])
					return offset + i
				else:
					# Test samples still feature LFs with AUX whose value cannot be found in the table. We return -1 as
					# test Examples will only be used to visualize and compare LFs in action list format
					return flat_table.index(self.body) if self.body in flat_table else None
			else:
				# At this point it should be an INF or AUX vale not in cs_values, thus a computed value
				return tbc_id_c
		return None

	@staticmethod
	def _action(func: str):
		"""
		A function in Logic2Text is a keyword of our AST grammar. Each keyword is related to only one prod rule
		Therefore, if func == keyword of the prod rule, we know the specific prod rule (and action) this part of the LF
		refers to.
		BEWARE: We are lucky our grammar only has one unique keyword for each production rule, therefore, if we know the
		keyword we know the production rule. If we update our grammar and the same keyword can be found in different
		production rules we have to check if the args comply with the production rule's grammar
		:param func:
		:return:
		"""
		g = Grammar()
		# for every action class in the grammar
		for action_cls in g.sketch_actions():
			productions = g.get_production(action_cls)
			# for every production rule in this action class
			for prod_id, production in enumerate(productions):
				# for each token in the production rule
				for token in production.split(' ')[1:]:
					if token in keywords and token == func:
						return action_cls(prod_id)
		# raise NotImplementedError("The function: {} can't be found in the grammar".format(func))
		return None

	def to_logic_str(self, resolve_substring=False):
		if len(self.args) == 0:
			return self.body

		str_args = [x.to_logic_str(resolve_substring) for x in self.args]
		if self.body == "":
			# bridge node e.g Obj->N or Obj->V jump right to print args
			result = ";".join(str_args)
		elif resolve_substring and (self.body == 'substring' or self.body == 'substring_range'):
			result = api[self.body]["function"](*str_args)
		else:
			result = "{func}{{{args}}}".format(func=self.body, args=";".join(str_args))
		return result

	def to_action_list(self):
		result = [self.action]
		for arg in self.args:
			result += arg.to_action_list()
		return result

	def execute(self, table: pd.DataFrame, engine: dict = None):
		engine = api if engine is None else engine
		if len(self.args) == 0:
			return table if self.body == "all_rows" else self.body

		args = [x.execute(table, engine) for x in self.args]
		if self.body == "":
			# bridge node e.g Obj->N or Obj->V jump right to print args
			result = args[0]
		else:
			result = engine[self.body]["function"](*args)
			if engine[self.body]["output"] == "str":
				result = str(result)
		return result

	def req_print_graph(self, tree, parent=None):
		"""
		Recursive function to build a TreePlt to print the graph in a readable way
		:param tree:
		:param parent:
		:return:
		"""
		if parent is None:
			node = tree.create_node(self.body)
		else:
			if self.body == "":
				# bridge node e.g Obj->N or Obj->V jump right to print args
				node = parent
			else:
				node = tree.create_node(self.body, parent=parent)

		for arg in self.args:
			if len(arg.args) == 0:
				text = arg.body if arg.masked is False else "{} (masked)".format(arg.body)
				tree.create_node(text, parent=node)
			else:
				arg.req_print_graph(tree, parent=node)

	def req_dist_graph(self) -> ZSSNode:
		"""
		Recursive function to build a ZSSNode to measure the edit distance between two graphs
		:param tree:
		:param parent:
		:return:
		"""
		node = ZSSNode(str(self.action))

		for arg in self.args:
			node.addkid(arg.req_dist_graph())
		return node

	def update_grammar(self, version:str='V3'):
		if version == 'V3':
			# Simplify grammar e.g: str_eq -> eq
			if self.body in changes[version]['modifications']:
				self.body = changes[version]['modifications'][self.body]
				self.action = self._action(self.body)

			for arg in self.args:
				arg.update_grammar(version)

	def update_body(self, old_value, new_value, node_action, value_case):
		if type(self.action) is node_action and self.action.case == value_case and self.body == old_value:
			self.body = new_value
		for arg in self.args:
			arg.update_body(old_value, new_value, node_action, value_case)



class ASTTree(object):
	def __init__(self, root: ASTNode, is_valid: bool = False):
		self.root:ASTNode = root
		self.is_valid:bool = is_valid

	@classmethod
	def from_logic_str(cls, logic_str: str, columns: List[str] = None, ordinals: List[str] = None,
					   flat_table: List[str]=None, cs_values: Dict[str, List[str]] = None, tbc_idx:List[int]=None,
					   values_in_extra_to_point:List[str]=None):
		"""
		:param logic_str: str representation of the logic form
		:param columns: list of columns
		:param ordinals: list of all ordinals []
		:param flat_table: table as it is fed to the model including 'row N' tokens
		:param cs_values: {"AUX":['50']} list of values fed as extra values to the model. Do not mention a Value case if
		 it isn't in cs values
		:param tbc_idx: index of the tbc in the pointer_values list.
		:param values_in_extra_to_point: ['AUX'] which of the value_cases_in_extra should point the model to. If an
		extra value is not here its index will correspond to; in case of TAB to the value in the table. In the case of
		INF to TBC token.
		:return:
		"""
		# remove the last '=true' part
		logic_str = re.sub(r' ?= ?true', '', logic_str)
		logic = _parse_dic(_parse(logic_str)[0])[0]
		first_fn = next(iter(logic))
		args = logic[first_fn]
		root = ASTNode.from_logic_str(first_fn, args)
		is_valid = root.validate()
		root.assign_case_to_values()

		if columns is not None and ordinals is not None and flat_table is not None:
			if tbc_idx is None or tbc_idx == []:
				# Wild guess that tbc_index will be last token of pointer values: len(flat_table) + len(cs_values) in values_in_extra_to_point
				tbc_id_c = len(flat_table) + sum([len(values) if cs_case in values_in_extra_to_point else 0 for cs_case, values in cs_values.items()])
			else:
				tbc_id_c = tbc_idx[0]
			if values_in_extra_to_point is None and cs_values is not None:
				# Default point to all cs_values
				values_in_extra_to_point = cs_values.keys()
			root.assign_id_to_terminal_nodes(columns, ordinals, flat_table, cs_values, tbc_id_c, values_in_extra_to_point)
		return ASTTree(root, is_valid)

	@classmethod
	def from_action_list(cls, actions: List[Action], columns: List[str]=None, ordinals: List[str]=None,
						 values: List[str]=None, table:pd.DataFrame=None, padding=False,
						 values_in_extra_to_point:List[str]=None):
		"""
		:param actions:
		:param columns:
		:param ordinals:
		:param values: list of all values, containing "row n"+TAB+INF+AUX+oov_token. The same list we feed the pointer
		:param table: the table in
		:param padding:
		:param values_in_extra_to_point: ['AUX'] which of the value_cases_in_extra should point the model to. If an
		extra value is not here its index will correspond to; in case of TAB to the value in the table. In the case of
		INF to TBC token.
		:return:
		"""
		root = ASTNode.from_action_list(actions, columns, ordinals, values, padding=padding)
		if table is not None and values_in_extra_to_point is not None:
			# We try to compute any values marked to be computed (id_c = computed)
			try:
				root.compute_values(table, values_in_extra_to_point)
			except:
				# Malformed LF will eventually evaluate to False
				None
		is_valid = root.validate()
		root.assign_case_to_values()
		return ASTTree(root, is_valid)

	def to_logic_str(self, add_dataset_formatting=False, resolve_substring=False):
		logic_str = self.root.to_logic_str(resolve_substring=resolve_substring)
		if add_dataset_formatting:
			logic_str = add_sep_spaces(logic_str)
			logic_str = logic_str + "= true"
		return logic_str

	def to_action_list(self):
		return self.root.to_action_list()

	def to_zss_tree(self):
		return self.root.req_dist_graph()

	def execute(self, table: pd.DataFrame, engine: dict = None):
		try:
			return self.root.execute(table, engine)
		except:
			#print(traceback.format_exc())
			return False

	def get_cased_values(self):
		return self.root.get_cased_values()

	def get_scope_values(self, table: pd.DataFrame = None, legacy_api: bool = False):
		return self.root.get_scope_values(table, legacy_api=legacy_api)

	def print_graph(self):
		tree = TreePlt()
		self.root.req_print_graph(tree)
		print()
		tree.show()

	def get_sketch(self):
		sketch = list()
		actions = self.to_action_list()
		if actions:
			for ta in actions:
				if type(ta) not in Grammar.terminal_actions():
					sketch.append(ta)
		return sketch

	def update_grammar(self, version:str='V3'):
		self.root.update_grammar(version)

	def get_columns(self):
		result = []
		self.root.get_columns(result)
		return result


# Helpers
def _append(lst: [], idx_lst: [int], obj):
	"""
	Given a nested list, a list of indexes indicating the position of the appended object and object. Appends the
	object to the last element of the index. This works in the same way we could append an object to a tensor
	:param lst: nested list in which the object will be appended
	:param idx_lst: [0, 1, 2] a list of indexes indicating the possition of the appended object. In this case obj will
	be appended to the element in position 2 of the list in the position 1 in the list of position 0 of the main list
	:param obj: object to append, can be str or list [[]]
	:return:
	"""
	curr_lst = lst
	for i, idx in enumerate(idx_lst):
		if i + 1 < len(idx_lst):
			curr_lst = curr_lst[idx]
		else:
			curr_lst[idx] += obj


def remove_sep_spaces(s: str) -> str:
	# clean spaces between separators
	for sep in ['{', '}', ';']:
		s = re.sub(r' ?{} ?'.format(sep), sep, s)
	return s


def add_sep_spaces(s: str) -> str:
	# add spaces between separators
	for sep in ['{', '}', ';']:
		s = re.sub(r'(?<! ){}'.format(sep), ' ' + sep, s)
		s = re.sub(r'{}(?! )'.format(sep), sep + ' ', s)
	return s


def _parse(s: str) -> []:
	s = remove_sep_spaces(s)
	idx: List[int] = [0, 0]
	queue = [[""]]
	for char in s:
		if char == "{":
			_append(queue, idx[:-1], [[""]])
			idx[-1] += 1
			idx.append(0)
		elif char == ";":
			_append(queue, idx[:-1], [""])
			idx[-1] += 1
		elif char == "}":
			idx.pop(-1)
		else:
			_append(queue, idx, char)
	return queue

def is_a_in_x(value_a, value_x):
	tokenizer = get_tokenizer()
	if tokenizer is None:
		return value_a in value_x
	val_full_tok: List[str] = tokenizer.tokenize(value_x)
	if value_a in val_full_tok:
		return True
	# If the list of tokens that form value_substring are within value_full but not as just one token
	val_sub_tok = tokenizer.tokenize(value_a)
	for i in range(len(val_full_tok) - len(val_sub_tok) + 1):
		if val_sub_tok == val_full_tok[i:i + len(val_sub_tok)]:
			return True
	return False

def _parse_dic(lst: []):
	"""
	Gets hierarchy well. Converts form:
	[['eq', ['hop', ['argmax', ['all_rows', 'average'], 'player'], 'alec bedser']]]
	to
	[{'eq': [{'hop': [{'argmax': ['all_rows', 'average']}, 'player']}, 'alec bedser']}]
	:param lst: lst from parse
	:return:
	"""
	result_lst = []
	while len(lst) > 0:
		if len(lst) == 1:
			result_lst.append(lst[0])
			lst = []
		elif type(lst[0]) is str and type(lst[1]) is list:
			result_lst.append({lst[0]: _parse_dic(lst[1])})
			lst = lst[2:]
		elif type(lst[0]) is str and type(lst[1]) is str:
			result_lst.append(lst[0])
			lst = lst[1:]
		else:
			raise Exception()

	return result_lst

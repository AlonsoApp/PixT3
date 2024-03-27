import re
from typing import List, Dict

import pandas as pd
import traceback
from itertools import chain

import dataset.logic2text.logical_form.APIs
from datasource.logic2text.model_utils import get_tokenizer
from datasource.logic2text.logical_form.lf_engine import api
from datasource.logic2text.logical_form.lf_grammar_versions import changes
from datasource.logic2text.logical_form.legacy.v2.lf_grammar import *
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
	def from_action_list(cls, actions: List[Action], predefined_values=None, padding=False):
		actions = actions.copy()
		return ASTNode._req_from_action_list(actions, predefined_values, padding)

	@staticmethod
	def _req_from_action_list(actions: List[Action], predefined_values=None, padding=False):
		"""
		:param actions:
		:param predefined_values:
		:param padding: if True we expect only sketch actions and we add the corresponding terminal action based on the
		expected next action. We don't propagate the recursion further because it is a terminal action
		:return:
		"""
		action = actions.pop(0)
		next_actions = action.get_next_action()
		args = []
		for next_action in next_actions:
			if padding and next_action in Grammar.terminal_actions():
				args.append(ASTNode(body='', args=[], action=next_action(0)))
			else:
				node = ASTNode._req_from_action_list(actions, predefined_values, padding)
				args.append(node)
		# body
		body = ''
		masked = False
		if type(action) in Grammar.terminal_actions():
			if predefined_values is not None:
				body = predefined_values[type(action)][action.id_c]
				if body == MASKED_TOKEN:
					masked = True
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

	def get_cased_values(self, values: Dict = None, avoid_duplicates=True):
		"""
		Returns a Dict with all the values TAB1, TAB2, INF and AUX within this node and its children
		:param values: the dict we are currently filling, we propagate it down the graph to fill it
		:param avoid_duplicates: for a reason that I cannot recall, the original func avoided duplicated values in each
		case list. This is now optional and can be set to False to return all values
		:return:
		"""
		values = {V.TAB1: [], V.TAB2: [], V.INF: [], V.AUX: []} if values is None else values
		if type(self.action) == V:
			if self.action.case not in values:
				# First time we encounter a case like this, initialize empty list
				values[self.action.case] = []
			if not avoid_duplicates or (avoid_duplicates and self.body not in values[self.action.case]):
				values[self.action.case].append(self.body)
		for arg in self.args:
			arg.get_cased_values(values, avoid_duplicates)
		return values

	def get_scope_values(self, table: pd.DataFrame, values: Dict = None, legacy_api: bool = False):
		"""
		Returns a dictionary with the values for each case. The actual value in the LF paired with a list of values that
		come as a result of executing the LF up from that point
		:param table: DataFrame of the entire table
		:param values: resulting dict to propagate throughout the recursive execution
		:param legacy_api: use legacy API? True = APIs (Chen et al. 2020's implementation), False = api (our implementation)
		:return: {'TAB1':[(value_in_lf, [values in scope])]}
		"""
		values = {V.TAB1: [], V.TAB2: [], V.INF: [], V.AUX: []} if values is None else values

		for arg in self.args:
			if type(arg.action) == Obj and arg.action.id_c == Obj.id_V:
				v_node = arg.args[0]
				if v_node.action.case in values.keys():
					#Get the sibling Node that is a View or a Row and execute it
					table_value = self.get_scope_value(table, v_node.action.case, v_node.body, legacy_api)
					values[v_node.action.case].append((v_node.body, table_value))
			else:
				arg.get_scope_values(table, values, legacy_api)
		return values

	def get_scope_value(self, table: pd.DataFrame, case: str, v_value: str, legacy_api: bool = False) -> List:
		"""
		Returns the values that, given the LF execution, should fit as the V value at this point of the LF
		:param table: full df of the table for execution
		:param case: Value case of the V
		:param v_value: Value of V
		:param legacy_api: use legacy API? True = APIs (Chen et al. 2020's implementation), False = api (our implementation)
		:return: list of value candidates that fit as the V value after executing the LF upto this point
		"""
		engine = dataset.logic2text.logical_form.APIs.APIs if legacy_api else api
		if case == V.TAB1:
			return [self.get_sibling(Obj, exclude_id_c=Obj.id_V).execute(table, engine=engine)]
		elif case == V.TAB2:
			if type(self.action) == Stat:
				if self.action.id_c == Stat.id_all_eq or self.action.id_c == Stat.id_all_str_eq:
					return self.get_sibling(View).execute(table, engine=engine)[self.get_sibling(C).body].values
				elif self.action.id_c == Stat.id_most_eq or self.action.id_c == Stat.id_most_str_eq or \
						self.action.id_c == Stat.id_most_not_eq or self.action.id_c == Stat.id_most_str_not_eq:
					# We use the value of V to query the values in this scope because if we just get the values that
					# "most" (+30%) equal each other in this scope we may get different results (values that match each
					# other with filter_eq may not do it in == and this could also be a 'contains' case)
					# 'not' values should behave in the same way as 'most' implies that the value should be contained in
					# the subview. I know this can be any OOV value but the dataset only has table values in 'not' clauses
					a_rule = "filter_eq" if self.action.id_c == Stat.id_most_eq else "filter_str_eq"
					view_filter = engine[a_rule]["function"](self.get_sibling(View).execute(table, engine=engine),
														   self.get_sibling(C).body, v_value)
					return view_filter[self.get_sibling(C).body].values
			elif type(self.action) == View:
				if self.action.id_c == View.id_filter_eq or self.action.id_c == View.id_filter_str_eq:
					return self.execute(table, engine=engine)[self.get_sibling(C).body].values
				elif self.action.id_c == View.id_filter_not_eq or self.action.id_c == View.id_filter_str_not_eq:
					# I know this can be any OOV value but the dataset only has table values in 'not' clauses
					a_rule = "filter_eq" if self.action.id_c == View.id_filter_not_eq else "filter_str_eq"
					view_filter = engine[a_rule]["function"](self.get_sibling(View).execute(table, engine=engine),
														   self.get_sibling(C).body, v_value)
					return view_filter[self.get_sibling(C).body].values
		elif case == V.INF:
			# Only two siblings come next to INF values: N and Obj
			sibling = self.get_sibling(Obj, exclude_id_c=Obj.id_V)
			if sibling is None:
				sibling = self.get_sibling(N)
			return [str(sibling.execute(table, engine=engine))]
		elif case == V.AUX:
			# Although we could speculate on these values, we are leaving it far more simple. Any value in the table is the scope
			# greater and less: get the value these are comparing to and suggest a higher/lower value to it
			# 'not' the dataset only contains values within the table (even for these 'not' clauses) thus possible
			# values should be all the values in the C that do not match the values in the resulting View at this point
			return list(chain.from_iterable(table.values.tolist()))

		return self.execute(table, engine=engine)

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

	def assign_id_to_terminal_nodes(self, predefined_values, masked_cases):
		"""
		Given a list of columns, ordinals, values_TAB1, values_TAB2, values_INF, and values_AUX assigns the index of the matching
		value of each list to its corresponding node type. Sets id_c of C, O and V
		:param predefined_values: {C: [], O: [], V: {"TAB1":[], "TAB2":[], "INF":[], "AUX":[]}} TAB1 must contain all table values including 'row N' tokens
		:param masked_cases: we need to know which value cases will be masked to calculate the index offset
		:return:
		"""
		if type(self.action) in Grammar.terminal_actions():
			self.action.id_c = self._find_id_c(predefined_values, masked_cases)
		for arg in self.args:
			arg.assign_id_to_terminal_nodes(predefined_values, masked_cases)

	def assign_case_to_values(self, cntxt=None):
		"""
		Considering the graph structure in which V values are, assigns its corresponding case (TAB1,TAB2,INF,AUX)
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
			if cntxt.action.potential_case_value() == V.TAB2:
				# the parent action is considered a potential TAB2 action "filter"
				self.args[0].action.case = V.TAB2  # set to V.TAB1 to merge TAB1 + TAB2
				found = True
			elif type(cntxt.action) == Stat and cntxt.action.potential_case_value() == "go_deeper":
				# the parent action is eq and we have to check its siblings to see if TAB or INF
				siblings: List = cntxt.args.copy()
				siblings.remove(self)
				for arg in siblings:
					if arg.action.potential_case_value() == V.TAB1 or \
							(arg.action.potential_case_value() == "go_deeper" and arg.args[
								0].action.potential_case_value() == V.TAB1):
						# parent is eq and sibling is TAB1 action (like hop). Thus, this value is TAB1
						# or this is an Obj N and its N child is TAB1 (like max)
						self.args[0].action.case = V.TAB1
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
		# print("---------------Rare case found---------------")
		# tree = TreePlt()
		# cntxt.req_print_graph(tree)
		# tree.show()
		# raise Exception("Value case not found in: {parent}: ")

		for arg in self.args:
			arg.assign_case_to_values(cntxt=self)

	def _find_id_c(self, predefined_values, masked_cases=None, fuzzy_eq=False):
		"""
		Gets the id_c of the C, O or V action for a text that matches a value of its corresponding predefined_values list
		:param text: the text for which we want to get the id_c
		:param action: the action of the current node
		:param predefined_values: {C: [], O: [], V: {"TAB1":[], "TAB2":[], "INF":[], "AUX":[]}} TAB1 must contain all table values including 'row N' tokens
		:param value_case: in case this is a V node, which value case is this (None = no case, 'TAB1', 'TAB2', 'INF' , 'AUX')
		:param fuzzy_eq: in case we want to merge cases 1a and 1b, we need fuzzy_eq if we want to find the id for some TAB2 values
		:return:
		"""
		masked_cases = [] if masked_cases is None else masked_cases
		if type(self.action) is V:
			if self.masked is True:
				# if masked we assign the last index +1. Index that would correspond to the oov_token
				return sum([len(val) if key not in masked_cases else 0 for key, val in predefined_values[V].items()])
			if self.action.case == V.TAB1:
				# First we try finding the case1 value with == which is stricter
				for i, val in enumerate(predefined_values[V][V.TAB1]):
					if self.body == val:
						return i
				if fuzzy_eq:
					# If == doesn't find any matches we go to fuzz eq
					for i, val in enumerate(predefined_values[V][V.TAB1]):
						if api['eq']['function'](self.body, val) is True or api['str_eq']['function'](self.body, val) is True:
							return i
					# eq and str_eq functionality is not the same as the one used in fuzzy_match filters of Logic2Text APIs
					for i, val in enumerate(predefined_values[V][V.TAB1]):
						#if fuzzy_match_eq(val, self.body):
							return i

			elif self.action.case == V.TAB2:
				i = predefined_values[V][V.TAB2].index(self.body)
				offset = 0 if V.TAB1 in masked_cases else len(predefined_values[V][V.TAB1])
				return offset + i
			elif self.action.case == V.INF:
				i = predefined_values[V][V.INF].index(self.body)
				offset = 0 if V.TAB1 in masked_cases else len(predefined_values[V][V.TAB1])
				offset += 0 if V.TAB2 in masked_cases else len(predefined_values[V][V.TAB2])
				return offset + i
			elif self.action.case == V.AUX:
				i = predefined_values[V][V.AUX].index(self.body)
				offset = 0 if V.TAB1 in masked_cases else len(predefined_values[V][V.TAB1])
				offset += 0 if V.TAB2 in masked_cases else len(predefined_values[V][V.TAB2])
				offset += 0 if V.INF in masked_cases else len(predefined_values[V][V.INF])
				return offset + i
		elif type(self.action) is I:
			return int(self.body)
		else:
			return predefined_values[type(self.action)].index(self.body)


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

	def to_logic_str(self):
		if len(self.args) == 0:
			return self.body

		str_args = [x.to_logic_str() for x in self.args]
		if self.body == "":
			# bridge node e.g Obj->N or Obj->V jump right to print args
			result = ";".join(str_args)
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

	def convert_value_to_substring(self, value_substring, value_full, case):
		"""
		WARNING: This is meant to be used in the fix_values_within_table. It does not update the id_cs of the resulting
		V nodes
		Converts an Obj(8)>V() into an Obj(9)>V(),I(),I()
		:param value_substring: The substring featured in the original LF
		:param value_full: The full value as it appears in the table
		:param case:
		:return:
		"""
		if type(self.action) is Obj and self.action.id_c == Obj.id_V and self.args[0].action.case == case \
				and self.args[0].body == value_substring:
			i_start, i_end = self.wordpiece_bounds(value_substring, value_full)
			self.action = Obj(Obj.id_substring if i_end is None else Obj.id_substring_range)
			self.body = self.action.production.split(' ')[1]  # 'substring' | 'substring_range'
			self.args = []
			self.args.append(ASTNode(value_full, args=[], action=V(0)))
			self.args.append(ASTNode(str(i_start), args=[], action=I(i_start)))
			if i_end is not None:
				self.args.append(ASTNode(str(i_end), args=[], action=I(i_end)))

		for arg in self.args:
			arg.convert_value_to_substring(value_substring, value_full, case)

	@staticmethod
	def substring_bounds(value_substring, value_full):
		i_start = re.search(re.escape(value_substring), value_full).start()
		i_end = re.search(re.escape(value_substring), value_full).end()
		return i_start, i_end

	@staticmethod
	def wordpiece_bounds(value_substring, value_full):
		tokenizer = get_tokenizer()
		val_full_tok:List[str] = tokenizer.tokenize(value_full)
		if value_substring in val_full_tok:
			# Sometimes the substring is already a token
			i = val_full_tok.index(value_substring)
			return i, None
		# If the list of tokens that form value_substring are within value_full but not as just one token
		val_sub_tok = tokenizer.tokenize(value_substring)
		for i in range(len(val_full_tok) - len(val_sub_tok) + 1):
			if val_sub_tok == val_full_tok[i:i + len(val_sub_tok)]:
				i_2 = i + len(val_sub_tok) - 1 if len(val_sub_tok)>1 else None  # -1 because both bounds must be inclusive [] not [)
				return i, i_2
		raise Exception("No wordpiece bounds found for '{}' -> '{}'".format(value_substring, value_full))


	def convert_to_contains(self, current_value:str, cell_value:str, case:str, parent_node=None):
		"""
		Converts the parent action of an Obj(V) representing the given value to its contains equivalent
		:param current_value:
		:param cell_value:
		:param case:
		:param parent_node:
		:return:
		"""
		# This looks terrible, I should reimplement it with node identifiers and proper upward graph navigation
		is_the_obj_v = type(self.action) is Obj and self.action.id_c == Obj.id_V and self.args[0].action.case == case \
				and self.args[0].body == current_value
		is_the_obj_subs = type(self.action) is Obj and self.action.id_c in [Obj.id_substring, Obj.id_substring_range] \
						  and self.args[0].body == cell_value
		if is_the_obj_v or is_the_obj_subs:
			# First, if the func is one of the old str_ ones we get its simplified version
			func_to_replace = changes['V3']['modifications'][parent_node.body] if parent_node.body in changes['V3']['modifications'] else parent_node.body
			if func_to_replace in changes['V3']['contains_equivalences']:
				# If we didn't already change it
				new_func_name = changes['V3']['contains_equivalences'][func_to_replace]
				parent_node.action = self._action(new_func_name)
				parent_node.body = new_func_name

		for arg in self.args:
			arg.convert_to_contains(current_value, cell_value, case, parent_node=self)



class ASTTree(object):
	def __init__(self, root: ASTNode, is_valid: bool = False):
		self.root:ASTNode = root
		self.is_valid:bool = is_valid

	@classmethod
	def from_logic_str(cls, logic_str: str, columns: List[str] = None, ordinals: List[str] = None,
					   cased_values: Dict[str, List[str]] = None, masked_cases: List[str] = None):
		"""
		:param logic_str: str representation of the logic form
		:param columns: list of columns
		:param ordinals: list of all ordinals []
		:param cased_values: {"TAB1":[], "TAB2":[], "INF":[], "AUX":[]} TAB1 must contain all table values including 'row N' tokens
		:param masked_cases: ["INF"] all cases in this list will be marked as masked and its id_c will be the index of oov_token (len+1)
		:return:
		"""
		# remove the last '=true' part
		logic_str = re.sub(r' ?= ?true', '', logic_str)
		logic = _parse_dic(_parse(logic_str)[0])[0]
		first_fn = next(iter(logic))
		args = logic[first_fn]
		root = ASTNode.from_logic_str(first_fn, args)
		predefined_values = None
		if columns is not None and ordinals is not None and cased_values is not None:
			predefined_values = {C: columns, O: ordinals, V: cased_values}
		is_valid = root.validate()
		root.assign_case_to_values()
		if masked_cases is not None and masked_cases != ['']:
			root.mask_values(masked_cases)
		if predefined_values is not None:
			# TODO uncomment this when adapted assign_id_to_terminal_nodes to new api implementation
			# root.assign_id_to_terminal_nodes(predefined_values, masked_cases)
			None
		return ASTTree(root, is_valid)

	@classmethod
	def from_action_list(cls, actions: List[Action], columns: List[str]=None, ordinals: List[str]=None, values: List[str]=None, padding=False):
		"""
		:param actions:
		:param columns:
		:param ordinals:
		:param values: list of all values, containing "row n"+TAB1+TAB2+INF+AUX+oov_token. The same list we feed the pointer
		:param padding:
		:return:
		"""
		predefined_values = None
		if columns is not None and ordinals is not None and values is not None:
			predefined_values = {C: columns, O: ordinals, V: values}
		root = ASTNode.from_action_list(actions, predefined_values, padding=padding)
		is_valid = root.validate()
		root.assign_case_to_values()
		return ASTTree(root, is_valid)

	def to_logic_str(self, add_dataset_formatting=False):
		logic_str = self.root.to_logic_str()
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
			print(traceback.format_exc())
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

	def replace_value(self, old_value, new_value, value_case=V.TAB2):
		"""
		WARNING: This function doesn't update id_c according to the new value. For now this func is only ment to be used
		during the dataset cleaning process
		Replaces the body of any V node with case = value_case and same current body as old_value
		:param old_value:
		:param new_value:
		:param value_case:
		:return:
		"""
		self.root.update_body(old_value, new_value, node_action=V, value_case=value_case)
		# With new value we should update all id_c
		# self.root.assign_id_to_terminal_nodes()

	def convert_value_to_substring(self, value_substring:str, value_full:str, case:str):
		"""
		WARNING: This is meant to be used in the fix_values_within_table. It does not update the id_cs of the resulting
		V nodes
		Converts an Obj(5)>V() into an Obj(6)>V(),I(),I()
		:param value_substring: The substring featured in the original LF
		:param value_full: The full value as it appears in the table
		:param case:
		:return:
		"""
		self.root.convert_value_to_substring(value_substring, value_full, case)

	def convert_to_contains(self, current_value:str, cell_value:str, case):
		"""
		Converts the parent action of an Obj(V) representing the given value to its contains equivalent
		:param current_value:
		:param cell_value:
		:param case:
		:return:
		"""
		self.root.convert_to_contains(current_value, cell_value, case)


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

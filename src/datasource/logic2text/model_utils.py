from transformers import AutoTokenizer
from transformers import BertTokenizer


global_tokenizer:BertTokenizer = None

def init_tokenizer():
	global global_tokenizer
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	global_tokenizer = tokenizer

def get_tokenizer() -> BertTokenizer:
	if global_tokenizer is None:
		init_tokenizer()
	return global_tokenizer

def set_tokenizer(tokenizer: BertTokenizer):
	global global_tokenizer
	global_tokenizer = tokenizer
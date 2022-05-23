from torch.nn import Module
from transformers import BertTokenizer

class AddSepTransform(Module):
	"""
	Add [SEP] between premise and hypothesis
	"""
	def __init__(self):
		super(AddSepTransform, self).__init__()
	
	def forward(self, premise, hypothesis):
		if isinstance(premise, str):
			return premise + ' [SEP] ' + hypothesis
		return [p + ' [SEP] ' + h for p, h in zip(premise, hypothesis)]
	
	
class BertTokenizeTransform(Module):
	
	def __init__(self, max_pad=150):
		super(BertTokenizeTransform, self).__init__()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.max_pad = max_pad
	
	def forward(self, text):
		tokens = self.tokenizer(text, padding="max_length", max_length=self.max_pad, truncation=True)
		return tokens.input_ids, tokens.attention_mask
	
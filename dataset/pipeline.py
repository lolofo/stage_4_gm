import sys
from collections import Callable

from spacy.tokens import Token
from torchtext.vocab import Vocab
from tqdm import tqdm

from helpers import env
from helpers.logger import log


class TextPipeline(nn.Module):
	
	def __init__(self, spacy_model):
		"""
		Using spacy model, the pipeline cut a sentences into array of word tokens. Words are left unchanged
		Args:
			spacy_model (spacy):
		"""
		self.sm = spacy_model
	
	def forward(self, texts):
		texts = [t.strip() for t in texts]
		
		piped_text = tqdm(self.sm.pipe(texts), desc=str(self)+' pipeline', total=len(texts), unit='sentences', file=sys.stdout, disable=env.disable_tqdm)
		return [[tk.text for tk in doc] for doc in piped_text]
	
	def __str__(self):
		return 'standard'
	
	def numericalizer(self, vocab):
		return NumericalizePipeline(self, vocab)
	
	
class MaskPipeline(nn.Module):
	
	def __init__(self, spacy_model, is_mask: Callable = None):
		self.sm = spacy_model
		self.is_mask: Callable[[Token], bool] = is_mask if is_mask is not None else self.__default_mask
		
	def __default_mask(self, token: Token):
		"""
		Return True if token should be masked
		Args:
			token (spacy.Token):

		Returns:
			mask (bool)
		"""
		return token.is_stop or token.pos_ not in ['NOUN', 'VERB', 'ADP', 'ADJ', 'PRON', 'PROPN']
		
	def forward(self, texts):
		
		texts = [t.strip() for t in texts]
		docs = [doc for doc in self.sm.pipe(texts)]
		
		mask = list()
		for doc in docs:
			
			# using default mask
			m = [self.is_mask(tk) for tk in doc]
			
			# tolerate aux
			if all(m):
			#	log.warning(f'All masking vector found: {doc}, tolerate AUX')
				m = [tk.pos_ not in ['NOUN', 'VERB', 'ADP', 'ADJ', 'ADV', 'AUX', 'PRON', 'PROPN'] for tk in doc]
			
			# tolerate all
			if all(m):
			#	log.warning(f'All masking vector found: {doc}, tolerate all tokens')
				m = [False] * len(doc) + [True]
			
			mask.append(m)
			
		return mask
	
	def __str__(self):
		return 'masking'


class LemmaPipeline(TextPipeline):
	"""
	Pipeline transforming list of sentence into list of word array. Words are lemmatized
	Args:
		spacy_model (spacy):
	"""
	
	def forward(self, texts):
		texts = [t.strip() for t in texts]
		piped_text = tqdm(self.sm.pipe(texts), desc=f'Processing by {self}', total=len(texts), unit='sentences', file=sys.stdout, disable=env.disable_tqdm)
		return [[tk.lemma_ for tk in doc] for doc in piped_text]
	
	def __str__(self):
		return 'lemma'


class CaseInsPipeline(TextPipeline):
	"""
	Pipeline transforming list of sentence into list of word array. Words are lower cased transformed
	"""
	
	def forward(self, texts):
		texts = [t.strip() for t in texts]
		piped_text = tqdm(self.sm.pipe(texts), desc=f'{self} pipeline', total=len(texts), unit='sentences', file=sys.stdout, disable=env.disable_tqdm)
		return [[tk.text.lower() for tk in doc] for doc in piped_text]
	
	def __str__(self):
		return 'lower'


class LemmaCasePipeline(TextPipeline):
	"""
	Pipeline transforming list of sentence into list of word array. Words are lemmatized and lower cased
	"""
	
	def forward(self, texts):
		texts = [t.strip() for t in texts]
		return [[tk.lemma_.lower() for tk in doc] for doc in self.sm.pipe(texts)]
	
	def __str__(self):
		return 'lemma-lower'
	
	
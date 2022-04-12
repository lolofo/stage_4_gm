import json
import os
import pickle
import shutil
import sys
from collections import Counter

import pandas as pd
from torch.utils.data import Dataset
from os import path

from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import vocab
from tqdm import tqdm

from data.pipeline import TextPipeline
from exception import InvalidOperationError
from helpers import env
from helpers.logger import log
from helpers.tools import heuristic, save, load


class SNLIDataset(Dataset):
	LABEL = ['neutral', 'entailment', 'contradiction']
	URL = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
	SPLITS = ['train', 'dev', 'test']
	
	def __init__(self, split: str = 'train', cache_path: str = '_out', n: int = -1, shuffle=True):
		"""
		
		Args:
			split       (str):
			cache_path  (str):
			n           (int): max of data to be loaded
			shuffle     (bool): shuffle if load limited data
						If n is precised and shuffle = True, dataset will sample n datas.
						If n is precised and shuffle = False, dataset will take only n first datas.
		"""
		
		# assert
		if split not in self.SPLITS:
			raise InvalidOperationError(f'split {split} doesnt exist in SNLI')
		
		self.split = split
		self.classes = self.LABEL
		root = self._root(cache_path=cache_path)
		self.csv_path = path.join(root, f'{split}.csv')
		self.jsonl_path = path.join(root, 'snli_1.0', f'snli_1.0_{split}.jsonl')
		self.zip_path = path.join(root, '_snli.zip')

		
		# Load csv into dataset
		if path.exists(self.csv_path):
			self._full = self._csv2data()
		
		# translate jsonl into csv if not exist
		else:
			# download and unzip into jsonl file if not exist
			if not path.exists(self.jsonl_path):
				self._url2jsonl(extract_path=root)
			
			# make csv
			self._full = self._jsonl2csv()
		
		self.data = self._full
		self.classes = self.LABEL
		self._encode_label()
		
		if n > 0:
			n = n // 3
			if shuffle:
				self.data = self.data.sample(frac=1).reset_index(drop=True)
			subset = [pd.DataFrame()] * 3
			for label in range(3):
				subset[label] = self.data[self.data['class'] == label]
				subset[label] = subset[label].sample(n=n) if shuffle else subset[label].head(n)
			self.data = pd.concat(subset).reset_index(drop=True)
			
	def __getitem__(self, index: int):
		"""
		
		Args:
			index ():

		Returns:

		"""
		
		# Load data and get label
		if index >= len(self): raise IndexError
		
		doc1 = self.data['premise'][index]
		doc2 = self.data['hypothesis'][index]
		y = self.data['class'][index]
		
		return (doc1, doc2), y
	
	def __len__(self):
		"""
		Denotes the total number of samples
		Returns: int
		"""
		return len(self.data)
	
	def __str__(self):
		return self.data.describe()
	
	def select_class(self, filters: list, reset_id: bool = False):
		"""
		Select class to keep, used in case to binarize the dataset
		
		Args:
			filters     (list): list of keeping class
			            List of keeping class, as label name (ex. ['neutral']) or its id (ex. [1, 0])
			reset_id    (bool): reset class id, only do when given string in filter
						When true, reset class id from 0. (Ex: filters=['neutral', 'contradiction']
						=> {neutral: 0, contradiction: 1})

		Returns:
			new class label (dict)
		"""
		
		assert 0 < len(filters) < 4, f'{filters} must contrains 2 or 3 class'
		
		# Case keep the original classes
		if len(filters) == 3:
			self.classes = self.LABEL
			self.data = self._full
			return self.classes
		
		# Case filter some class:
		if type(filters[0]) == int:
			filters = [self.LABEL[l] for l in filters]  # [0, 1, 2] -> ['neutral', 'entail', 'contracdict']
		
		# Filter out category
		self.data = self._full[self._full['judgment'].isin(filters)].copy()
		self.classes = filters
		
		# If given string and reset idx, then reset class encoding
		if type(filters[0]) == str and reset_id:
			self._encode_label()
		
		return self
	
	@classmethod
	def _root(self, cache_path: str):
		return path.join(cache_path, 'dataset', 'snli')
	
	def _csv2data(self, csv_path=None, split=None):
		"""
		load csv into data
		"""
		if csv_path is None: csv_path = self.csv_path
		if split is None: split = self.split
		
		if not path.exists(csv_path):
			raise InvalidOperationError(f'File csv {csv_path} does not exist')
		
		coltype = {'premise': str, 'hypothesis': str, 'judgment': 'category'}
		desc = f'Load {split}'
		
		with tqdm(total=1, desc=desc, file=sys.stdout, disable=env.disable_tqdm) as bar:
			dataset = pd.read_csv(csv_path, dtype=coltype)
			
			if dataset.isnull().values.any():
				dataset = dataset.dropna().reset_index()
				dataset.to_csv(csv_path, index=False)
			
			bar.update(1)
			bar.set_postfix({'load_from': csv_path})
		
		if env.disable_tqdm: log.info(f'{desc} from {csv_path}')
		return dataset
	
	def _jsonl2csv(self, jsonl_path=None, csv_path=None):
		"""
		Load json to data
		return data
		"""
		if jsonl_path is None: jsonl_path = self.jsonl_path
		if csv_path is None: csv_path = self.csv_path
		
		if not path.exists(jsonl_path):
			raise InvalidOperationError(f'File jsonl {jsonl_path} does not exist')
		
		with open(jsonl_path, 'r') as f:
			jsonl = [json.loads(line) for line in f]
		
		with tqdm(total=len(jsonl), desc=csv_path, file=sys.stdout, unit='rows', disable=env.disable_tqdm) as pbar:
			
			hmap = {'premise': 'sentence1', 'hypothesis': 'sentence2', 'judgment': 'gold_label'}  # header map
			data_dict = {'premise': list(), 'hypothesis': list(), 'judgment': list()}
			
			for line in jsonl:
				for k in data_dict.keys():
					data_dict[k].append(line[hmap[k]])
					pbar.update(1)
			
			data = pd.DataFrame(data_dict)
			nan = ['-', 'N/A', 'n/a']
			data = data[~data['judgment'].isin(nan) & ~data['premise'].isin(nan) & ~data['hypothesis'].isin(
				nan)].dropna().reset_index()
			data.to_csv(csv_path, index=False)
			
		if env.disable_tqdm: log.info(f'Translate from .jsonl to {csv_path}')
		
		return data
	
	def _url2jsonl(self, extract_path=None):
		"""
		Download zip data from url, extract all files, clean up unnecessary ones
		Args:
			url     (str):
			json    ():

		Returns:

		"""
		
		# download the zip set if not exist
		if not path.exists(self.zip_path):
			download_from_url(self.URL, self.zip_path)
		files = extract_archive(self.zip_path, extract_path)
		
		# clean up unnecessary files
		shutil.rmtree(path.join(extract_path, '__MACOSX'), ignore_errors=False)
		useless = ['README.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt', 'snli_1.0_train.txt']
		useless = [path.join(extract_path, 'snli_1.0', f) for f in useless]
		for f in useless:
			if os.path.exists(f):
				os.remove(f)
				files.remove(f)
		
		return files
	
	def _encode_label(self):
		self.data['judgment'] = self.data['judgment'].astype('category')
		self.data['judgment'] = self.data['judgment'].cat.reorder_categories(self.classes)
		self.data['class'] = self.data['judgment'].cat.codes

class HeuristicSNLIDataset(SNLIDataset):
	
	def __init__(self, spacy_model, split: str = 'train', cache_path: str = '_out', n: int = -1, shuffle=True):
		super(HeuristicSNLIDataset, self).__init__(split, cache_path, n=-1, shuffle=False)
		root = self._root(cache_path=cache_path)
		self.heuristic_path = path.join(root, f'{split}_heuristic.pkl')
		
		if path.exists(self.heuristic_path):
			#df_heuristic = pd.read_csv(self.heuristic_path)
			heuristic_dict = load(self.heuristic_path)
			df_heuristic = pd.DataFrame(heuristic_dict)
			log.info(f'Loaded heuristics vector at {self.heuristic_path}')
		else:
			snli = SNLIDataset(split, cache_path, -1, False)
			h = heuristic(snli, spacy_model)
			heuristic_premise = [h_[0] for h_ in h[1]]
			heuristic_hypothesis = [h_[1] for h_ in h[1]]
			heuristic_dict = {'heuristic_premise': heuristic_premise, 'heuristic_hypothesis': heuristic_hypothesis}
			save(heuristic_dict, self.heuristic_path)
			df_heuristic = pd.DataFrame(heuristic_dict)
			#df_heuristic.to_csv(self.heuristic_path, index=False, encoding='utf-8')
			log.info(f'Saved heuristics vector at {self.heuristic_path}')
		
		self._full = pd.concat([self._full, df_heuristic], axis=1)
		self.data = self._full
		
		if n > 0:
			n = n // 3
			if shuffle:
				self.data = self.data.sample(frac=1).reset_index(drop=True)
			subset = [pd.DataFrame()] * 3
			for label in range(3):
				subset[label] = self.data[self.data['class'] == label]
				subset[label] = subset[label].sample(n=n) if shuffle else subset[label].head(n)
			self.data = pd.concat(subset).reset_index(drop=True)
	
	def __getitem__(self, index: int):
		"""

		Args:
			index ():

		Returns:

		"""
		
		# Load data and get label
		if index >= len(self): raise IndexError
		
		doc1 = self.data['premise'][index]
		doc2 = self.data['hypothesis'][index]
		y = self.data['class'][index]
		h1 = self.data['heuristic_premise'][index]
		h2 = self.data['heuristic_hypothesis'][index]
		
		return (doc1, doc2), y, (h1, h2)

def load_dataset(cache_path: str = '_out', n: int = -1) -> dict:
	
	return {
		'train': SNLIDataset('train', cache_path, n, shuffle=True),
		'val': SNLIDataset('dev', cache_path, n),
		'test': SNLIDataset('test', cache_path, n)
	}


def build_vocab(dataset: SNLIDataset, pipeline: TextPipeline, vectors:str=None, cache_path: str = '_out'):
	"""
	Build vocab from given dataset (it should be on train set).
	Args:
		dataset ():
		pipeline ():
		vectors ():
		cache_path ():

	Returns:

	"""
	
	vocab_path = path.join(dataset._root(cache_path), f'vocab_{pipeline}.pkl')
	
	if path.exists(vocab_path):
		with tqdm(total=1, desc=f'Load vocab {vocab_path}', file=sys.stdout, disable=env.disable_tqdm) as bar:
			with open(vocab_path, 'rb') as f:
				vocabulary = pickle.load(f)
			bar.update(1)
		
		if env.disable_tqdm: log.info(f'Load vocab, path="{vocab_path}"')
		
		return vocabulary
	
	sentences = pd.concat([dataset.data['premise'], dataset.data['hypothesis']]).tolist()
	counter = Counter()
	pbar_sentences = tqdm(pipeline(sentences), desc=f'Building vocab', unit='sentences', file=sys.stdout,
	                      disable=env.disable_tqdm)
	if env.disable_tqdm: log.info(f'Building vocabulary')
	for tokens in pbar_sentences:
		counter.update(tokens)
	
	vocabulary = vocab(counter, min_freq=1, specials=['<pad>', '<unk>'])
	vocabulary.set_default_index(vocabulary['<unk>'])
	# Save for next time use
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocabulary, f)
		pbar_sentences.set_postfix({'path': 'vocab_path'})
		if env.disable_tqdm: log.info(f'Saving vocab at: {vocab_path}')
	
	pbar_sentences.close()
	
	return vocabulary

def load_vocab(cache_path: str, pipeline: TextPipeline):
	"""
	Load vocab object that contains vector
	Args:
		cache_path (str):
		pipeline (TextPipeline):

	Returns:

	"""
	vocab_path = path.join(SNLIDataset._root(cache_path), f'vocab_{pipeline}.pkl')
	
	with tqdm(total=1, desc=f'Load vocab {vocab_path}', file=sys.stdout, disable=env.disable_tqdm) as bar:
		with open(vocab_path, 'rb') as f:
			vocabulary = pickle.load(f)
		bar.update(1)
	
	if env.disable_tqdm: log.info(f'Load vocab, path="{vocab_path}"')
	
	return vocabulary
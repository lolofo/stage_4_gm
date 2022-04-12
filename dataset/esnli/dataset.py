import os
import pickle
import re
import shutil
import sys
from collections import Counter

import pandas as pd
from torch.utils.data import Dataset
from os import path

from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import vocab
from tqdm import tqdm, trange

from dataset.pipeline import TextPipeline
from exception import InvalidOperationError
from helpers import env
from helpers.logger import log

class ESNLIDataset(Dataset):
	LABEL = ['neutral', 'entailment', 'contradiction']
	URL = 'https://github.com/OanaMariaCamburu/e-SNLI/archive/refs/heads/master.zip'
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
			raise InvalidOperationError(f'split {split} doesnt exist in eSNLI')
		
		self.split = split
		self.classes = self.LABEL
		root = self._root(cache_path=cache_path)
		self.csv_path = path.join(root, f'{split}.csv')
		self.zip_path = path.join(root, '_esnli.zip')
		
		# download and prepare csv file if not exist
		if not path.exists(self.csv_path):
			self._url2csv(extract_path=root)
		
		self._full = self._csv2data()
		self.data = self._full
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
		return f'ESNLI Dataset, split {self.split}'
	
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
		return path.join(cache_path, 'dataset', 'esnli')
	
	def _csv2data(self, csv_path=None, split=None):
		"""
		load csv into data
		"""
		if csv_path is None: csv_path = self.csv_path
		if split is None: split = self.split
		
		if not path.exists(csv_path):
			raise InvalidOperationError(f'File csv {csv_path} does not exist')
		
		coltype = {'premise': str, 'hypothesis': str, 'judgment': 'category', 'explanation': str,
		           'highlight_premise': str, 'highlight_hypothesis': str}
		desc = f'Load {split}'
		
		with tqdm(total=1, desc=desc, file=sys.stdout, disable=env.disable_tqdm) as bar:
			dataset = pd.read_csv(csv_path, dtype=coltype)
			
			if dataset.isnull().values.any():
				dataset = dataset.dropna().reset_index()
				dataset.to_csv(csv_path, index=False)
			
			bar.update(1)
			bar.set_postfix({'path': csv_path})
		
		if env.disable_tqdm: log.info(f'{desc}, path="{csv_path}"')
		return dataset
	
	def _url2csv(self, extract_path=None):
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
		extract_archive(self.zip_path, extract_path)
		
		# Copy only dataset files
		for f in ['esnli_dev.csv', 'esnli_test.csv', 'esnli_train_1.csv', 'esnli_train_2.csv']:
			shutil.move(path.join(extract_path, 'e-SNLI-master', 'dataset', f), path.join(extract_path, f))
		
		# Fusion train.csv
		files_train = ['esnli_train_1.csv', 'esnli_train_2.csv']
		files_train = [path.join(extract_path, f) for f in files_train]
		csv_data = pd.concat([pd.read_csv(f) for f in files_train])
		csv_data = self._reformat_csv(csv_data)
		csv_data.to_csv(path.join(extract_path, 'train.csv'), index=False, encoding='utf-8')
		
		for split in ['dev.csv', 'test.csv']:
			os.rename(path.join(extract_path, f'esnli_{split}'), path.join(extract_path, split))
			csv_data = pd.read_csv(path.join(extract_path, split))
			csv_data = self._reformat_csv(csv_data)
			csv_data.to_csv(path.join(extract_path, split), index=False, encoding='utf-8')
		
		# clean up unnecessary files
		shutil.rmtree(path.join(extract_path, 'e-SNLI-master'))
		for f in files_train: os.remove(f)
	
	def _reformat_csv(self, data: pd.DataFrame):
		"""
		Remove unecessary columns, rename columns for better understanding. Notice that we also remove extra explanation
		columns.
		Args: data (pandas.DataFrame): Original data given by eSNLI dataset

		Returns:
			(pandas.DataFrame) clean data
		"""
		
		rename_cols = {
			'Sentence1': 'premise',
			'Sentence2': 'hypothesis',
			'gold_label': 'judgment',
			'Explanation_1': 'explanation',
			'Sentence1_marked_1': 'highlight_premise',
			'Sentence2_marked_1': 'highlight_hypothesis'
		}
		
		drop_cols = ['pairID', 'WorkerId'
		             'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
		             'Explanation_2', 'Sentence1_marked_2', 'Sentence2_marked_2',
		             'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
		             'Explanation_3', 'Sentence1_marked_3', 'Sentence2_marked_3',
		             'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3']
		
		
		if data.isnull().values.any():
			log.warning('Original dataset contain NA values, drop these lines.')
			data = data.dropna().reset_index()
		
		# rename column
		data = data.rename(
			columns=rename_cols
		# drop unneeded
		).drop(
			columns=drop_cols, errors='ignore'
		)[['premise', 'hypothesis', 'judgment', 'explanation', 'highlight_premise', 'highlight_hypothesis']]
		
		def correct_quote(txt, hl):
			"""
			Find the incoherent part in text and replace the corresponding in highlight part
			"""""
			
			# find different position between the 2
			diff = [i for i, (l, r) in enumerate(zip(txt, hl.replace('*', ''))) if l != r]
			
			# convert into list to be able to modify character
			txt, hl = list(txt), list(hl)
			
			idx = 0
			for pos_c, c in enumerate(hl):
				if c == '*': continue
				if idx in diff: hl[pos_c] = txt[idx]
				idx += 1

			hl = ''.join(hl)
			return hl
		
		# correct some error
		for side in ['premise', 'hypothesis']:
			data[side] = data[side].str.strip()\
				.str.replace('\\', '', regex=False)\
				.str.replace('*', '')
			data[f'highlight_{side}'] = data[f'highlight_{side}'] \
				.str.strip() \
				.str.replace('\\', '')\
				.str.replace('**', '*', regex=False)
			
			# replace all the simple quote (') by double quote (") as orignal phrases
			idx_incoherent = data[side] != data[f'highlight_{side}'].str.replace('*', '', regex=False)
			sub_data = data[idx_incoherent]
			replacement_hl = [correct_quote(txt, hl) for txt, hl in zip(sub_data[side].tolist(), sub_data[f'highlight_{side}'].tolist())]
			data.loc[idx_incoherent, f'highlight_{side}'] = replacement_hl
			
		return data
	
	def _encode_label(self):
		self.data['judgment'] = self.data['judgment'].astype('category')
		self.data['judgment'] = self.data['judgment'].cat.reorder_categories(self.classes)
		self.data['class'] = self.data['judgment'].cat.codes
	
	@property
	def num_class(self) -> int:
		return self.data['class'].nunique()

class ExplainableESNLIDataset(ESNLIDataset):
	"""
	Extract extra information of explanation column
	"""
	
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
		hl1 = self.data['highlight_premise'][index]
		hl2 = self.data['highlight_hypothesis'][index]
		y = self.data['class'][index]
		
		return (doc1, doc2), y, (hl1, hl2)
	

def build_vocab(dataset: ESNLIDataset, pipeline: TextPipeline, cache_path: str = '_out'):
	"""
	Short hand function that help to build vocabulary from corresponding dataset (searching appropriate columns
	Args:
		dataset (ESNLIDataset): train set to learn vocab
		pipeline (TextPipeline): Transformation done in dataset to tokenize
		cache_path (str): where to save / load vocabulary

	Returns:

	"""
	
	vocab_path = path.join(dataset._root(cache_path), f'vocab_{pipeline}.pkl')
	
	if path.exists(vocab_path):
		return load_vocab(cache_path, pipeline)
	
	sentences = pd.concat([dataset.data['premise'], dataset.data['hypothesis']]).tolist()
	counter = Counter()
	pbar_sentences = tqdm(pipeline(sentences), desc=f'Building vocab', unit='sentences', file=sys.stdout, disable=env.disable_tqdm)
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
	vocab_path = path.join(ESNLIDataset._root(cache_path), f'vocab_{pipeline}.pkl')
	
	with tqdm(total=1, desc=f'Load vocab {vocab_path}', file=sys.stdout, disable=env.disable_tqdm) as bar:
		with open(vocab_path, 'rb') as f:
			vocabulary = pickle.load(f)
		bar.update(1)
	
	if env.disable_tqdm: log.info(f'Load vocab, path="{vocab_path}"')
	
	return vocabulary
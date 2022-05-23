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


class SNLIDataset(Dataset):
	LABEL = ['neutral', 'entailment', 'contradiction']
	URL = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
	SPLITS = ['train', 'dev', 'test']
	
	def __init__(self, split: str = 'train', root: str = '_out', n_data: int = -1, transforms=None):
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
		if split not in _EXTRACTED_FILES.keys():
			raise InvalidOperationError(f'split argument {split} doesnt exist for eSNLI')
		
		root = self.root(root)
		self.split = split
		self.csv_path = path.join(root, _EXTRACTED_FILES[split])
		self.zip_path = path.join(root, ZIP_FILEPATH)
		self.transforms = transforms
		
		# download and prepare csv file if not exist
		download_format_dataset(root, split)
		
		# load the csv file to data
		coltype = {'premise': str, 'hypothesis': str, 'label': 'category', 'explanation': str,
		           'highlight_premise': str, 'highlight_hypothesis': str}
		self.data = pd.read_csv(self.csv_path, dtype=coltype)
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			subset = [pd.DataFrame()] * len(_unique_label)
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
	def __getitem__(self, index: int):
		"""

		Args:
			index ():

		Returns:

		"""
		
		# Load data and get label
		if index >= len(self): raise IndexError  # meet the end of dataset
		
		sample = self.data.loc[index].to_dict()
		
		if self.transforms is not None:
			sample = self.transforms(sample)
		
		return sample
	
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

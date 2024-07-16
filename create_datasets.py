from datasets import Dataset, DatasetDict
import json
import logging
from logging import info
import numpy as np
import os
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Iterator, Sequence

DEBUG = os.environ.get('DEBUG', None)

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO' if DEBUG != None else 'DEBUG').upper()
logging.basicConfig(level=logging.getLevelName(LOG_LEVEL), format='%(message)s', force=True)

def get_data(path: str) -> Iterator[dict]:
    """
    Load the data from the given path and return a list of dicts
    Training data has all of the fields, while test data only contains question and choice_list
    Some data in the dataset is numberic, so to make sure we convert all non-arrays to strings
    """
    return [{k: (v if isinstance(v, Sequence) else str(v)) for k, v in entry.items()} for entry in np.load(path, allow_pickle=True)]

def create_dataset(file: str):
  """
  Create a dataset for the given file with the given number of folds and test size
  """

  n_splits = 5
  test_size = 0.1

  info(f'Creating dataset for {file} with {n_splits} folds and {int(test_size * 100)}% test size')

  data = get_data(f'data/{file}.npy')
  train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

  folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
  splits = folds.split(train_data, [d['choice_order'].index(0) for d in train_data])

  fold_datasets = {
    'test': Dataset.from_list(test_data),
    'folds': []
  }

  for train_idx, validate_idx in splits:
    fold_datasets['folds'].append(DatasetDict({
        "train": Dataset.from_list([data[i] for i in train_idx]),
        "validation": Dataset.from_list([data[i] for i in validate_idx])
    }))

  output_dir = 'datasets'
  output_file = f'{file}.pkl'
  output_path = os.path.join(output_dir, output_file)
  os.makedirs(output_dir, exist_ok=True)
  with open(output_path, 'wb') as f:
    pickle.dump(fold_datasets, f)
    info(f'Dataset saved to {output_path}')

def main():
  files = os.environ.get(FILES).split(',') if os.environ.get('FILES') else ['SP-train', 'WP-train']
  for file in files:
    create_dataset(file)

if __name__ == "__main__":
    main()
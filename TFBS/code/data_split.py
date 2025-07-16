import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from utils import extract_value_as_list, get_split_dirs
from seq_similarity import get_non_similar_rows, handle_inter_train_test_similarity

class DataSplit:

    def __init__(self, logger, config, label='label'):
        self.logger = logger
        self.config = config
        self.label = label
    
    @classmethod
    def split(cls, logger, config, label='label'):
        instance = cls(logger, config, label)
        return instance._split()

    def _split(self):
        split_path, val_split_path = get_split_dirs(self.config)

        if os.path.exists(val_split_path):
            raise ValueError(f"Data_split dir already exists {val_split_path}!")

        os.makedirs(split_path, exist_ok=True)
        df = self._read_dataset(self.config.dataset_path)

        if self.config.partition_mode == 'strict':  # handle similarity between test and the rest
            df = handle_inter_train_test_similarity(
                self.logger, df, self.config.sim, self.config.word_size, split_path)
        df = df.drop(columns=['_in_test'])

        self.logger.log_message(
            f"Creating '{self.config.test_split_type}' - '{self.config.val_split_type}'"
            f" splits into '{split_path}' - '{val_split_path}'")

        # create train-val and test splits
        if self.config.test_split_type == "random":
            df_train_val, df_test = train_test_split(df, test_size=self.config.test_size,
                                                     stratify=df[self.label],
                                                     random_state=self.config.random_state)
        else:  # self.config.test_split_type == "cross"
            df_test, df_train_val = self._split_dataset_by_id(df, 'test', self.config.test_ids)

        # now create train and val splits
        if self.config.val_split_type == "cross":
            df_val, df_train = self._split_dataset_by_id(df_train_val, 'val', self.config.val_ids)
            dfs_train, dfs_val = [df_train], [df_val]
        else:  # self.config.val_split_type == 'n-fold'
            dfs_train, dfs_val = self._n_fold_split(df_train_val)

        # save dataset splits
        df_test.to_csv(os.path.join(split_path, 'test_dataset.csv'), index=False)
        for i, (train_df, val_df) in enumerate(zip(dfs_train, dfs_val), 1):
            current_split_path = os.path.join(val_split_path,
                                              f'Fold_{i}') if self.config.val_split_type == 'n-fold' else val_split_path
            os.makedirs(current_split_path, exist_ok=True)
            train_df.to_csv(os.path.join(current_split_path, 'train_dataset.csv'), index=False)
            val_df.to_csv(os.path.join(current_split_path, 'val_dataset.csv'), index=False)

        # log
        self.logger.log_message(f"Data splits distribution:")
        for i, (train_df, val_df) in enumerate(zip(dfs_train, dfs_val), 1):
            self.logger.log_message(f"\nTrain set distribution - fold-{i}: {train_df[self.label].value_counts()}")
            self.logger.log_message(f"\nValidation set distribution - fold-{i}: {val_df[self.label].value_counts()}")
        self.logger.log_message(f"\nTest set distribution: {df_test[self.label].value_counts()}")

        self.logger.log_message(f"Splits have been saved to {split_path} - {val_split_path}!")

    @classmethod
    def get_splits(cls, logger, config, label='label'):
        instance = cls(logger, config, label)
        return instance._get_splits()

    def _get_splits(self):
        split_path, val_split_path = get_split_dirs(self.config)

        if not os.path.exists(val_split_path):
            raise ValueError(f'Data_split dir does not exist at {split_path}!\nRun 02-data_split.py first.')

        self.logger.log_message(f"Loading '{self.config.test_split_type}' test split from {split_path}")
        df_test = pd.read_csv(os.path.join(split_path, 'test_dataset.csv'))

        self.logger.log_message(
            f"Loading '{self.config.val_split_type}' train-val splits from {val_split_path}")

        if self.config.val_split_type == 'n-fold':
            dfs_train = {
                fold: (
                    pd.read_csv(
                        os.path.join(val_split_path, f'Fold_{fold}', 'train_dataset.csv')
                    )
                )
                for fold in range(1, self.config.fold + 1)
            }
            dfs_val = {
                fold: (
                    pd.read_csv(
                        os.path.join(val_split_path, f'Fold_{fold}', 'val_dataset.csv')
                    )
                )
                for fold in range(1, self.config.fold + 1)
            }
        else:  # self.config.val_split_type == "cross"
            dfs_train = {1: pd.read_csv(os.path.join(val_split_path, 'train_dataset.csv'))}
            dfs_val = {1: pd.read_csv(os.path.join(val_split_path, 'val_dataset.csv'))}

        return dfs_train, dfs_val, df_test

    def _read_dataset(self, dataset_paths):
        dfs = []
        dataset_paths = extract_value_as_list(dataset_paths)
        self.logger.log_message(f'Reading datasets...')

        # iterate through each file path and read the CSV
        for path in dataset_paths:
            self.logger.log_message(f"'------------------'\nDataset '{path}':")
            df = pd.read_csv(path)
            df['origin'] = (path.split('/')[-1].split('.')[0]).replace('_', '-')
            # add column 1 for train, 0 for test, if known to use that to prioritize keeping training data
            if self.config.test_split_type == 'cross':
                df['_in_test'] = df[self.config.id_column].isin(self.config.test_ids).astype(int)
            else:
                df['_in_test']= False
            dfs.append(self._get_unique(df))

        df = pd.concat(dfs, ignore_index=True)

        # convert lower case nucleotides to upper case
        df['sequence'] = df['sequence'].str.upper()
        # randomly rearrange the rows of df (shuffle rows)
        df = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        self.logger.log_message(f"Union of all datasets:")
        return self._get_unique(df)

    def _get_unique(self, df):
        self.logger.log_message(f'Original number of rows: {len(df)}')

        subset = ['sequence']#['chromosomeId', 'sequence']
        redundant_df = df[df.duplicated(subset=subset, keep=False)]
        self.logger.log_message(f'Number of duplicate rows for {subset}: {len(redundant_df)}'
                                f'\nNeed to remove {len(redundant_df)//2} rows, prioritizing training data if known...')

        if len(redundant_df) > 0:
            unique_df = get_non_similar_rows(self.logger, df, subset)
        else:
            unique_df = df
        return unique_df

    # Splits the dataset based on the 'id' column for 'cross' split type.
    def _split_dataset_by_id(self, df, split, values):
        self.logger.log_message(f"{split} 'cross-{self.config.id_column}' split: {values}")

        # check for provided values that are missing in the dataset
        missing_values = set(values) - set(df[self.config.id_column].unique())
        if missing_values:
            raise ValueError(
                f"Warning: The following '{self.config.id_column}'s for '{split}' do not exist in the dataset:",
                missing_values)

        df_values = df[df[self.config.id_column].isin(values)]
        df_rest = df[~df[self.config.id_column].isin(values)]
        return df_values, df_rest

    def _n_fold_split(self, df):
        self.logger.log_message(f"Making '{self.config.fold}-Fold' validation set")
        dfs_train = []
        dfs_val = []

        kf = KFold(n_splits=self.config.fold, shuffle=True, random_state=self.config.random_state)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

            dfs_train.append(df_train)
            dfs_val.append(df_val)

        return dfs_train, dfs_val
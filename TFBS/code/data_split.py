import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from utils import extract_value_as_list

class DataSplit:

    def __init__(self, logger, dataset_path, dataset_split_path, dataset_split, label='label'):
        self.logger = logger
        self.dataset_path = dataset_path
        self.dataset_split_path = dataset_split_path
        self.dataset_split = dataset_split
        self.label = label

    @classmethod
    def split(cls, logger, dataset_path, dataset_split_path, dataset_split, label='label'):
        instance = cls(logger, dataset_path, dataset_split_path, dataset_split, label)
        return instance._split()

    def _split(self):
        if self.dataset_split.test_split_type == 'cross':
            split_path = os.path.join(self.dataset_split_path,
                                      f'test_{self.dataset_split.test_split_type}-{self.dataset_split.id_column}',
                                      f'test-{self.dataset_split.test_ids}')
        elif self.dataset_split.test_split_type == 'random':
            split_path = os.path.join(self.dataset_split_path, f'test_{self.dataset_split.test_split_type}',
                                      str(self.dataset_split.test_size))
        else:
            raise ValueError(
                f"Given dataset_split.test_split_type ({self.dataset_split.test_split_type}) is not valid.")

        if self.dataset_split.val_split_type == 'cross':
            val_split_path = os.path.join(split_path, f'val_{self.dataset_split.val_split_type}-{self.dataset_split.id_column}',
                                          f'tr-{self.dataset_split.train_ids}_val-{self.dataset_split.val_ids}')
        elif self.dataset_split.val_split_type == 'n-fold':
            val_split_path = os.path.join(split_path, f'val_{self.dataset_split.fold}-fold')
        else:
            raise ValueError(
                f"Given dataset_split.val_split_type ({self.dataset_split.val_split_type}) is not valid.")

        dfs_train = []
        dfs_val = []

        if os.path.exists(val_split_path):
            self.logger.log_message(f"Loading '{self.dataset_split.test_split_type}' test split from {split_path}")
            df_test = pd.read_csv(os.path.join(split_path, 'test_dataset.csv'))

            self.logger.log_message(f"Loading '{self.dataset_split.val_split_type}' train-val splits from {val_split_path}")
            if self.dataset_split.val_split_type == 'n-fold':
                for fold in range(1, self.dataset_split.fold + 1):
                    current_split_path = os.path.join(val_split_path, f'Fold_{fold}')
                    self.logger.log_message(f"Loading Fold {fold} from {current_split_path}")
                    dfs_train.append(pd.read_csv(os.path.join(current_split_path,'train_dataset.csv')))
                    dfs_val.append(pd.read_csv(os.path.join(current_split_path, 'val_dataset.csv')))
            else: # cross
                dfs_train.append(pd.read_csv(os.path.join(val_split_path, 'train_dataset.csv')))
                dfs_val.append(pd.read_csv(os.path.join(val_split_path, 'val_dataset.csv')))

        else: # do not exist
            df = self._read_dataset(self.dataset_path)
            self.logger.log_message(
                f"Creating {self.dataset_split.test_split_type} - {self.dataset_split.val_split_type}"
                f" splits into {val_split_path}")
            if self.dataset_split.test_split_type == "random":
                dfs_train, dfs_val, df_test = self._random_test_split(df)
            elif self.dataset_split.test_split_type == "cross":
                dfs_train, dfs_val, df_test = self._split_dataset_by_id(df)

            # Save dataset splits
            for fold in range(1, self.dataset_split.fold+1):
                current_split_path = os.path.join(val_split_path, f'Fold_{fold}')\
                    if self.dataset_split.val_split_type == 'n-fold' else val_split_path
                os.makedirs(current_split_path, exist_ok=True)
                dfs_train[fold-1].to_csv(os.path.join(current_split_path, 'train_dataset.csv'), index=False)
                dfs_val[fold-1].to_csv(os.path.join(current_split_path, 'val_dataset.csv'), index=False)

            df_test.to_csv(os.path.join(split_path, 'test_dataset.csv'), index=False)

            self.logger.log_message(f"Splits saved at {val_split_path}")

        for fold in range(self.dataset_split.fold):
            self.logger.log_message(f"\nTrain label distribution - fold-{fold+1}: {dfs_train[fold][self.label].value_counts()}")
            self.logger.log_message(f"\nValidation label distribution - fold-{fold+1}: {dfs_val[fold][self.label].value_counts()}")
        self.logger.log_message(f"\nTest label distribution: {df_test[self.label].value_counts()}")

        return dfs_train, dfs_val, df_test


    def _read_dataset(self, dataset_paths):
        dfs = []
        dataset_paths = extract_value_as_list(dataset_paths)
        self.logger.log_message(f'Reading datasets...')

        # iterate through each file path and read the CSV
        for path in dataset_paths:
            self.logger.log_message(f'Dataset: {path}')
            df = pd.read_csv(path)
            dfs.append(self._get_unique(df))
            self.logger.log_message('------------------')
        df = pd.concat(dfs, ignore_index=True)

        df['sequence'] = df['sequence'].str.upper()
        # randomly rearrange the rows of df (shuffle rows)
        df = df.sample(frac=1, random_state=self.dataset_split.random_state).reset_index(drop=True)
        return self._get_unique(df, self.dataset_split.train_ids)


    """Remove duplicate sequences, keeping training set duplicates when they exist."""
    def _get_unique(self, df, train_ids=[]):
        self.logger.log_message(f'Original number of rows: {len(df)}')
        subset = ['chromosomeId', 'sequence']
        redundant_df = df[df.duplicated(subset=subset, keep=False)]
        self.logger.log_message(f'Number of duplicate rows for {subset}: {len(redundant_df)}')

        unique_df = df
        if len(redundant_df) > 0:
            redundant_df = df[df.duplicated(subset=subset)]
            self.logger.log_message(f'Number of rows to remove for {subset}: {len(redundant_df)}')

            if len(redundant_df) > 0:
                if self.dataset_split.id_column in df.columns:
                    # add column - 1 for train, 0 for test
                    df['_in_train'] = df[self.dataset_split.id_column].isin(train_ids).astype(int)
                    # sort by _in_train (train first) so duplicates keep training rows
                    df = df.sort_values('_in_train', ascending=False)

                unique_df = df.drop_duplicates(subset=subset, keep='first')

                ## Remove equal number of non-peak (label=0) sequences from same chromosomes from test set
                chrom_counts = redundant_df['chromosomeId'].value_counts()
                non_peaks_to_remove = []

                for chrom, count in chrom_counts.items():
                    # get non-peak sequences from this chromosome
                    chrom_non_peaks = unique_df[(unique_df['chromosomeId'] == chrom) & (unique_df['label'] == 0)]
                    if '_in_train' in df.columns:
                        # get non-peak sequences from this chromosome from test set
                        chrom_non_peaks = chrom_non_peaks[chrom_non_peaks['_in_train'] == 0]

                    if len(chrom_non_peaks) > 0:  # randomly select the same number to remove
                        remove_n = min(count, len(chrom_non_peaks))
                        to_drop = chrom_non_peaks.sample(n=remove_n,
                                                         random_state=self.dataset_split.random_state).index
                        non_peaks_to_remove.extend(to_drop)

                # remove the selected non-peak sequences
                unique_df = unique_df.drop(non_peaks_to_remove)
                if '_in_train' in df.columns:
                    unique_df = unique_df.drop('_in_train', axis=1)
                self.logger.log_message(f'Removed {len(non_peaks_to_remove)} non-peak sequences for balance')

                self.logger.log_message(f'Final number of unique rows for {subset}: {len(unique_df)}')
        return unique_df

    def _random_test_split(self, df):
        # separate test data from the train_val
        df_train_val, df_test = train_test_split(df, test_size=self.dataset_split.test_size,
                                               stratify=df[self.label], random_state=self.dataset_split.random_state)

        dfs_train, dfs_val = self._n_fold_split(df_train_val)
        return dfs_train, dfs_val, df_test

    # Splits the dataset based on the 'id' column.
    def _split_dataset_by_id(self, df):
        dfs_train = []
        dfs_val = []

        id = self.dataset_split.id_column
        self.logger.log_message(f"Train-test 'cross-{id}' split: train={self.dataset_split.train_ids}, test={self.dataset_split.test_ids}")

        # check for provided ids that are missing in the dataset
        unique_ids = set(df[id].unique())
        self.logger.log_message(f"unique '{id}'s in datasets: {unique_ids}")
        missing_train = set(self.dataset_split.train_ids) - unique_ids
        missing_test = set(self.dataset_split.test_ids) - unique_ids

        if missing_train:
            self.logger.log_message(f"Warning: The following {id}s for training do not exist in the dataset:", missing_train)
        if missing_test:
            self.logger.log_message(f"Warning: The following {id}s for testing do not exist in the dataset:", missing_test)

        df_train = df[df[id].isin(self.dataset_split.train_ids)]
        df_test = df[df[id].isin(self.dataset_split.test_ids)]

        if self.dataset_split.val_split_type == "cross":
            self.logger.log_message(f"Val 'cross-{id}' split: {self.dataset_split.val_ids}")
            missing_val = set(self.dataset_split.val_ids) - unique_ids
            if missing_val:
                self.logger.log_message(f"Warning: The following {id}s for validation do not exist in the dataset:", missing_val)
            df_val = df[df[id].isin(self.dataset_split.val_ids)]
            dfs_train.append(df_train)
            dfs_val.append(df_val)
        else: # 'n-fold'
            dfs_train, dfs_val = self._n_fold_split(df_train)

        # check for rows not assigned to any split.
        assigned = set(self.dataset_split.train_ids) | set(self.dataset_split.val_ids) | set(self.dataset_split.test_ids)
        df_unassigned = df[~df[id].isin(assigned)]
        if not df_unassigned.empty:
            self.logger.log_message(f"Warning: Some rows were not assigned to any split. These rows have {id}s:"
                                    , df_unassigned[id].unique())

        return dfs_train, dfs_val, df_test

    def _n_fold_split(self, df):
        self.logger.log_message(f"Making '{self.dataset_split.val_split_type}' validation set")
        dfs_train = []
        dfs_val = []

        kf = KFold(n_splits=self.dataset_split.fold, shuffle=True, random_state=self.dataset_split.random_state)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

            dfs_train.append(df_train)
            dfs_val.append(df_val)

        return dfs_train, dfs_val
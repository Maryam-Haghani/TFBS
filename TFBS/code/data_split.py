import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import extract_single_value, extract_value_as_list

class DataSplit:
    def __init__(self, logger, dataset_path, dataset_split_path, split_type, train_size, train_ids, random_state, label='label'):
        self.logger = logger
        self.dataset_path = dataset_path
        self.dataset_split_path = dataset_split_path
        self.split_type = split_type
        self.train_size = train_size
        self.random_state = random_state
        self.train_ids = train_ids
        self.label = label

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
                # add column - 1 for train, 0 for test
                df['_in_train'] = df['dataset'].isin(train_ids).astype(int)
                # sort by _in_train (train first) so duplicates keep training rows
                df = df.sort_values('_in_train', ascending=False)

                unique_df = df.drop_duplicates(subset=subset, keep='first')

                ## Remove equal number of non-peak (label=0) sequences from same chromosomes from test set
                chrom_counts = redundant_df['chromosomeId'].value_counts()
                non_peaks_to_remove = []

                for chrom, count in chrom_counts.items():
                    # get non-peak sequences from this chromosome from test set
                    chrom_non_peaks = unique_df[(unique_df['chromosomeId'] == chrom) &
                                                (unique_df['label'] == 0) & (unique_df['_in_train'] == 0)]

                    if len(chrom_non_peaks) > 0: # randomly select the same number to remove
                        remove_n = min(count, len(chrom_non_peaks))
                        to_drop = chrom_non_peaks.sample(n=remove_n, random_state=self.random_state).index
                        non_peaks_to_remove.extend(to_drop)

                # remove the selected non-peak sequences
                unique_df = unique_df.drop(non_peaks_to_remove)
                unique_df = unique_df.drop('_in_train', axis=1)
                self.logger.log_message(f'Removed {len(non_peaks_to_remove)} non-peak sequences for balance')

                self.logger.log_message(f'Final number of unique rows for {subset}: {len(unique_df)}')
                input()
        input()
        return unique_df

    def _read_dataset(self, dataset_paths):
        dfs = []
        dataset_paths = extract_value_as_list(dataset_paths)

        # Iterate through each file path and read the CSV into a DataFrame
        for path in dataset_paths:
            self.logger.log_message(f'df path: {path}')
            df = pd.read_csv(path)
            dfs.append(self._get_unique(df))
            self.logger.log_message('------------------')

        # Concatenate all DataFrames into one
        df = pd.concat(dfs, ignore_index=True)

        df['sequence'] = df['sequence'].str.upper()
        # randomly rearrange the rows of df (shuffle rows)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.df = self._get_unique(df, self.train_ids)

    def split(self, id, val_ids, test_ids):
        split_path = os.path.join(self.dataset_split_path, self.split_type)

        if os.path.exists(split_path):
            self.logger.log_message(f"Loading {self.split_type} splits from {split_path}")
            df_train = pd.read_csv(os.path.join(split_path, 'train_dataset.csv'))
            df_val = pd.read_csv(os.path.join(split_path, 'val_dataset.csv'))
            df_test = pd.read_csv(os.path.join(split_path, 'test_dataset.csv'))
        else:
            self._read_dataset(self.dataset_path)
            if self.split_type == "random":
                df_train, df_val, df_test = self._random_split()
            elif self.split_type == "cross_id":
                df_train, df_val, df_test = self._split_dataset_by_id(id, val_ids, test_ids)

            # Save dataset splits
            os.makedirs(split_path, exist_ok=True)
            df_train.to_csv(os.path.join(split_path, 'train_dataset.csv'), index=False)
            df_val.to_csv(os.path.join(split_path, 'val_dataset.csv'), index=False)
            df_test.to_csv(os.path.join(split_path, 'test_dataset.csv'), index=False)
            self.logger.log_message(f"Splits saved at {split_path}")

        self.logger.log_message("Original label distribution:")
        self.logger.log_message(self.df[self.label].value_counts())

        self.logger.log_message(f"\nTrain label distribution: {df_train[self.label].value_counts()}")
        self.logger.log_message(f"\nValidation label distribution: {df_val[self.label].value_counts()}")
        self.logger.log_message(f"\nTest label distribution: {df_test[self.label].value_counts()}")

        return df_train, df_val, df_test

    def _random_split(self):
        # separate training data from the rest
        df_train, temp_data = train_test_split(self.df, train_size=self.train_size,
                                               stratify=self.df[self.label], random_state=self.random_state)

        df_val, df_test = train_test_split(
            temp_data, test_size=0.5,  # this splits temp_data equally into val and test
            stratify=temp_data[self.label], random_state=self.random_state
        )

        return df_train, df_val, df_test

    # Splits the dataset based on the 'id' column.
    def _split_dataset_by_id(self, id, val_ids, test_ids):
        self.logger.log_message(f"Cross {id} split train={self.train_ids}, test={test_ids}")

        # check for provided ids that are missing in the dataset
        unique_ids = set(self.df[id].unique())
        self.logger.log_message(f"unique {id}s: {unique_ids}")
        missing_train = set(self.train_ids) - unique_ids
        missing_test = set(test_ids) - unique_ids

        if missing_train:
            self.logger.log_message(f"Warning: The following {id}s for training do not exist in the dataset:", missing_train)
        if missing_test:
            self.logger.log_message(f"Warning: The following {id}s for testing do not exist in the dataset:", missing_test)

        df_train = self.df[self.df[id].isin(self.train_ids)]
        df_test = self.df[self.df[id].isin(test_ids)]

        if val_ids == "random":
            # Split the training and validation into two parts
            df_train, df_val = train_test_split(df_train, train_size=self.train_size, stratify=df_train[self.label],
                                                random_state=self.random_state)
        else: # specific ids have been given
            missing_val = set(val_ids) - unique_ids
            if missing_val:
                self.logger.log_message(f"Warning: The following {id}s for validation do not exist in the dataset:", missing_val)
            df_val = self.df[self.df[id].isin(val_ids)]


        # check for rows not assigned to any split.
        assigned = set(self.train_ids) | set(val_ids) | set(test_ids)
        df_unassigned = self.df[~self.df[id].isin(assigned)]
        if not df_unassigned.empty:
            self.logger.log_message(f"Warning: Some rows were not assigned to any split. These rows have {id}s:", df_unassigned[id].unique())

        return df_train, df_val, df_test
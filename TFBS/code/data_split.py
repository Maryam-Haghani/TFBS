import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataSplit:
    def __init__(self, logger, df, dataset_split_path, split_type, train_size, random_state, label='label'):
        self.logger = logger
        self.df = df
        self.dataset_split_path = dataset_split_path
        self.split_type = split_type
        self.train_size = train_size
        self.random_state = random_state
        self.label = label

    def split(self, id, train_ids, val_ids, test_ids):
        split_path = os.path.join(self.dataset_split_path, self.split_type)

        if os.path.exists(split_path):
            self.logger.log_message(f"Loading {self.split_type} splits from {split_path}")
            df_train = pd.read_csv(os.path.join(split_path, 'train_dataset.csv'))
            df_val = pd.read_csv(os.path.join(split_path, 'val_dataset.csv'))
            df_test = pd.read_csv(os.path.join(split_path, 'test_dataset.csv'))
        else:
            if self.split_type == "random":
                df_train, df_val, df_test = self._random_split()
            elif self.split_type == "cross_id":
                df_train, df_val, df_test = self._split_dataset_by_id(id, train_ids, val_ids, test_ids)

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
    def _split_dataset_by_id(self, id, train_ids, val_ids, test_ids):
        self.logger.log_message(f"Cross {id} split train={train_ids}, test={test_ids}")

        # check for provided ids that are missing in the dataset
        unique_ids = set(self.df[id].unique())
        self.logger.log_message(f"unique {id}s: {unique_ids}")
        missing_train = set(train_ids) - unique_ids
        missing_test = set(test_ids) - unique_ids

        if missing_train:
            self.logger.log_message(f"Warning: The following {id}s for training do not exist in the dataset:", missing_train)
        if missing_test:
            self.logger.log_message(f"Warning: The following {id}s for testing do not exist in the dataset:", missing_test)

        df_train = self.df[self.df[id].isin(train_ids)]
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
        assigned = set(train_ids) | set(val_ids) | set(test_ids)
        df_unassigned = self.df[~self.df[id].isin(assigned)]
        if not df_unassigned.empty:
            self.logger.log_message(f"Warning: Some rows were not assigned to any split. These rows have {id}s:", df_unassigned[id].unique())

        return df_train, df_val, df_test
import torch

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_length, sequence_column='sequence', label_column='label'):
        self.df = df
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx][self.sequence_column]
        label = self.df.iloc[idx][self.label_column]

        padded_sequence = self.pad_or_truncate(sequence, self.max_length)
        encoded_sequence = self.one_hot_encode(padded_sequence)

        label = torch.LongTensor([label])

        return sequence, encoded_sequence, label

    def pad_or_truncate(self, sequence, length):
        """
        Pads or truncates a sequence to the given length.
        Args:
            sequence (str): The DNA sequence to be padded or truncated.
            length (int): The length to pad or truncate the sequence to.
        """
        current_length = len(sequence)

        if current_length > length:
            sequence = sequence[:length] # truncate
        elif current_length < length: # pad with 'N'
            sequence += 'N' * (length - current_length)

        return sequence

    def one_hot_encode(self, sequence):
        """
        One-hot encode a DNA sequence.
        Args:
            sequence (str): The DNA sequence to be one-hot encoded.
        """
        encoding_dict = {'A': [1, 0, 0, 0, 0],
                         'T': [0, 1, 0, 0, 0],
                         'C': [0, 0, 1, 0, 0],
                         'G': [0, 0, 0, 1, 0],
                         'N': [0, 0, 0, 0, 1]}

        one_hot_sequence = [encoding_dict.get(nucleotide, [0, 0, 0, 0, 1]) for nucleotide in sequence]

        return torch.tensor(one_hot_sequence).unsqueeze(0).float()
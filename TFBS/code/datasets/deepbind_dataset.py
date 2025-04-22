import torch

class DeepBindDataset(torch.utils.data.Dataset):
    def __init__(self, df, kernel_length, sequence_column='sequence', label_column='label'):
        self.df = df
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.m = kernel_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx][self.sequence_column]
        label = self.df.iloc[idx][self.label_column]

        padded_sequence = self.pad(sequence)
        encoded_sequence = self.one_hot_encode(padded_sequence)

        label = torch.LongTensor([label])

        return sequence, encoded_sequence, label

    def pad(self, sequence):
        """
        Pads 'N' to the left and right of the sequence with length n to have a length n + 2m - 2.

        Args:
            sequence (str): The DNA sequence to be padded.

        Returns:
            str: The padded sequence.
        """
        left_padding = 'N' * (self.m - 1)
        right_padding = 'N' * (self.m - 1)

        padded_sequence = left_padding + sequence + right_padding

        return padded_sequence

    def one_hot_encode(self, sequence):
        """
        One-hot encode a DNA sequence, based on Deep-Bind paper.
        Args:
            sequence (str): The DNA sequence to be one-hot encoded.
        """
        encoding_dict = {'A': [1, 0, 0, 0],
                         'C': [0, 1, 0, 0],
                         'G': [0, 0, 1, 0],
                         'T': [0, 0, 0, 1],
                         'N': [0.25, 0.25, 0.25, 0.25] # uniform distribution for the four nucleotides.
                         }

        one_hot_sequence = [encoding_dict.get(nucleotide, [0.25, 0.25, 0.25, 0.25]) for nucleotide in sequence]

        return torch.tensor(one_hot_sequence).unsqueeze(0).float()
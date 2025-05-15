import torch
import torch.utils.data as Data

class DNABERT2_dataset(Data.Dataset):
    def __init__(self, tokenizer, df, max_length):
        self.tokenizer = tokenizer
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def pad_sequence_to_max_length(self, seq):
        # Pad or truncate sequence to max_length
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        padding = torch.full((self.max_length - len(seq),), self.tokenizer.pad_token_id)
        padded_seq = torch.cat([seq, padding])
        return padded_seq

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']

        # Tokenize the sequence
        tokenized_seq = self.tokenizer(sequence, return_tensors='pt')["input_ids"].squeeze(0)  # get rid of batch dimension
        tokenized_seq = self.pad_sequence_to_max_length(tokenized_seq)

        tokenized_seq = torch.LongTensor(tokenized_seq)
        label = torch.LongTensor([label])

        return sequence, tokenized_seq, label

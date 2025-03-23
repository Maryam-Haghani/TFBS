import torch
from torch.nn.functional import softmax
import torch.nn.functional as F

class DNADataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, use_padding=True, add_eos=False):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.use_padding = use_padding
        self.add_eos = add_eos

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']

        # Tokenize the sequence
        tokenized_seq = self.tokenizer(sequence,
                             add_special_tokens=False,
                             padding="max_length" if self.use_padding else None,
                             max_length=self.max_length,
                             truncation=True,
                             )['input_ids']

        # Add EOS token if required
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        tokenized_seq = torch.LongTensor(tokenized_seq)
        label = torch.LongTensor([label])

        return sequence, tokenized_seq, label
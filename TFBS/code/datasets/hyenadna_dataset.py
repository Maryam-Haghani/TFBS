import torch
import os
import sys
hyena_dna_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../hyena-dna"))
sys.path.insert(0, hyena_dna_dir)
from standalone_hyenadna import CharacterTokenizer

class HyenaDNA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, max_length, use_padding=True, add_eos=False):
        self.df = df
        self.max_length = max_length
        self.use_padding = use_padding
        self.add_eos = add_eos
        self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=max_length,
                add_special_tokens=False,
                padding_side='left',
            )

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
            sequence.append(self.tokenizer.sep_token_id)

        tokenized_seq = torch.LongTensor(tokenized_seq)
        label = torch.LongTensor([label])

        return sequence, tokenized_seq, label
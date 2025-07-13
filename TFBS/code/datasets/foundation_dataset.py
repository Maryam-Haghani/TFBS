import torch

class FoundationDataset(torch.utils.data.Dataset):
    def __init__(self, model_name, tokenizer, df, max_length, use_padding=True, add_eos=False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.df = df
        self.max_length = max_length
        self.use_padding = use_padding
        self.add_eos = add_eos

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']

        # Tokenize the sequence
        if self.model_name == "BNABERT-2":
            tokenized_seq = self.tokenizer(sequence, return_tensors='pt')["input_ids"].squeeze(
                0)  # get rid of batch dimension
            tokenized_seq = self.pad_sequence_to_max_length(tokenized_seq)
        else:
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
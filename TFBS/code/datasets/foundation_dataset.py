import torch
import ast

class FoundationDataset(torch.utils.data.Dataset):
    def __init__(self, model_name, tokenizer, data, max_length, window_size=None, stride=None, use_padding=True, add_eos=False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.use_padding = use_padding
        self.add_eos = add_eos

    def __len__(self):
        return len(self.data)

    def pad_sequence_to_max_length(self, seq):
        # Pad or truncate sequence to max_length
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        padding = torch.full((self.max_length - len(seq),), self.tokenizer.pad_token_id)
        padded_seq = torch.cat([seq, padding])
        return padded_seq

    def __getitem__(self, idx):
        uid = self.data.iloc[idx]['peak_uid']
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['label']
        peak_start, peak_end = ast.literal_eval(self.data.iloc[idx]['peak_start_end_index'])

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
        return sequence, tokenized_seq, uid, peak_start, peak_end, label
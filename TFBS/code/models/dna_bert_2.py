from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class MeanPool(nn.Module):
    """Mean‐pool across the sequence dimension."""
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, emb_size)
        return torch.mean(hidden_states, dim=1) # → (batch_size, emb_size)

class DNABERT2(nn.Module):
    def __init__(self, pretrained_model_name="zhihan1996/DNABERT-2-117M", embedding_size=768):
        super(DNABERT2, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        # mean-pool layer:
        self.pool = MeanPool()
        self.classification_head = nn.Linear(embedding_size, 2)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True)

    def forward(self, X):
        outputs = self.bert(X)
        last_hidden = outputs[0] # of shape (batch_size, sequence_length, embedding_size)
        pooled = self.pool(last_hidden) # mean across sequence_length: of shape (batch_size, embedding_size)
        logits = self.classification_head(pooled)
        return logits
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class DNABERT2(nn.Module):
    def __init__(self, pretrained_model_name="zhihan1996/DNABERT-2-117M", embedding_size=768):
        super(DNABERT2, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.classification_head = nn.Linear(embedding_size, 2)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True)

    def forward(self, X):
        outputs = self.bert(X)
        last_hidden_state = outputs[0] # of shape (batch_size, sequence_length, embedding_size)
        mean_embeddings = torch.mean(last_hidden_state, dim=1) # mean across sequence_length: of shape (batch_size, embedding_size)
        logits = self.classification_head(mean_embeddings)
        return logits
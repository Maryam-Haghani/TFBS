from transformers import AutoModel, AutoTokenizer
from .cbam import *

# linear module
class ClassificationHead(nn.Module):
    def __init__(self, embedding_size):
        super(ClassificationHead, self).__init__()

        self.linear = nn.Linear(embedding_size, 2)
        # self.linear1 = nn.Linear(input_channel * embedding_size, input_channel)
        # self.relu = nn.ReLU()
        # self.drop = nn.Dropout(0.5)
        # self.linear2 = nn.Linear(input_channel, 2)

    def forward(self, x):
        x = self.linear(x)
        # x = x.view(x.shape[0], -1)
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.linear2(x)
        return x

# CNN module
class CNNNET_V2(nn.Module):
    def __init__(self, input_channel, embedding_size):
        super(CNNNET_V2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(0.4))
        self.linear1 = nn.Linear(180 * embedding_size, 180)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(180, 2)

    def forward(self, x):
        x = self.conv1(x)
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

# CNN module and CBAM
class CNNNET_complete(nn.Module):
    def __init__(self, input_channel, embedding_size):
        super(CNNNET_complete, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.ChannelGate = ChannelGate(gate_channels=180, reduction_ratio=12, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()
        self.residual_BN = nn.Sequential(
            nn.Conv1d(180, 180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(0.4))
        self.linear1 = nn.Linear(180 * embedding_size, 180)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(180, 2)

    def forward(self, x):
        x = self.conv1(x)
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        residual = x
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        x = x + self.residual_BN(residual)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

class BERT_TFBS(nn.Module):
    def __init__(self, input_channel, pretrained_model_name, embedding_size, model_version):
        super(BERT_TFBS, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.model_version = model_version
        for param in self.bert.parameters():
            param.requires_grad = True

        if self.model_version == 'complete':
            self.model = CNNNET_complete(input_channel, embedding_size)
        elif self.model_version == 'V1':
            self.model = ClassificationHead(embedding_size)
        elif self.model_version == 'V2':
            self.model = CNNNET_V2(input_channel, embedding_size)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True)

    def forward(self, X):
        outputs = self.bert(X)
        last_hidden_state = outputs[0] # of shape (batch_size, sequence_length, embedding_size)

        if self.model_version == "V1":
            mean_embeddings = torch.mean(last_hidden_state, dim=1) # mean across sequence_length: of shape (batch_size, embedding_size)
            logits = self.model(mean_embeddings)
        else:
            logits = self.model(last_hidden_state)
        return logits

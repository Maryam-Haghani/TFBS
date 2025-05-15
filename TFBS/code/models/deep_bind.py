import torch.nn as nn

class DeepBind(nn.Module):
    def __init__(self, kernel_length):
        super(DeepBind, self).__init__()
        self.Convolutions = nn.Sequential(
            # nn.ZeroPad2d((11, 12, 0, 0)), #removed as we are already padded the input
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(kernel_length, 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16)
        )

        self.GlobalMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self.Dense = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2) # changed binding score to 2 outputs (binding/unbinding)
        )

    def forward(self, input):
        input = input.float()

        x = self.Convolutions(input)
        x = self.GlobalMaxPool(x)
        x = self.flatten(x)
        logits = self.Dense(x)

        return logits
import torch
import torch.nn as nn
from einops import rearrange


class Classifier(nn.Module):
    def __init__(self, c_in, c_h, c_out):
        super(Classifier, self).__init__()
        self.in_layer = nn.Linear(c_in, c_h)
        self.conv_relu_block = nn.Sequential(
            nn.Conv1d(c_h, c_h, 3),
            nn.ReLU(),
            nn.Conv1d(c_h, c_h, 3),
            nn.ReLU(),
            nn.Conv1d(c_h, c_h, 3),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(c_h, c_out)

    def forward(self, x):
        """
        x: (batch_size, hidden_dim, seg_len)
        """
        x = rearrange(x, 'n c t -> n t c')
        y = self.in_layer(x)
        y = rearrange(y, 'n t c -> n c t')
        y = self.conv_relu_block(y)
        y = y.mean(-1)
        y = self.out_layer(y)
        return y

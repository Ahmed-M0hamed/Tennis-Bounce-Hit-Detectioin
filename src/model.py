
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 conv to match channel count for the residual add, only when needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        residual = self.downsample(x)
        return self.relu(out + residual)


class TennisEventCNN(nn.Module):
    """
    Residual 1D CNN, window_size=15.

    Input:  (batch, window=15, input_features)
    Output: (batch, num_classes) -- single label for the window
            (matches your existing "classify the center/candidate frame
            using ±half_window context" setup)
    """
    def __init__(self, input_features=8, num_classes=3, dropout=0.3):
        super().__init__()

        # small, mild dilation growth (1, 2, 2) is plenty to cover a
        # 15-frame window without over-expanding receptive field relative
        # to the tiny input length
        self.block1 = ResidualBlock1D(input_features, 64, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = ResidualBlock1D(64, 128, kernel_size=3, dilation=2, dropout=dropout)
        self.block3 = ResidualBlock1D(128, 256, kernel_size=3, dilation=2, dropout=dropout)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),   # *2 because we concat avg + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, window, features) -> conv1d wants channels-first
        x = x.transpose(1, 2)          # (batch, features, window)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)  # (batch, 512, 1)
        pooled = pooled.squeeze(-1)                                       # (batch, 512)

        return self.classifier(pooled)
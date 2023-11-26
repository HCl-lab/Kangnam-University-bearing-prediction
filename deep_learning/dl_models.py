import torch
import torch.nn as nn
import torch.nn.functional as F

import math

hidden_size = 10
num_layers = 2

class CustomRNN(nn.Module):
    """
    模型参数
    input_size = 10   # 输入序列中每个元素的大小
    hidden_size = 20  # RNN隐藏层的大小
    num_layers = 2    # RNN的层数
    num_classes = 5   # 分类数
    """
    def __init__(self, num_classes, input_size=1, hidden_size=hidden_size, num_layers=num_layers):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.log_softmax(out, dim=-1)

        return out

class CustomGRU(nn.Module):
    """
    # 模型参数
    input_size = 10   # 输入序列中每个元素的大小
    hidden_size = 20  # GRU隐藏层的大小
    num_layers = 2    # GRU的层数
    num_classes = 5   # 分类数
    """
    def __init__(self, num_classes, input_size=1, hidden_size=hidden_size, num_layers=num_layers):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.log_softmax(out, dim=-1)

        return out

class CustomLSTM(nn.Module):
    """
    # 模型参数
    input_size = 10   # 输入序列中每个元素的大小
    hidden_size = 20  # LSTM隐藏层的大小
    num_layers = 2    # LSTM的层数
    num_classes = 5   # 分类数
    """
    def __init__(self, num_classes, input_size=1, hidden_size=hidden_size, num_layers=num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.log_softmax(out, dim=-1)

        return out


class TransformerModel(nn.Module):
    """
    # 模型参数
    input_size = 10   # 输入序列中每个元素的大小
    num_classes = 5   # 分类数
    d_model = 512     # Transformer的特征维度
    nhead = 8         # 多头注意力中头的数量
    num_layers = 2    # Transformer编码器层数
    """
    def __init__(self, num_classes, input_size=1, d_model=128, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.encoder = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_classes)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        # 使用全局平均池化
        output = output.mean(dim=1)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 建立一个模型的字典，方便实例化选择
models = {
    'RNN': CustomRNN,
    'GRU': CustomGRU,
    'LSTM': CustomLSTM,
    'Transformer': TransformerModel
}

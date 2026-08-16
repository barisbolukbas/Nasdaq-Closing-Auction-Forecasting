import torch
from torch import nn

from config import GLOBAL_ATTENTION_STRIDE, LOCAL_ATTENTION_WINDOW


# Transformer mask
def build_local_global_mask(seq_len, device, local_window=LOCAL_ATTENTION_WINDOW, global_stride=GLOBAL_ATTENTION_STRIDE):
    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)

    causal = k <= q
    local = k >= q - local_window + 1
    global_anchor = k % global_stride == 0
    allowed = causal & (local | global_anchor)
    return ~allowed


# Residual head
class ResidualRegressionHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        bottleneck = max(hidden_size // 2, 1)

        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.residual = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, x):
        h = self.norm(x)
        return self.linear(h) + self.residual(h)


# LSTM + Transformer
class FlexibleSequenceModel(nn.Module):
    def __init__(
        self, input_size, hidden_size=96, num_layers=2, dropout=0.3,
        use_transformer=True, n_transformer_heads=4, n_transformer_layers=1,
        local_window=LOCAL_ATTENTION_WINDOW, global_stride=GLOBAL_ATTENTION_STRIDE,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.local_window = local_window
        self.global_stride = global_stride

        self.recurrent = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=n_transformer_heads,
                dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.head = ResidualRegressionHead(hidden_size, dropout=dropout)

    def forward(self, x):
        outputs, _ = self.recurrent(x)
        if self.use_transformer:
            mask = build_local_global_mask(outputs.size(1), outputs.device, self.local_window, self.global_stride)
            outputs = self.transformer(outputs, mask=mask)
        return self.head(outputs).squeeze(-1)


# Model builder
def build_model(input_size, device, use_transformer=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    return FlexibleSequenceModel(
        input_size=input_size, hidden_size=96, num_layers=2, dropout=0.3,
        use_transformer=use_transformer, local_window=LOCAL_ATTENTION_WINDOW,
        global_stride=GLOBAL_ATTENTION_STRIDE,
    ).to(device)
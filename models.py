import torch
import torch.nn as nn
import dataclasses
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

@dataclasses.dataclass
class MLPArgs:
    hiddens: list = dataclasses.field(default_factory=lambda: [20, 10])
    activation: str = 'relu'

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens=None, activation=None):
        super(MLP, self).__init__()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Identity() # NOTE: CE loss in PyTorch applies softmax internally

        # Build the MLP model
        mlp = []
        prev_hidden = input_size
        for h in hiddens:
            mlp.append(nn.Linear(prev_hidden, h))
            mlp.append(nn.ReLU())
            prev_hidden = h
        mlp.append(nn.Linear(prev_hidden, output_size))
        mlp.append(self.activation)
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x, lengths):
        # Average along the time axis
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)
        return self.mlp(x)
    


@dataclasses.dataclass
class LSTMArgs:
    num_layer: int = 1
    hidden_size: int = 32
    dropout: float = 0.2
    bidirectional: bool = False
    norm: bool = False
    activation: str = 'tanh'
    pad: bool = True

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1, dropout=0.2, bidirectional=False, norm=False, activation='tanh'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.norm = norm

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Normalization layer
        if norm:
            self.norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Identity() # NOTE: CE loss in PyTorch applies softmax internally
        else:
            raise ValueError("Unsupported activation function. Use 'tanh', 'relu', or 'sigmoid'.")
        
    def forward(self, x, lengths):
        # Pack the sequence for LSTM if needed
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x) # The output of the LSTM and the hidden states, we only need the last hidden state

        if self.norm:
            out = self.norm(h_n[-1]) # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        else:
            out = h_n[-1]
        
        # Final output layer
        out = self.output_layer(out) # [batch_size, output_size]
        out = self.activation(out)

        return out
    


@dataclasses.dataclass
class TransformerArgs:
    d_model: int = 512
    num_layer: int = 6
    nheads: int = 8
    dropout: float = 0.1


# Positional encoding class implemented in utils.py
class transformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layer, nheads, dropout):
        super(transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer block
        # HACK: 20250409, Do we need the transformer decoder here? Haven't finished.
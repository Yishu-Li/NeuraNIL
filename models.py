import torch
import torch.nn as nn
import dataclasses

@dataclasses.dataclass
class MLPArgs:
    hiddens: list = dataclasses.field(default_factory=lambda: [20, 10])

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens=None, activation=None):
        super(MLP, self).__init__()
        
        # Initialize the model and parameters
        if activation is None:
            activation = nn.ReLU()
        if hidden_size is None:
            hidden_size = [20, 10]

        # Build the MLP model
        mlp = []
        prev_hidden = input_size
        for h in hiddens:
            mlp.append(nn.Linear(prev_hidden, h))
            mlp.append(activation)
            prev_hidden = h
        mlp.append(nn.Linear(prev_hidden, output_size))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)
    


@dataclasses.dataclass
class LSTMArgs:
    num_layer: int = 1
    hidden_size: int = 100
    dropout: float = 0.2
    bidirectional: bool = False
    norm: bool = False
    activation: str = 'tanh'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1, dropout=0.2, bidirectional=False, norm=False, activation='tanh'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

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
        else:
            raise ValueError("Unsupported activation function. Use 'tanh', 'relu', or 'sigmoid'.")
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x) # The output of the LSTM and the hidden states, we only need the last hidden state

        if self.norm:
            out = self.norm(h_n[-1]) # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class transformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layer, nheads, dropout):
        super(transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Transformer block
        # HACK: 20250409, Do we need the transformer decoder here? Haven't finished.
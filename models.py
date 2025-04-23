import torch
import torch.nn as nn
import dataclasses
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

@dataclasses.dataclass
class MLPArgs:
    hiddens: list = dataclasses.field(default_factory=lambda: [20, 10])
    activation: str = 'relu'
    dropout: float = 0.2

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens=None, activation=None, dropout=0.2):
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
        mlp.append(nn.BatchNorm1d(input_size, affine=False, track_running_stats=False)) # Normalization is very important!!!
        for h in hiddens:
            mlp.append(nn.Linear(prev_hidden, h))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
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
    num_layer: int = 2
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

        if norm:
            self.norm = nn.BatchNorm1d(input_size, affine=False)
            # self.norm = nn.LayerNorm(input_size, elementwise_affine=False) # LayerNorm is more suitable for RNNs

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

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
        batch_size, seq_len, _ = x.shape
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        if self.norm:
            x = x.view(-1, self.input_size)
            x = self.norm(x)
            x = x.view(batch_size, seq_len, self.input_size)  

        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x) # The output of the LSTM and the hidden states, we only need the last hidden state

        out = h_n[-1]
        
        out = self.dropout(out) # Apply dropout to the last hidden state
        
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


@dataclasses.dataclass
class LDAArgs:
    n_components: int = 2


class LDA(nn.Module):
    def __init__(self, n_components=2):
        super(LDA, self).__init__()
        # Initialize the LDA model with the given arguments
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)

    def fit(self, X, y):
        # Fit the LDA model to the data
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.fit_transform(X)  # Normalize the data
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        self.lda.fit(X, y)

    def transform(self, X):
        # Transform the data using the fitted LDA model
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.transform(X)
        return torch.tensor(self.lda.transform(X))
    
    def fit_transform(self, X, y):
        # Fit the LDA model and transform the data
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.fit_transform(X)  # Normalize the data
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        return torch.tensor(self.lda.fit_transform(X, y))
    
    def forward(self, X):
        # Forward pass through the LDA model
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.transform(X)
        return torch.tensor(self.lda.predict(X))
    

class GNB(nn.Module):
    def __init__(self, n_components=2):
        super(GNB, self).__init__()
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler() # This is very important for GNB model!!!!!
        self.gnb = GaussianNB()

    def fit(self, X, y):
        # Fit the GNB model to the data
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.fit_transform(X) # To normalize the data
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        self.gnb.fit(X, y)

    def forward(self, X):
        # Predict using the fitted GNB model
        X = torch.mean(X, dim=1).numpy()
        X = self.scaler.transform(X)
        return torch.tensor(self.gnb.predict(X))
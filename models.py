import torch
import torch.nn as nn
import dataclasses
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

@dataclasses.dataclass
class MLPArgs:
    hiddens: list = dataclasses.field(default_factory=lambda: [20, 10])
    activation: str = 'relu'
    dropout: float = 0.2
    norm: bool = True

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens=None, activation=None, norm=False, dropout=0.2):
        super(MLP, self).__init__()
        self.norm = norm
        
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
        if norm:
            mlp.append(nn.BatchNorm1d(input_size, affine=False, track_running_stats=False)) # Normalization is very important!!!
        for h in hiddens:
            print(f"MLP hidden layer: {prev_hidden} -> {h}")
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
    ifconv: bool = False
    convoutput: int = 1

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ifconv=False, convoutput=1, num_layer=1, dropout=0.2, bidirectional=False, norm=False, activation='tanh'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.norm = norm
        self.ifconv = ifconv

        if norm:
            self.norm = nn.BatchNorm1d(input_size, affine=False)
            # self.norm = nn.LayerNorm(input_size, elementwise_affine=False) # LayerNorm is more suitable for RNNs

        # Conv1d layer
        if ifconv:
            self.conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size*convoutput,  # 3x input channels for LSTM input (input, hidden, cell)
                kernel_size=20,
                stride=10,
                padding=0,
            )
        else:
            convoutput = 1

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_size*convoutput,
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

        # Apply convolution to the input
        if self.ifconv:
            x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_size, seq_len) for Conv1d
            x = self.conv(x)  # Apply convolution
            x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_len, input_size)

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
    num_layers: int = 6
    nheads: int = 8
    dropout: float = 0.1


# Positional encoding class implemented in utils.py
class Transformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layers, nheads, dropout):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.nheads = nheads
        self.dropout = dropout

        # Transformer block
        self.norm = nn.BatchNorm1d(input_size, affine=False)
        from utils import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nheads,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(d_model, output_size)
        self.mask = None

    def forward(self, x, lengths):
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"Input size {input_size} does not match model input size {self.input_size}"

        # Apply normalization
        # Input to norm1d is (batch_size, input_size, seq_len)
        x = x.view(-1, self.input_size)
        x = self.norm(x)
        # Change back to (batch_size, seq_len, input_size)
        x = x.view(batch_size, seq_len, self.input_size)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder + decoder
        x = self.encoder(x)
        x = self.decoder(x)

        # Want output shape to be (batch_size, output_size) but currently it's (batch_size, seq_len, output_size)
        # Take the mean over the sequence length
        x = torch.mean(x, dim=1)
        return x


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
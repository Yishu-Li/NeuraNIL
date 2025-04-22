import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np


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
    

# To load the padded lstm signals
def collate_fn_lstm(batch):
    # batch: list of (sample, label, day_label, length)
    samples, labels, day_labels, lengths = zip(*batch)
    samples = pad_sequence(samples, batch_first=True, padding_value=0)
    if lengths[0] is not None:
        lengths = torch.tensor(lengths, dtype=torch.long)
    else:
        lengths = None
    labels = torch.stack(labels)
    day_labels = torch.stack(day_labels)
    return samples, labels, day_labels, lengths


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix using numpy.
    Args:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        num_classes: int, number of classes (if None, inferred from data)
    Returns:
        cm: (num_classes, num_classes) numpy array
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def parse_exclude_list(exclude):
    """
    Handles cases like ['[5]'], ['[5,6]'], [5], [], or None.
    Returns a list of ints.
    """
    if not exclude or exclude == []:
        return []
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(exclude[0], str) and exclude[0].startswith('['):
        items = exclude[0].strip('[]').split(',')
        return [int(x) for x in items if x.strip() != '']
    return [int(x) for x in exclude]
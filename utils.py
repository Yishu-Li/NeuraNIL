import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import TensorDataset, Subset
import matplotlib.pyplot as plt
from collections import defaultdict


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
    if len(batch[0]) == 4: 
        samples, labels, day_labels, lengths = zip(*batch)
        samples = pad_sequence(samples, batch_first=True, padding_value=0)
    elif len(batch[0]) == 3:
        samples, labels, day_labels = zip(*batch)
        lengths = [None] * len(samples)  # No lengths provided
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



def plot_lda(X_lda, y, run_name="", if_test=False):
    """
    Plot the LDA transformed data.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('LDA Projection')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.colorbar(scatter, label='Class Label')
    
    if if_test:
        save_path = f'results/{run_name}/lda_projection_test.png'
    else:
        save_path = f'results/{run_name}/lda_projection.png'
        
    if run_name != "":
        os.makedirs(os.path.dirname(f'results/{run_name}/'), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()



def support_query_split(batch, support_ratio=0.5):
    """
    Randomly split the dataset into support and query sets.
    """
    # Read the data from batch
    data, labels, day_labels, lengths = batch
    data = torch.stack(data)

    n_samples = len(data)
    indices = np.arange(n_samples)

    # Shuffle the indices for support and query
    np.random.shuffle(indices)
    support_size = int(n_samples * support_ratio)
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]

    # Extract support and query data
    support_data = data[support_indices]
    support_labels = labels[support_indices]
    support_day_labels = day_labels[support_indices]
    support_lengths = lengths[support_indices] if lengths is not None else None
    query_data = data[query_indices]
    query_labels = labels[query_indices]
    query_day_labels = day_labels[query_indices]
    query_lengths = lengths[query_indices] if lengths is not None else None
    return (support_data, support_labels, support_day_labels, support_lengths), \
           (query_data, query_labels, query_day_labels, query_lengths)



def split_by_day(dataset, train_days, test_days):
    '''
    Split the dataset into train and test sets based on the days.
    We can train on all days and refit the meta-classifier on the test days.
    '''

    day_labels = dataset.day_labels.numpy()
    train_indices = [i for i, day in enumerate(day_labels) if day in train_days]
    test_indices = [i for i, day in enumerate(day_labels) if day in test_days]

    # Create TensorDataset from the original dataset
    full_dataset = TensorDataset(dataset.data, dataset.labels, dataset.day_labels)
    if dataset.lengths is not None:
        # If lengths exist, add as a fourth tensor
        full_dataset = TensorDataset(dataset.data, dataset.labels, dataset.day_labels, dataset.lengths)

    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)

    # Print the labels and days in each subset
    from dataset import print_labels_and_days
    print_labels_and_days(train_set, "Train")
    print_labels_and_days(test_set, "Test")
    return train_set, test_set


class DaySampler(torch.utils.data.Sampler):
    '''
    A sampler that samples data using day labels.
    '''
    def __init__(self, days_labels):
        self.days_labels = days_labels
        self.batches = []
        self.label_to_indices = defaultdict(list)

        for idx, day in enumerate(days_labels):
            self.label_to_indices[int(day)].append(int(idx))

        for day in self.label_to_indices.keys():
            self.batches.append(self.label_to_indices[day])

        print(f"DaySampler: {len(self.batches)} batches created.")
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)
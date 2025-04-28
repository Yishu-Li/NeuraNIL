import torch
from pathlib import Path
import numpy as np
import dataclasses
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.functional import one_hot
from collections import Counter

def find_session_dirs(dataset, excludeSession=None):
    """
    Find all session directories in the given session directories.
    """

    use_session_dirs = []
    if dataset == "BG":
        session_dirs = Path("./Data/BG")
    elif dataset == "FALCON":
        session_dirs = Path("./Data/FALCON")
    
    # Iterate through all session directories
    for ses_dir in session_dirs.iterdir():
        if ses_dir.is_dir():
            excludeFlag = 0

            # Exclude
            if excludeSession is not None:
                for exSes in excludeSession:
                    if exSes in ses_dir.name:
                        excludeFlag = 1
                        break
            

            if not excludeFlag:
                use_session_dirs.append(ses_dir)
    
    # Sort the block directories
    use_session_dirs.sort()

    return use_session_dirs


def load_data_with_daylabels(dataset):
    """
    Load data with day labels.
    """
    session_dirs = find_session_dirs(dataset)
    data = []
    labels = []
    day_labels = []
    for i, ses_dir in enumerate(session_dirs):
        # Load the data from the session directory
        data.append(np.load(ses_dir / "data.npz")['X'])
        labels.append(np.load(ses_dir / "data.npz")['y'])
        day_labels.append([i] * len(labels[-1]))

        # Print the number of trials in each session
        print(f"Session {ses_dir.name} has {len(labels[-1])} trials.")

    return data, labels, day_labels


# Print unique labels and day labels with their counts for each split
def print_labels_and_days(dataset_split, split_name):
    all_labels = []
    all_day_labels = []
    for i in range(len(dataset_split)):
        sample = dataset_split[i]
        if isinstance(sample, dict):
            all_labels.append(sample['label'])
            all_day_labels.append(sample['day_label'])
        else:
            all_labels.append(sample[1].item() if hasattr(sample[1], 'item') else sample[1])
            all_day_labels.append(sample[2].item() if hasattr(sample[2], 'item') else sample[2])

    label_counts = Counter(all_labels)
    day_counts = Counter(all_day_labels)

    print(f"{split_name} label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count}")

    print(f"{split_name} day label counts:")
    for day, count in sorted(day_counts.items()):
        print(f"  Day {day}: {count}")




class NeuralDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, day_labels, lstm_pad=False, labels_exclude=set(), days_exclude=set()):
        # Convert the data to PyTorch tensors
        self.data = [torch.tensor(s, dtype=torch.float32) for d in data for s in d]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self.day_labels = [torch.tensor(dl, dtype=torch.long) for dl in day_labels]

        # Pad the data if lstm_pad is True
        original_length = [d.shape[0] for d in self.data]
        if lstm_pad:
            self.lengths = [torch.tensor([d.shape[1]] * d.shape[0]) for d in data]
            self.lengths = torch.cat(self.lengths)[:, torch.newaxis]
            max_length = max(original_length)
            self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
            print(f"Data padded to max length: {max_length}, now shape: {self.data.shape}")
        else:
            self.lengths = None
            min_length = min(original_length)
            self.data = [d[:min_length, :] for d in self.data]
            self.data = torch.stack(self.data)
            print(f"Data truncated to min length: {min_length}, now shape: {self.data.shape}")

        # Convert labels and day_labels to tensors
        self.labels = torch.cat(self.labels)
        self.day_labels = torch.cat(self.day_labels)

        # NOTE: Do not do the one-hot encoding for labels, PyTorch's CE loss does it for you!!!!!

        # Exclude labels and day_labels if specified
        if labels_exclude:
            mask = torch.tensor([int(label.item()) not in labels_exclude for label in self.labels])
            self.data = self.data[mask, :, :]
            self.labels = self.labels[mask]
            self.day_labels = self.day_labels[mask]
        if days_exclude:
            mask = torch.tensor([int(day_label.item()) not in days_exclude for day_label in self.day_labels])
            self.data = self.data[mask, :, :]
            self.labels = self.labels[mask]
            self.day_labels = self.day_labels[mask]

        # Change the labels to range(0, num_classes)
        label_mapping = {label: i for i, label in enumerate(set(self.labels.tolist()))}
        self.labels = torch.tensor([label_mapping[label.item()] for label in self.labels], dtype=torch.long)

        self.num_samples = len(self.data)
        self.num_classes = len(set(label for label in self.labels.tolist()))
        self.num_days = len(set(day for day in self.day_labels.tolist()))
        self.num_features = data[0].shape[-1]

        print("Number of samples:", self.num_samples)
        print("Number of classes:", self.num_classes)
        print("Number of days:", self.num_days)
        print("Number of features:", self.num_features)

        print("Data shape:", self.data.shape)
        print("Labels shape:", self.labels.shape)
        print("Day labels shape:", self.day_labels.shape)


    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        samples = self.data[idx, :, :] if self.lengths is None else self.data[idx]
        label = self.labels[idx]
        day_label = self.day_labels[idx]
        length = None if self.lengths is None else self.lengths[idx].item()
        return samples, label, day_label, length






if __name__ == "__main__":
    # For debugging only
    dataset = "BG"
    use_session_dirs = find_session_dirs(dataset)

    data, labels, day_labels = load_data_with_daylabels(dataset)
    
    neural_dataset = NeuralDataset(data, labels, day_labels, lstm_pad=True)
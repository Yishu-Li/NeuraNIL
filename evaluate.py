import torch
import numpy as np
from utils import compute_confusion_matrix

def evaluate_model(model, data_loader, device, loss_fn, stats_prefix=""):
    """
    Evaluate the model on the given data loader.
    """
    
    all_labels = []
    all_preds = []
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch in data_loader:
            if hasattr(model, 'learner'):
                # TODO: Implement the evaluation behavior for NeuraNIL
                raise NotImplementedError("NeuraNIL evaluation not implemented yet.")
            else:
                data, labels, day_labels, lengths = batch
                if lengths is not None:
                    data, labels, day_labels, lengths = data.to(device), labels.to(device), day_labels.to(device), lengths.to(device)
                else:
                    data, labels, day_labels = data.to(device), labels.to(device), day_labels.to(device)
                    lengths = None
                
                # Forward pass
                y_pred = model(data, lengths)
                
                # Calculate loss
                batch_loss = loss_fn(y_pred, labels)
                total_loss += batch_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(y_pred, dim=1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Collect for confusion matrix
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total_samples
        print(f"{stats_prefix} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Plot confusion matrix for test set
        if stats_prefix == "Test":
            import matplotlib.pyplot as plt
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            num_classes = max(y_true.max(), y_pred.max()) + 1
            cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
            cm = cm.T
            # Normalize by true label count (per column)
            true_label_counts = cm.sum(axis=0, keepdims=True)
            cm_norm = cm / (true_label_counts + 1e-8)
            plt.figure(figsize=(8, 8))
            plt.imshow(cm_norm, interpolation='nearest', cmap="Blues", vmin=0, vmax=1)
            plt.title("Confusion Matrix (Test Set)")
            plt.colorbar()
            tick_marks = np.arange(num_classes)
            plt.xticks(tick_marks)
            plt.yticks(tick_marks)
            plt.xlabel("True label")
            plt.ylabel("Predicted label")
            # Show values in cells (with 2 decimals)
            thresh = 0.5
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                             ha="center", va="center",
                             color="white" if cm_norm[i, j] > thresh else "black")
            plt.tight_layout()
            import os
            os.makedirs("results", exist_ok=True)
            plt.savefig("results/confusion_matrix_test.png")
            plt.close()
        return avg_loss, accuracy
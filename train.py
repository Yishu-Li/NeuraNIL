import os
import torch
import tqdm
import matplotlib.pyplot as plt

from evaluate import evaluate_model

def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, epochs, run_name=""):
    """
    Plot training and validation loss and accuracy curves.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    save_path = f'results/{run_name}/training_curves.png'
    os.makedirs(os.path.dirname(f'results/{run_name}/'), exist_ok=True)
    plt.savefig(save_path)

def train(model, device, train_loader, valid_loader, opt, loss_fn, run_name, args):
    """
    Train the model.
    """
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # for epoch in tqdm.trange(args.options.epochs):
    for epoch in range(args.options.epochs):
        # Train step
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            opt.zero_grad()
            if hasattr(model, 'learner'):
                raise NotImplementedError("NeuraNIL training not implemented yet.")
            else:
                data, labels, day_labels, lengths = batch
                data, labels, day_labels = data.to(device), labels.to(device), day_labels.to(device)
                y_pred = model(data, lengths)
                batch_loss = loss_fn(y_pred, labels)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.item()
                
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Training accuracy
        model.eval()
        train_loss, train_accuracy = evaluate_model(model, train_loader, device, loss_fn, stats_prefix="Train")
        train_accuracies.append(train_accuracy)

        # Validation step
        valid_loss, valid_accuracy = evaluate_model(model, valid_loader, device, loss_fn, stats_prefix="Validation")
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Display training process
        print(f"\rEpoch {epoch+1}/{args.options.epochs} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.2f} | Val Loss: {valid_loss:.4f} Acc: {valid_accuracy:.2f}", end="", flush=True)

    # Plot training and validation loss and accuracy
    plot_training_curves(
        train_losses, valid_losses, train_accuracies, valid_accuracies, args.options.epochs, run_name=run_name
    )

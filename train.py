import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from evaluate import evaluate_model
import models
import utils

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



def train(model, device, train_loader, valid_loader, run_name, args):
    """
    Train the model.
    """

    if isinstance(model, models.LDA) or isinstance(model, models.GNB):
        # Get all the data and labels from the data_loader
        all_data = []
        all_labels = []
        for data, labels, _, _ in train_loader:
            all_data.append(data)
            all_labels.append(labels)
        for data, labels, _, _ in valid_loader:
            all_data.append(data)
            all_labels.append(labels)
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        print(f'Train data shape: {all_data.shape}')
        print(f'Train labels shape: {all_labels.shape}')

        if isinstance(model, models.LDA):
            # Fit the LDA model
            X_lda = model.fit_transform(all_data, all_labels)
            print("LDA model fitted on the training data.")
            utils.plot_lda(X_lda.numpy(), all_labels.numpy(), run_name=run_name)
        else:
            # Fit the GNB model
            model.fit(all_data, all_labels)
            print("GNB model fitted on the training data.")

        # Calculate the training loss and accuracy
        model.eval()
        train_loss, train_accuracies = evaluate_model(model, train_loader, device, loss_fn, stats_prefix="Train", run_name=run_name)
        print(f"Training Accuracy: {train_accuracies:.2f}")
        return



    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Initialize the optimizer
    if hasattr(model, 'learner'):
        opt = torch.optim.Adam(
            list(model.learner.parameters()),
            lr=args.options.lr
        )
    else: 
        opt = torch.optim.Adam(model.parameters(), lr=args.options.lr)

    # Loss function
    loss_fn = nn.CrossEntropyLoss() 


    # for epoch in tqdm.trange(args.options.epochs):
    for epoch in range(args.options.epochs):
        # Train step
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            opt.zero_grad()
            if hasattr(model, 'learner'):
                # Split the batch into support and query sets for meta-learning
                support_ratio = args.meta.support_ratio
                (support_x, support_y, _, support_lengths), \
                (query_x, query_y, _, query_lengths) = utils.support_query_split(batch, support_ratio)
                support_x, support_y = support_x.to(device), support_y.to(device)
                query_x, query_y = query_x.to(device), query_y.to(device)

                # Forward pass
                query_pred = model(support_x, support_y, query_x, support_lengths, query_lengths)
                batch_loss = loss_fn(query_pred, query_y)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.item()
            else:
                data, labels, day_labels, lengths = batch
                data, labels, day_labels = data.to(device), labels.to(device), day_labels.to(device)
                y_pred = model(data, lengths)
                batch_loss = loss_fn(y_pred, labels)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.item()
                support_ratio = 0  # Not used in standard training

        # Training accuracy
        # model.eval() # NOTE: Interferes with the meta-learning inner loop update
        train_loss, train_accuracy = evaluate_model(args, model, train_loader, device, stats_prefix="Train")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation step
        valid_loss, valid_accuracy = evaluate_model(args, model, valid_loader, device, stats_prefix="Validation")
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Display training process
        print(f"\rEpoch {epoch+1}/{args.options.epochs} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.2f} | Val Loss: {valid_loss:.4f} Acc: {valid_accuracy:.2f}", end="", flush=True)

    # Plot training and validation loss and accuracy
    plot_training_curves(
        train_losses, valid_losses, train_accuracies, valid_accuracies, args.options.epochs, run_name=run_name
    )

import torch
import numpy as np
import dataclasses
from pathlib import Path
import simple_parsing as sp
import random
# import wandb

import models
import dataset
from evaluate import evaluate_model
from utils import collate_fn_lstm, parse_exclude_list



def setup_model(args, input_size, output_size):
    """
    Setup the model based on the arguments.
    """
    if args.options.model == "LSTM":
        model = models.LSTM(
            input_size=input_size,
            hidden_size=args.lstm.hidden_size,
            output_size=output_size,
            num_layer=args.lstm.num_layer,
            dropout=args.lstm.dropout,
            bidirectional=args.lstm.bidirectional,
            norm=args.lstm.norm,
            activation=args.lstm.activation,
        )
    elif args.options.model == "MLP":
        model = models.MLP(
            input_size=input_size,
            output_size=output_size,
            hiddens=args.mlp.hiddens,
            activation=args.mlp.activation,
        )
    elif args.options.model == "Transformer":
        model = models.Transformer(
            input_size=input_size,
            output_size=output_size,
            num_layers=args.transformer.num_layers,
            num_heads=args.transformer.num_heads,
            d_model=args.transformer.d_model,
            dropout=args.transformer.dropout,
        )
    elif args.options.model == "NeuraNIL":
        # TODO: Implement NeuraNIL model
        raise NotImplementedError("NeuraNIL model is not implemented yet.")
    else:
        raise ValueError(f"Unknown model type: {args.options.model}")
    
    return model
    



def main():
    # ---------------------------- Setup arguments -----------------------------
    @dataclasses.dataclass
    class NeuralDatasetArgs:
        data: str = "BG"
        split_ratio: float = 0.8
        random_split: bool = False
        labels_exclude: list = dataclasses.field(default_factory=list)
        days_exclude: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class TrainArgs:
        model: str = "LSTM"
        model_learner: str = "LSTM"
        model_classifier: str = "MLP"
        epochs: int = 50
        cuda: bool = True
        seed: int = 42
        project: str = "NeuraNIL"
        model_path: str = None


    @dataclasses.dataclass
    class MetaArgs:
        n_shots: int = 10
        n_queries: int = 10
        hidden_size: int = 8


    # Create the args
    parser = sp.ArgumentParser(add_dest_to_option_strings=True)
    parser.add_arguments(NeuralDatasetArgs, dest="dataset")
    parser.add_arguments(TrainArgs, dest="options")
    parser.add_arguments(MetaArgs, dest="meta")
    parser.add_arguments(models.MLPArgs, dest="mlp")
    parser.add_arguments(models.LSTMArgs, dest="lstm")
    parser.add_arguments(models.TransformerArgs, dest="transformer")
    args = parser.parse_args()


    # -------------------------- Setup torch device ----------------------------
    # HACK: Idk how to create a wandb team
    random.seed(args.options.seed)
    np.random.seed(args.options.seed)
    torch.manual_seed(args.options.seed)
    if args.options.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.options.seed)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # --------------- Start a new wandb to track this script ------------------
    if 'BG' in args.dataset.data:
        data_tag = "t17"
    elif 'FALCON' in args.dataset.data:
        data_tag = "FALCON"

    if args.options.model == "NeuraNIL":
        model_details = f'{args.options.model_learner}_{args.options.model_classifier}_{args.meta.hidden_size}dims'
        model_details += f'_{args.meta.n_shots}shots_{args.meta.n_queries}queries'
    else:
        model_details = ''

    if args.dataset.random_split:
        model_details += f'_random_split_{args.dataset.split_ratio}'
    if args.dataset.labels_exclude:
        model_details += f'_labels_exclude_{"_".join(args.dataset.labels_exclude)}'
    if args.dataset.days_exclude:
        model_details += f'_days_exclude_{"_".join(args.dataset.days_exclude)}'

    wandb_name = f"{data_tag}_{args.options.model}_{model_details}"

    # wandb.init(
    #     entity='CSCI-1470',
    #     project=args.options.project,
    #     name=wandb_name,
    # )


    # ---------------------------- Setup datasets ------------------------------
    train_dataset = []
    valid_dataset = []
    test_dataset = []

    # Initialize the dataset
    data, labels, day_labels = dataset.load_data_with_daylabels(args.dataset.data)

    # Parse exclude lists using the utility function
    labels_exclude = parse_exclude_list(args.dataset.labels_exclude)
    days_exclude = parse_exclude_list(args.dataset.days_exclude)

    neural_dataset = dataset.NeuralDataset(
        data, labels, day_labels, 
        lstm_pad=args.lstm.pad,
        labels_exclude=labels_exclude,
        days_exclude=days_exclude,
    )

    # Split the dataset into train, valid, and test sets
    if args.dataset.random_split:
        # Randomly split the dataset
        total_size = len(neural_dataset)
        train_size = int(total_size * args.dataset.split_ratio)
        valid_size = int((total_size - train_size) / 2)
        test_size = total_size - train_size - valid_size

        from torch.utils.data import random_split
        train_set, val_set, test_set = random_split(neural_dataset, [train_size, valid_size, test_size])

        # Print the labels and days in each subset
        from dataset import print_labels_and_days
        print_labels_and_days(train_set, "Train")
        print_labels_and_days(val_set, "Validation")
        print_labels_and_days(test_set, "Test")


    # Read the dataset infos
    n_features = neural_dataset.num_features
    n_classes = neural_dataset.num_classes
    n_days = neural_dataset.num_days


    # ---------------------------- Setup model -------------------------------
    model = setup_model(args, n_features, n_classes)

    # Read the model if the path is given
    if args.options.model_path:
        model.load_state_dict(torch.load(args.options.model_path))
        print(f"Model loaded from {args.options.model_path}")
        skip_training = True
    else:
        skip_training = False

    # Move the model to the device
    model.to(device)
    print(f"Model: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ---------------------------- Setup dataloader -----------------------------
    # HACK: Probably have to change if we want to include the day labels in meta-learning process
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_lstm)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(valid_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")


    # --------------------------------- Training ----------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    if not skip_training:
        from train import train
        # Train the model
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            opt=opt,
            loss_fn=loss_fn,
            args=args,
            run_name=wandb_name,  # Use the wandb name for saving results
        )


    # --------------------------------- Testing ----------------------------------
    if len(test_loader.dataset) > 0:
        model.eval()
        evaluate_model(
            model=model,
            device=device,
            data_loader=test_loader,
            loss_fn=loss_fn,
            stats_prefix="Test",
            run_name=wandb_name,  # Use the wandb name for saving results
        )




if __name__ == "__main__":
    main()
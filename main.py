import torch
import numpy as np
import dataclasses
from pathlib import Path
import simple_parsing as sp
import random
# import wandb

import models
from NeuraNIL import NeuraNIL
import dataset
from evaluate import evaluate_model
from utils import collate_fn_lstm, parse_exclude_list
import utils



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
            ifconv=args.lstm.ifconv,
            convoutput=args.lstm.convoutput,
        )
    elif args.options.model == "MLP":
        model = models.MLP(
            input_size=input_size,
            output_size=output_size,
            hiddens=args.mlp.hiddens,
            activation=args.mlp.activation,
            dropout=args.mlp.dropout,
            norm=args.mlp.norm,
        )
    elif args.options.model == "Transformer":
        model = models.Transformer(
            input_size=input_size,
            output_size=output_size,
            num_layers=args.transformer.num_layers,
            nheads=args.transformer.nheads,
            d_model=args.transformer.d_model,
            dropout=args.transformer.dropout,
        )
    elif args.options.model == "LDA":
        model = models.LDA(
            n_components=args.lda.n_components,
        )
    elif args.options.model == "GNB":
        model = models.GNB()
    elif args.options.model == "NeuraNIL":
        # Setup the learner and classifierrandom
        args.options.model = args.meta.learner
        learner = setup_model(args, input_size, args.meta.hidden_size)
        args.options.model = args.meta.classifier
        args.mlp.norm = False
        classifier = setup_model(args, args.meta.hidden_size, output_size)

        # Setup the NeuraNIL model
        model = NeuraNIL(
            learner=learner,
            classifier=classifier,
            k=args.meta.k,
            inner_lr=args.meta.inner_lr,
        )

        # Set the arg back
        args.options.model = "NeuraNIL"
    else:
        raise ValueError(f"Unknown model type: {args.options.model}")
    
    return model
    



def main():
    # ---------------------------- Setup arguments -----------------------------
    @dataclasses.dataclass
    class NeuralDatasetArgs:
        data: str = "BG"
        split_ratio: float = 0.8
        split_method: str = "random"  # Options: 'random', 'day'
        train_days: list = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_days: list = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels_exclude: list = dataclasses.field(default_factory=list)
        days_exclude: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class TrainArgs:
        model: str = "LSTM"
        epochs: int = 50
        cuda: bool = True
        seed: int = 42
        model_path: str = None
        lr: float = 0.001

    @dataclasses.dataclass
    class MetaArgs:
        support_ratio: float = 0.5
        hidden_size: int = 8
        learner: str = "LSTM"
        classifier: str = "MLP"
        k: int = 5
        inner_lr: float = 0.001


    # Create the args
    parser = sp.ArgumentParser(add_dest_to_option_strings=True)
    parser.add_arguments(NeuralDatasetArgs, dest="dataset")
    parser.add_arguments(TrainArgs, dest="options")
    parser.add_arguments(MetaArgs, dest="meta")
    parser.add_arguments(models.MLPArgs, dest="mlp")
    parser.add_arguments(models.LSTMArgs, dest="lstm")
    parser.add_arguments(models.TransformerArgs, dest="transformer")
    parser.add_arguments(models.LDAArgs, dest="lda")
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

    print(f'Model: {args.options.model}')
    if args.options.model == "NeuraNIL":
        model_details = f'{args.meta.learner}_{args.meta.classifier}_{args.meta.hidden_size}dims'
        # model_details += f'_{args.meta.n_shots}shots_{args.meta.n_queries}queries'
    else:
        model_details = ''
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
    if args.dataset.split_method == "random":
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

    elif args.dataset.split_method == "day":
        # Split the dataset by day
        train_days = parse_exclude_list(args.dataset.train_days)
        test_days = parse_exclude_list(args.dataset.test_days)
        print(f"Train days: {train_days}")
        print(f"Test days: {test_days}")

        from utils import split_by_day
        train_set, test_set = split_by_day(
            neural_dataset,
            train_days=train_days,
            test_days=test_days,
            )
        val_set = test_set  


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
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_lstm)
    # valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
    if args.dataset.split_method == "day":
        val_day_labels = [neural_dataset.day_labels[idx] for idx in val_set.indices]
        test_dat_labels = [neural_dataset.day_labels[idx] for idx in test_set.indices]

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_lstm)
        valid_loader = torch.utils.data.DataLoader(val_set, batch_sampler=utils.DaySampler(val_day_labels), collate_fn=collate_fn_lstm)
        test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=utils.DaySampler(test_dat_labels), collate_fn=collate_fn_lstm)
    elif args.dataset.split_method == "random":
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_lstm)
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn_lstm)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(valid_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")


    # --------------------------------- Training ----------------------------------
    if not skip_training:
        from train import train
        # Train the model
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            args=args,
            run_name=wandb_name,  # Use the wandb name for saving results
        )


    # --------------------------------- Testing ----------------------------------
    if len(test_loader.dataset) > 0:
        if hasattr(model, 'learner'):
            # Set the learner to eval mode for meta-learning
            model.learner.eval()
        else:
            model.eval()
        test_loss,  test_acc = evaluate_model(
            args=args,
            model=model,
            device=device,
            data_loader=test_loader,
            stats_prefix="Test",
            run_name=wandb_name,  # Use the wandb name for saving results
        )

        print(f'\n{"*"*35} Test Results {"*"*35}')
        print(f'Test Loss: {test_loss} | Test Accuracy: {test_acc}')




if __name__ == "__main__":
    main()
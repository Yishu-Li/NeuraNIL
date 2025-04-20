import torch
import numpy as np
import dataclasses
from pathlib import Path
import simple_parsing as sp
import random
import wandb

import models




def main():
    # ---------------------------- setup arguments -----------------------------
    @dataclasses.dataclass
    class NeuralDatasetArgs:
        data: str = "BG"
        split_ratio: float = 0.8

    @dataclasses.dataclass
    class TrainArgs:
        model: str = "LSTM"
        model_learner: str = "LSTM"
        model_classifier: str = "MLP"
        cuda: bool = True
        seed: int = 42
        project: str = "NeuraNIL"

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


    # -------------------------- setup torch device ----------------------------
    random.seed(args.options.seed)
    np.random.seed(args.options.seed)
    torch.manual_seed(args.options.seed)
    if args.options.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.options.seed)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # --------------- start a new wandb to track this script ------------------
    if 'BG' in args.dataset.data:
        data_tag = "t17"
    elif 'FALCON' in args.dataset.data:
        data_tag = "FALCON"

    if args.options.model == "NeuraNIL":
        model_details = f'{args.options.model_learner}_{args.options.model_classifier}_{args.meta.hidden_size}dims'
        model_details += f'_{args.meta.n_shots}shots_{args.meta.n_queries}queries'
    else:
        model_details = ''

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
    




if __name__ == "__main__":
    main()
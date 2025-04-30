import sys, os
import matplotlib.pyplot as plt
import numpy as np
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
print(f"Project root: {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
import main as main_mod


def build_args(labels_to_exclude):
    train_days = range(0, 10)
    test_days = range(0, 10)

    sys.argv[:] = [
        "main.py",
        "--data","FALCON",
        "--dataset.split_method","random",
        "--model","NeuraNIL",
        "--epochs","20",
        "--mlp.hiddens","[32]",
        "--mlp.activation","softmax",
        "--mlp.dropout","0.2",
        "--mlp.norm","True",
        "--lstm.ifconv","True",
        "--lstm.num_layer","1",
        "--lstm.norm","True",
        "--meta.learner", "MLP",
        "--meta.classifier", "MLP",
        "--meta.k","5",
        "--meta.n_support", "8",
        "--train_days","%s" % str(list(train_days)),
        "--test_days","%s" % str(list(test_days)),
        "--batch_size", "128",
        "--dataset.labels_exclude", f"{labels_to_exclude}",
    ]

def train_old_tasks(tasks, model=None):
    labels_to_exclude = [i for i in range(26) if i not in tasks]
    print(f"Training on old tasks, excluding labels: {labels_to_exclude}")
    build_args(labels_to_exclude)
    if model is None:
        model, _, _ = main_mod.main()
    else:
        model, _, _ = main_mod.main(model, keep_training=True)
    return model

def test_new_tasks(model):
    labels_to_exclude = np.arange(4, 26).tolist()
    print(f"Testing on new tasks, excluding labels: {labels_to_exclude}")
    build_args(labels_to_exclude)
    _, _, acc = main_mod.main(model)
    return acc
    


if __name__=="__main__":
    model = train_old_tasks([4, 5, 6, 7])
    for i in range(5, 22):
        model = train_old_tasks([i, i+1, i+2, i+3], model=model)
    acc = test_new_tasks(model)
    print(f"Accuracy on new tasks: {acc:.2f}")
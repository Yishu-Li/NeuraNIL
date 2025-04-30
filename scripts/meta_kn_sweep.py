import sys, os
import matplotlib.pyplot as plt
import numpy as np
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
print(f"Project root: {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
import main as main_mod


def build_args(k, n):
    train_days = range(0, 6)
    test_days = [6, 7, 8, 9, 10]

    sys.argv[:] = [
        "main.py",
        "--data","FALCON",
        "--dataset.split_method","day",
        "--model","NeuraNIL",
        "--epochs","50",
        "--mlp.hiddens","[32]",
        "--mlp.activation","softmax",
        "--mlp.dropout","0.2",
        "--mlp.norm","True",
        "--lstm.ifconv","True",
        "--lstm.num_layer","1",
        "--lstm.norm","True",
        "--meta.learner", "MLP",
        "--meta.classifier", "MLP",
        "--meta.k","%d" % k,
        "--meta.n_support", "%d" % n,
        "--train_days","%s" % str(list(train_days)),
        "--test_days","%s" % str(list(test_days)),
        "--batch_size", "256",
    ]

K_LIST = [0, 2, 5, 10, 20, 50]
N_LIST = [1, 2, 3, 4]

def run_k_sweep():
    # sweep over support counts (n) and shots (k)
    accuracy_matrix = []
    for n in N_LIST:
        row = []
        for k in K_LIST:
            build_args(k, n)
            _, _, acc = main_mod.main()
            row.append(acc)
        accuracy_matrix.append(row)
    return accuracy_matrix

def plot_k_sweep(accuracy_matrix):
    """Plot heatmap of accuracy for each n (rows) and k (columns)."""
    acc_arr = np.array(accuracy_matrix)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(acc_arr, aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(im, label='Average Accuracy')
    plt.xticks(range(len(K_LIST)), K_LIST)
    plt.yticks(range(len(N_LIST)), N_LIST)
    plt.xlabel('k (shots)')
    plt.ylabel('n_support')
    plt.title('Accuracy Heatmap')
    plt.tight_layout()
    plt.savefig('results/kn_heatmap.png')
    plt.show()
    print("K-N sweep heatmap saved to results/kn_heatmap.png")

if __name__=="__main__":
    accuracy_matrix = run_k_sweep()
    plot_k_sweep(accuracy_matrix)
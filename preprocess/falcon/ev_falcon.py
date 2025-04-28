
"""
1) Preprocess FALCON H2 into Data/FALCON/H2/data.npz
2) Train & evaluate MLP, LSTM, Transformer, and CNN via main.py

!!!!Run from project root:
  python preprocess/falcon/ev_falcon.py
"""

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
from falcon_preprocess import main as preprocess_h2
preprocess_h2()
import main as main_mod


def build_args(model_name):
    sys.argv[:] = [
        "main.py",
        "--data","FALCON",
        "--split_ratio","0.8",
        "--model",model_name,
        "--epochs","30",
        "--lr","0.001",
        "--num_layer","2",
        "--lstm.hidden_size","32",
        "--lstm.dropout","0.2",
        "--bidirectional","False",
        "--hiddens","[50,20]",
        "--mlp.activation","leaky_relu",
        "--mlp.dropout","0.2",
        "--d_model","128",
        "--num_layers","4",
        "--nheads","4",
        "--transformer.dropout","0.1",
        "--n_components","2",
    ]
    
    
if __name__=="__main__":
    build_args("MLP");    main_mod.main()
    build_args("LSTM");   main_mod.main()
    build_args("Transformer"); main_mod.main()
    build_args("LDA");    main_mod.main()
    build_args("GNB");    main_mod.main()

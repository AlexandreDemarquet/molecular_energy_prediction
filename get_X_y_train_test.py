import numpy as np
from dataset.QM7XDataset import QM7XEmbedDataset
from pathlib import Path
import torch
import pickle
from sklearn.model_selection import train_test_split

import numpy as np


# === 1. CONFIGURATION ===
DATA_ROOT   = Path("data")
XYZ_DIR     = DATA_ROOT / "atoms" / "train"
ENERGY_CSV  = DATA_ROOT / "energies" / "train.csv"
# Force CPU to avoid torch_cluster CUDA issues
DEVICE      = torch.device('cpu')
ATOM_TYPES  = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']

def get_X_y_train_test(avec_coef_contrastive=False):
    """
    Load the dataset and split it into training and testing sets.
    Args:
        avec_coef_contrastive (bool): If True, include contrastive embedding coefficients in the features.
                                        Else only scatering coefficients are used.
    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
   
    scattering = pickle.load(open("scattering_coef_64_64_64.pkl", "rb"))
    # scattering = pickle.load(open("scattering_coef_train_32_32_32.pkl", "rb"))
    X = scattering
    if avec_coef_contrastive:
        embeding = pickle.load(open("coeff_embbeding.pkl", "rb"))
        X = np.hstack([scattering, embeding])
    embed_dataset = QM7XEmbedDataset(XYZ_DIR, ENERGY_CSV)
    feature_list, y_list = [], []
    for idx in range(len(embed_dataset)):
        data, energy = embed_dataset[idx]
        y_list.append(energy)

    y = np.array(y_list) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    return X_train, X_test, y_train, y_test
    



import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import torch
import time
import os
from sklearn import (linear_model, model_selection, preprocessing,
                     pipeline)
from scipy.spatial.distance import pdist
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians
from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
import os
import csv
import numpy as np
import pickle
from get_scat_coeff import get_scat_coeff
# === PARAMÈTRES ===
data_dir = "data"
atoms_dir = os.path.join(data_dir, "atoms", "train")
energies_file = os.path.join(data_dir, "energies", "train.csv")

# === MAPPAGE DES SYMBOLES ATOMIQUES VERS NUMÉROS ATOMIQUES ===
symbol_to_number = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    # tu peux en ajouter d'autres si besoin
}

# === LECTURE DES ÉNERGIES ===
energies_dict = {}
with open(energies_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mol_id = int(row["id"])
        energy = float(row["energy"])
        energies_dict[mol_id] = energy

# === EXTRACTION DES .XYZ ===
positions_list = []
charges_list = []
energies_list = []

max_atoms = 0
molecule_ids = sorted(energies_dict.keys())

for mol_id in molecule_ids:
    filename = f"id_{mol_id}.xyz"
    filepath = os.path.join(atoms_dir, filename)

    with open(filepath, "r") as f:
        lines = f.readlines()[2:]  # skip the 2 header lines

    charges = []
    positions = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            symbol = parts[0]
            xyz = list(map(float, parts[1:]))

            charges.append(symbol_to_number.get(symbol, 0))  # 0 pour atomes inconnus
            positions.append(xyz)

    num_atoms = len(positions)
    max_atoms = max(max_atoms, num_atoms)

    charges_list.append(charges)
    positions_list.append(positions)
    energies_list.append(energies_dict[mol_id])

# === TRANSFORMATION EN TABLEAUX AVEC PADDING ===
num_molecules = len(charges_list)

positions_array = np.zeros((num_molecules, max_atoms, 3), dtype=np.float32)
charges_array = np.zeros((num_molecules, max_atoms), dtype=np.int32)

for i in range(num_molecules):
    n = len(positions_list[i])
    positions_array[i, :n, :] = positions_list[i]
    charges_array[i, :n] = charges_list[i]

energies_array = np.array(energies_list, dtype=np.float32)

# === DICTIONNAIRE FINAL ===
dataset = {
    "positions": positions_array,
    "charges": charges_array,
    "energies": energies_array
}
n_molecules = len(dataset['energies'])

scattering_coef = pickle.load(open("scattering_coef_64_64_64.pkl", "rb"))    


n_folds = 5
P = np.random.permutation(n_molecules)
folds = np.array_split(P, n_folds)
cross_val_folds = []

for i in range(n_folds):
    val_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(n_folds) if j != i])

    cross_val_folds.append({
        "train_idx": train_idx,
        "val_idx": val_idx
    })


# Set the desired hyperparameters grid
param_grid = {
    'boosting__n_estimators': [ 200],
    'boosting__learning_rate': [  0.0001],
    'boosting__estimator__alpha': [ 1e-6]
}

# Define the base regressor with the specified alpha
ridge = linear_model.Ridge()
# Create the AdaBoost regressor
boosting_regressor = AdaBoostRegressor(estimator=ridge)

# Create the pipeline
poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    
scaler = preprocessing.StandardScaler()
regressor = pipeline.Pipeline([ 
    ('pca', PCA(n_components=50)),
    ('poly', poly),
    ('scaler', scaler),
    ('boosting', boosting_regressor)
])

# === Pipeline standardisation + Ridge ===
# scaler = preprocessing.StandardScaler()
# ridge = linear_model.Ridge()

# regressor = pipeline.make_pipeline(scaler, ridge)

# # === Grille de valeurs de alpha à tester ===
# param_grid = {
#     'ridge__alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]
# }

target = dataset['energies']

# === Validation croisée ===
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# === Recherche sur grille ===
grid_search = model_selection.GridSearchCV(
    regressor,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',  # ou 'neg_root_mean_squared_error'
    n_jobs=-1,
    verbose=1,
    refit="MAE"
)

grid_search.fit(scattering_coef, target)

# === Résultats ===
print("Meilleur alpha :", grid_search.best_params_['boosting__estimator__alpha'])
print("MAE moyen :", -grid_search.best_score_)

# Tu peux afficher les scores pour toutes les valeurs :
means = -grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
alphas = param_grid['boosting__estimator__alpha']

print("\nRésultats par alpha :")
for alpha, mean, std in zip(alphas, means, stds):
    print(f"alpha={alpha:<6} --> MAE = {mean:.4f} ± {std:.4f}")



# === GÉNÉRATION DU FICHIER DE PRÉDICTION POUR LE DOSSIER TEST ===

# Chemin du dossier test
test_dir = os.path.join(data_dir, "atoms", "test")

# Récupération des fichiers .xyz
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".xyz")])
test_ids = [int(f.split("_")[1].split(".")[0]) for f in test_files]

test_positions_list = []
test_charges_list = []

for mol_id in test_ids:
    filepath = os.path.join(test_dir, f"id_{mol_id}.xyz")
    with open(filepath, "r") as f:
        lines = f.readlines()[2:]

    charges = []
    positions = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            symbol = parts[0]
            xyz = list(map(float, parts[1:]))

            charges.append(symbol_to_number.get(symbol, 0))
            positions.append(xyz)

    test_charges_list.append(charges)
    test_positions_list.append(positions)

# Padding
num_test = len(test_charges_list)
positions_array_test = np.zeros((num_test, max_atoms, 3), dtype=np.float32)
charges_array_test = np.zeros((num_test, max_atoms), dtype=np.int32)

for i in range(num_test):
    n = len(test_positions_list[i])
    positions_array_test[i, :n, :] = test_positions_list[i]
    charges_array_test[i, :n] = test_charges_list[i]

# Même prétraitement
pos_test = positions_array_test
full_charges_test = charges_array_test



M = 32
N = 32
O = 32
J = 3
L = 3
scattering_coef_test = get_scat_coeff(pos_test, full_charges_test, M, N, O, J, L,str_type="test")


# Prédictions
predicted_energies = grid_search.predict(scattering_coef_test)

# Sauvegarde dans un CSV
output_file = "test_predictions_323232_3_3_1.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "energy"])
    for mol_id, energy in zip(test_ids, predicted_energies):
        writer.writerow([mol_id, energy])

print(f"✅ Fichier de prédictions sauvegardé sous : {output_file}")

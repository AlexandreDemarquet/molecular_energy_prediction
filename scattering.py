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

import os
import csv
import numpy as np

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

# ✅ Vérification
print("positions shape:", dataset["positions"].shape)
print("charges shape:", dataset["charges"].shape)
print("energies shape:", dataset["energies"].shape)

pos = dataset['positions']
full_charges = dataset['charges']

n_molecules = pos.shape[0]

mask = full_charges <= 2
valence_charges = full_charges * mask

mask = np.logical_and(full_charges > 2, full_charges <= 10)
valence_charges += (full_charges - 2) * mask

mask = np.logical_and(full_charges > 10, full_charges <= 18)
valence_charges += (full_charges - 10) * mask

overlapping_precision = 1e-1
sigma = 2.0
min_dist = np.inf

for i in range(n_molecules):
    n_atoms = np.sum(full_charges[i] != 0)
    pos_i = pos[i, :n_atoms, :]
    min_dist = min(min_dist, pdist(pos_i).min())

delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
pos = pos * delta / min_dist


#M, N, O = 192, 128, 96
M, N, O = 64, 64, 64

grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
grid = np.fft.ifftshift(grid)

J = 2
L = 3
integral_powers = [0.5, 1.0, 2.0, 3.0]

scattering = HarmonicScattering3D(J=J, shape=(M, N, O),
                                  L=L, sigma_0=sigma,
                                  integral_powers=integral_powers)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
scattering.to(device)

batch_size = 8
n_batches = int(np.ceil(n_molecules / batch_size))

order_0, orders_1_and_2 = [], []
print('Computing solid harmonic scattering coefficients of '
      '{} molecules from the QM7 database on {}'.format(
        n_molecules,   "GPU" if use_cuda else "CPU"))
print('sigma: {}, L: {}, J: {}, integral powers: {}'.format(
        sigma, L, J, integral_powers))

this_time = None
last_time = None
for i in range(n_batches):
    this_time = time.time()
    if last_time is not None:
        dt = this_time - last_time
        print("Iteration {} ETA: [{:02}:{:02}:{:02}]".format(
                    i + 1, int(((n_batches - i - 1) * dt) // 3600),
                    int((((n_batches - i - 1) * dt) // 60) % 60),
                    int(((n_batches - i - 1) * dt) % 60)))
    else:
        print("Iteration {} ETA: {}".format(i + 1, '-'))
    last_time = this_time
    time.sleep(1)

    # Extract the current batch.
    start = i * batch_size
    end = min(start + batch_size, n_molecules)

    pos_batch = pos[start:end]
    full_batch = full_charges[start:end]
    val_batch = valence_charges[start:end]

    # Calculate the density map for the nuclear charges and transfer
    # to PyTorch.
    full_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, full_batch, sigma)
    full_density_batch = torch.from_numpy(full_density_batch)
    full_density_batch = full_density_batch.to(device).float()

    # Compute zeroth-order, first-order, and second-order scattering
    # coefficients of the nuclear charges.
    full_order_0 = TorchBackend3D.compute_integrals(full_density_batch,
                                     integral_powers)
    full_scattering = scattering(full_density_batch)

    # Compute the map for valence charges.
    val_density_batch = generate_weighted_sum_of_gaussians(grid,
            pos_batch, val_batch, sigma)
    val_density_batch = torch.from_numpy(val_density_batch)
    val_density_batch = val_density_batch.to(device).float()

    # Compute scattering coefficients for the valence charges.
    val_order_0 = TorchBackend3D.compute_integrals(val_density_batch,
                                    integral_powers)
    val_scattering = scattering(val_density_batch)

    # Take the difference between nuclear and valence charges, then
    # compute the corresponding scattering coefficients.
    core_density_batch = full_density_batch - val_density_batch

    core_order_0 = TorchBackend3D.compute_integrals(core_density_batch,
                                     integral_powers)
    core_scattering = scattering(core_density_batch)

    # Stack the nuclear, valence, and core coefficients into arrays
    # and append them to the output.
    batch_order_0 = torch.stack(
        (full_order_0, val_order_0, core_order_0), dim=-1)
    batch_orders_1_and_2 = torch.stack(
        (full_scattering, val_scattering, core_scattering), dim=-1)

    order_0.append(batch_order_0)
    orders_1_and_2.append(batch_orders_1_and_2)

order_0 = torch.cat(order_0, dim=0)
orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

order_0 = order_0.cpu().numpy()
orders_1_and_2 = orders_1_and_2.cpu().numpy()

order_0 = order_0.reshape((n_molecules, -1))
orders_1_and_2 = orders_1_and_2.reshape((n_molecules, -1))

scattering_coef = np.concatenate([order_0, orders_1_and_2], axis=1)

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

# === Pipeline standardisation + Ridge ===
scaler = preprocessing.StandardScaler()
ridge = linear_model.Ridge()

regressor = pipeline.make_pipeline(scaler, ridge)

# === Grille de valeurs de alpha à tester ===
param_grid = {
    'ridge__alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001]
}

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
    verbose=1
)

grid_search.fit(scattering_coef, target)

# === Résultats ===
print("Meilleur alpha :", grid_search.best_params_['ridge__alpha'])
print("MAE moyen :", -grid_search.best_score_)

# Tu peux afficher les scores pour toutes les valeurs :
means = -grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
alphas = param_grid['ridge__alpha']

print("\nRésultats par alpha :")
for alpha, mean, std in zip(alphas, means, stds):
    print(f"alpha={alpha:<6} --> MAE = {mean:.4f} ± {std:.4f}")
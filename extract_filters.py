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
import matplotlib.pyplot as plt

import os
import csv
import numpy as np
import pickle

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


# M, N, O = 192, 128, 96
M, N, O = 64, 64, 64

grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
grid = np.fft.ifftshift(grid)

J = 2
L = 4
integral_powers = [0.5, 1.0, 2.0, 4.0]

scattering = HarmonicScattering3D(J=J, shape=(M, N, O),
                                  L=L, sigma_0=sigma,
                                  integral_powers=integral_powers)

pickle.dump(scattering.filters, open(f"scattering_{M}_{N}_{O}.pkl", "wb"))
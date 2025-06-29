
from itertools import combinations
from pathlib import Path
import torch


DATA_ROOT = Path("./data")
XYZ_DIR = DATA_ROOT / "atoms" / "train"
ENERGY_CSV = DATA_ROOT / "energies" / "train.csv"
ATOM_TYPES  = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']

def load_xyz(filepath):
    atom_types = []
    positions = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]  # sauter les 2 premières lignes
        for line in lines:
            parts = line.split()
            atom_types.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
    return atom_types, torch.tensor(positions, dtype=torch.float32)

# === Encodage simple des atomes ===
element_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Cl': 5, 'S' : 1}  # à compléter si besoin
def encode_atoms(atom_types):
    return torch.tensor([element_map[a] for a in atom_types], dtype=torch.long).unsqueeze(1)

# === Construction d'un graphe simple ===
# def build_graph(file_id):
#     xyz_path = XYZ_DIR / f"{file_id}.xyz"
#     atom_types, pos = load_xyz(xyz_path)
#     x = encode_atoms(atom_types)

#     edge_index = []
#     for i, j in combinations(range(len(pos)), 2):
#         dist = torch.norm(pos[i] - pos[j])
#         if dist < 1.6:  # seuil de liaison covalente
#             edge_index.append([i, j])
#             edge_index.append([j, i])
#     if edge_index:
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)

#     y = torch.tensor([id_to_energy[file_id]], dtype=torch.float32)
#     return Data(x=x, edge_index=edge_index, pos=pos, y=y, id=file_id)



# === 2. UTILS ===
def atom_to_int(symbol: str) -> int:
    return ATOM_TYPES.index(symbol)
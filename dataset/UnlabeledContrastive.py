import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, Batch

import numpy as np
from torch_geometric.data import Data
# from torch_cluster import radius_graph
# import torch_cluster

from utils import atom_to_int


class UnlabeledContrastive(torch.utils.data.Dataset):
    def __init__(self, xyz_dir, cutoff=5.0):
        self.xyz_files = sorted(Path(xyz_dir).glob("*.xyz"))
        self.cutoff = cutoff

    def __len__(self):
        return len(self.xyz_files)

    def _load_xyz(self, file_path):
        lines = open(file_path).read().splitlines()[2:]
        atoms, pos = [], []
        for line in lines:
            s, x, y, z = line.split()
            atoms.append(atom_to_int(s))
            pos.append([float(x), float(y), float(z)])
        return torch.tensor(atoms, dtype=torch.long), torch.tensor(pos, dtype=torch.float)

    def augment(self, pos):
        u = torch.rand(3)
        q = torch.tensor([
            torch.sqrt(1-u[0]) * torch.sin(2*torch.pi*u[1]),
            torch.sqrt(1-u[0]) * torch.cos(2*torch.pi*u[1]),
            torch.sqrt(u[0])   * torch.sin(2*torch.pi*u[2]),
            torch.sqrt(u[0])   * torch.cos(2*torch.pi*u[2])
        ])
        w, x, y, z = q
        R = torch.tensor([
            [1-2*(y**2+z**2), 2*(x*y - z*w),   2*(x*z + y*w)],
            [2*(x*y + z*w),   1-2*(x**2+z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x**2+y**2)]
        ])
        t = torch.randn(3)
        return pos @ R.T + t

    def __getitem__(self, idx):
        z, pos = self._load_xyz(self.xyz_files[idx])
        pos1 = self.augment(pos)
        pos2 = self.augment(pos)
        return Data(z=z, pos=pos1), Data(z=z, pos=pos2)
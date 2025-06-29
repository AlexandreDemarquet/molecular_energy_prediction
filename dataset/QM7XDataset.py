
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, Batch
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_add_pool
# from torch_cluster import radius_graph
# import torch_cluster
from utils import atom_to_int




class QM7XDataset(torch.utils.data.Dataset):
    def __init__(self, xyz_dir, energy_csv):
        self.xyz_dir   = Path(xyz_dir)
        self.energy_df = pd.read_csv(energy_csv, index_col='id')

    def __len__(self):
        return len(self.energy_df)

    def _load_xyz(self, file_path):
        lines = open(file_path).read().splitlines()[2:]
        atoms, pos = [], []
        for line in lines:
            s, x, y, z = line.split()
            atoms.append(atom_to_int(s))
            pos.append([float(x), float(y), float(z)])
        return torch.tensor(atoms, dtype=torch.long), torch.tensor(pos, dtype=torch.float)

    def augment(self, pos):
        # random rotation via quaternion
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
        mol_id   = idx + 1
        xyz_file = self.xyz_dir / f"id_{mol_id}.xyz"
        z, pos   = self._load_xyz(xyz_file)
        # two augmented views
        pos1 = self.augment(pos)
        pos2 = self.augment(pos)
        data1 = Data(z=z, pos=pos1)
        data2 = Data(z=z, pos=pos2)
        return data1, data2

# custom collate for two-view batches
def collate_fn(batch):
    views1, views2 = zip(*batch)
    return Batch.from_data_list(views1), Batch.from_data_list(views2)

class QM7XEmbedDataset(QM7XDataset):
    def __getitem__(self, idx):
        mol_id   = idx + 1
        xyz_file = self.xyz_dir / f"id_{mol_id}.xyz"
        z, pos   = self._load_xyz(xyz_file)
        data     = Data(z=z, pos=pos)
        data.batch = torch.zeros(z.size(0), dtype=torch.long)
        return data, float(self.energy_df.loc[mol_id, 'energy'])
    
from utils import ATOM_TYPES
class QM7XTestDataset(torch.utils.data.Dataset):
    def __init__(self, xyz_dir):
        self.xyz_dir = Path(xyz_dir)
        self.files = sorted(self.xyz_dir.glob("id_*.xyz"))

    def __len__(self):
        return len(self.files)

    def _load_xyz(self, file_path):
        lines = file_path.open().read().splitlines()[2:]
        atoms, pos = [], []
        for line in lines:
            s, x, y, z = line.split()
            atoms.append(ATOM_TYPES.index(s))
            pos.append([float(x), float(y), float(z)])
        return torch.tensor(atoms, dtype=torch.long), torch.tensor(pos, dtype=torch.float)

    def __getitem__(self, idx):
        z, pos = self._load_xyz(self.files[idx])
        data = Data(z=z, pos=pos)
        data.batch = torch.zeros(z.size(0), dtype=torch.long)
        return data

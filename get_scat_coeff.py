import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import torch
import time
import os
from scipy.spatial.distance import pdist
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians

import os
import csv
import numpy as np
import pickle



def get_scat_coeff(pos, full_charges,M,N,O,J,L, str_type="train"):

    n_molecules = pos.shape[0]

    mask = full_charges <= 2
    valence_charges = full_charges * mask

    mask = np.logical_and(full_charges > 2, full_charges <= 10)
    valence_charges += (full_charges - 2) * mask

    mask = np.logical_and(full_charges > 10, full_charges <= 18)
    valence_charges += (full_charges - 10) * mask

    overlapping_precision = 1e-1
    sigma = 1
    min_dist = np.inf

    for i in range(n_molecules):
        n_atoms = np.sum(full_charges[i] != 0)
        pos_i = pos[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos_i).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    pos = pos * delta / min_dist


    # M, N, O = 192, 128, 96
    # M, N, O = 64, 64, 64

    grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
    grid = np.fft.ifftshift(grid)

    # J = 2
    # L = 3
    integral_powers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

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

    with open(f"scattering_coef_{str_type}_{M}_{N}_{O}.pkl", "wb") as f:
        pickle.dump(scattering_coef, f)

    return scattering_coef
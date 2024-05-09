import torch
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_matrix, get_permutation, mpo_to_tensor
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np


tl.set_backend('pytorch')

torch.random.manual_seed(0)

# bond_dims = [1,10,20,30,40,50]
bond_dims = [1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256]
econ_bond_dims = bond_dims[:7]
pics_path = "../pics/"

device = torch.device("cuda")

def get_product_time(bond_dim, rounding=False, ITERS=3):
    rank = 16
    N = 4096
    W = torch.randn(N, N).to(device)
    x = torch.randn(N, N).to(device)
    y = torch.matmul(W, x)
    W_reshaped = W.reshape([rank, rank, rank, rank, rank, rank])
    x_reshaped = x.reshape([rank, rank, rank, rank, rank, rank])
    order = get_permutation(W_reshaped, decompose=True)
    W_reshaped = W_reshaped.permute(order)
    x_reshaped = x_reshaped.permute(order)
    W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
    x_mpo = mpo_decompostion(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
    start = time.time()
    for _ in range(ITERS):
        if rounding:
            y_mpo = matrix_by_matrix(W_mpo, x_mpo, round_dim=int(np.sqrt(bond_dim)))
        else:
            y_mpo = matrix_by_matrix(W_mpo, x_mpo)
    end = time.time()
    mpo_time = (end - start) / ITERS
    y_tensor = mpo_to_tensor(y_mpo).clone().detach()
    y_tensor = y_tensor.permute(get_permutation(y_tensor, decompose=False))
    y_tensor = y_tensor.reshape(-1)
    y = y.reshape(-1)
    # err = torch.max(torch.abs(y - y_tensor))
    err = torch.norm(y - y_tensor, p=2) / torch.norm(y, p=2)
    start = time.time()
    for _ in range(ITERS):
        torch.matmul(W, x)
    end = time.time()
    matmul_time = (end - start) / ITERS
    return mpo_time, err, matmul_time

def time_product():
    times = []
    errs = []
    matmul_times = []
    plain_times = []
    plain_errs = []
    plain_matmul_times = []
    for b in bond_dims:
        print(f"rounding, bond_dim = {b}")
        t, err, matmul_t = get_product_time(b, rounding=True)
        times.append(t)
        errs.append(err)
        matmul_times.append(matmul_t)
    for b in econ_bond_dims:
        print(f"no rounding, bond_dim={b}")
        t, err, matmul_t = get_product_time(b, rounding=False)
        plain_times.append(t)
        plain_errs.append(err)
        plain_matmul_times.append(matmul_t)
    return times, errs, matmul_times, plain_times, plain_errs, plain_matmul_times

def plot_product_time(times, matmul_times, plain_times):
    # x = bond_dims, y = times
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title(f"(4096,4096) @ (4096, 4096) Rounded Matrix by Matrix Product Time (CPU)")
    else:
        ax.set_title(f"(4096,4096) @ (4096, 4096) Rounded Matrix by Matrix Product Time (GPU)")
    ax.plot(bond_dims, times, label="rounded")
    ax.plot(bond_dims, matmul_times, label="torch.matmul")
    ax.plot(econ_bond_dims, plain_times, label="not rounded")
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Time")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}rounded_matrix_by_matrix_product_time_cpu.png")
    else:
        plt.savefig(f"{pics_path}rounded_matrix_by_matrix_product_time_gpu.png")

def plot_product_error(errs, plain_errs):
    # x = bond_dims, y = errs
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title(f"(4096,4096) @ (4096, 4096) Rounded Matrix by Matrix Product Error (CPU)")
    else:
        ax.set_title(f"(4096,4096) @ (4096, 4096) Rounded Matrix by Matrix Product Error (GPU)")
    ax.plot(bond_dims, errs, label="rounded")
    ax.plot(econ_bond_dims, plain_errs, label="not rounded")
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Error")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}rounded_matrix_by_matrix_product_error_cpu.png")
    else:
        plt.savefig(f"{pics_path}rounded_matrix_by_matrix_product_error_gpu.png")

if __name__ == "__main__":
    times, errs, matmul_times, plain_times, plain_errs, plain_matmul_times = time_product()
    errs = [err.item() for err in errs]
    plain_errs = [err.item() for err in plain_errs]
    plot_product_time(times, matmul_times, plain_times)
    plot_product_error(errs, plain_errs)


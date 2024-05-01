import torch
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_matrix, get_permutation, mpo_to_tensor
import matplotlib.pyplot as plt
import tensorly as tl


tl.set_backend('pytorch')

torch.random.manual_seed(0)

bond_dims = [60]
pics_path = "../pics/"

device = torch.device("cuda")

def get_product_time(bond_dim, ITERS=3):
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
        y_mpo = matrix_by_matrix(W_mpo, x_mpo)
    end = time.time()
    mpo_time = (end - start) / ITERS
    y_tensor = mpo_to_tensor(y_mpo).clone().detach()
    y_tensor = y_tensor.reshape(-1)
    y = y.reshape(-1)
    err = torch.max(torch.abs(y - y_tensor))
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
    for b in bond_dims:
        t, err, matmul_t = get_product_time(b)
        times.append(t)
        errs.append(err)
        matmul_times.append(matmul_t)
    return times, errs, matmul_times

def plot_product_time(times, matmul_times):
    # x = bond_dims, y = times
    fig, ax = plt.subplots()
    ax.plot(bond_dims, times, label="mpo")
    ax.plot(bond_dims, matmul_times, label="torch.matmul")
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Time")
    ax.legend()
    plt.savefig(f"{pics_path}matrix_by_matrix_product_time.png")

def plot_product_error(errs):
    # x = bond_dims, y = errs
    fig, ax = plt.subplots()
    ax.plot(bond_dims, errs)
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Error")
    plt.savefig(f"{pics_path}matrix_by_matrix_product_error.png")

times, errs, matmul_times = time_product()
errs = [err.item() for err in errs]
plot_product_time(times, matmul_times)
plot_product_error(errs)


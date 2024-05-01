import torch
from tensorly.decomposition import matrix_product_state
from tensorly.tt_tensor import tt_to_tensor
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_vector, get_permutation
import matplotlib.pyplot as plt
import tensorly as tl

tl.set_backend('pytorch')

pics_path = "../pics/"

torch.random.manual_seed(0)

device = torch.device("cuda")

def get_product_time(rank, bond_dim, ITERS=3):
    N = rank * rank * rank
    W = torch.randn(N, N).to(device)
    x = torch.randn(N).to(device)
    y = torch.matmul(W, x)
    W_reshaped = W.reshape([rank, rank, rank, rank, rank, rank])
    x_reshaped = x.reshape([rank, rank, rank])
    order = get_permutation(W_reshaped, decompose=True)
    W_reshaped = W_reshaped.permute(order)
    W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
    x_mpo = matrix_product_state(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
    start = time.time()
    for _ in range(ITERS):
        y_mpo = matrix_by_vector(W_mpo, x_mpo)
    end = time.time()
    mpo_time = (end - start) / ITERS
    y_tensor = tt_to_tensor(y_mpo).clone().detach()
    y_tensor = y_tensor.reshape(-1)
    y = y.reshape(-1)
    err = torch.max(torch.abs(y - y_tensor))
    start = time.time()
    for _ in range(ITERS):
        y = torch.matmul(W, x)
    end = time.time()
    matmul_time = (end - start) / ITERS
    return mpo_time, err, matmul_time

def time_product():
    times = []
    errs = []
    matmul_times = []
    ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    dims = [r**3 for r in ranks]
    for b in [1, 10, 20, 40, 80, 100]:
        times.append([])
        errs.append([])
        matmul_times.append([])
        for r in ranks:
            t, err, matmul_t = get_product_time(r, b)
            times[-1].append(t)
            errs[-1].append(err)
            matmul_times[-1].append(matmul_t)
    return dims, times, errs, matmul_times

def plot_product_time(dims, times, matmul_times):
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title("Matrix-by-vector Product Time (CPU)")
    else:
        ax.set_title("Matrix-by-vector Product Time (GPU)")
    for i, b in enumerate([1, 10, 20, 40, 80, 100]):
        ax.plot(dims, times[i], label=f"Bond dimension: {b}")
    ax.plot(dims, matmul_times[0], label="torch.matmul")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (s)")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}matrix_by_vector_product_time_cpu.png")
    else:
        plt.savefig(f"{pics_path}matrix_by_vector_product_time_gpu.png")

def plot_product_error(dims, errs):
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title("Matrix-by-vector Product Error (CPU)")
    else:
        ax.set_title("Matrix-by-vector Product Error (GPU)")
    for i, b in enumerate([1, 10, 20, 40, 80, 100]):
        ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], errs[i], label=f"Bond dimension: {b}")
    ax.set_xlabel("Cube Root of Dimension")
    ax.set_ylabel("Error")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}matrix_by_vector_product_error_cpu.png")
    else:
        plt.savefig(f"{pics_path}matrix_by_vector_product_error_gpu.png")


dims, times, errs, matmul_times = time_product()
errs = [[float(e) for e in err] for err in errs]
plot_product_time(dims, times, matmul_times)
plot_product_error(dims, errs)
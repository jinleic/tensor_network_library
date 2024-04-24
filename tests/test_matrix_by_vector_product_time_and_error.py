import torch
from tensorly.decomposition import matrix_product_state
from tensorly.tt_tensor import tt_to_tensor
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_vector, get_permutation
import matplotlib.pyplot as plt

torch.random.manual_seed(0)

def get_product_time(rank, bond_dim, ITERS=3):
    N = rank * rank * rank
    W = torch.randn(N, N)
    x = torch.randn(N)
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
    y_tensor = torch.tensor(tt_to_tensor(y_mpo))
    y_tensor = y_tensor.reshape(-1)
    y = y.reshape(-1)
    err = torch.max(torch.abs(y - y_tensor))
    return (end - start) / ITERS, err

def time_product():
    times = []
    errs = []
    ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    dims = [r**3 for r in ranks]
    for b in [1, 10, 20, 40, 80, 100]:
        times.append([])
        errs.append([])
        for r in ranks:
            t, err = get_product_time(r, b)
            times[-1].append(t)
            errs[-1].append(err)
    return dims, times, errs

def plot_product_time(dims, times):
    fig, ax = plt.subplots()
    for i, b in enumerate([1, 10, 20, 40, 80, 100]):
        ax.plot(dims, times[i], label=f"Bond dimension: {b}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.savefig("Product Time.png")

def plot_product_error(dims, errs):
    fig, ax = plt.subplots()
    for i, b in enumerate([1, 10, 20, 40, 80, 100]):
        ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], errs[i], label=f"Bond dimension: {b}")
    ax.set_xlabel("Cube Root of Dimension")
    ax.set_ylabel("Error")
    ax.legend()
    plt.savefig("Product Error.png")

dims, times, errs = time_product()
plot_product_time(dims, times)
plot_product_error(dims, errs)
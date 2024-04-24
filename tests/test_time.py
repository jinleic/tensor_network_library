import torch
from tensorly.decomposition import matrix_product_state
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_vector, matrix_by_matrix, get_permutation, validate_product, validate_stacked_product
import matplotlib.pyplot as plt

torch.random.manual_seed(0)

def get_decomposition_time(rank, bond_dim, ITERS=10):
    N = rank * rank * rank
    W = torch.randn(N, N)
    W_reshaped = W.reshape([rank, rank, rank, rank, rank, rank])
    order = get_permutation(W_reshaped, decompose=True)
    W_reshaped = W_reshaped.permute(order)
    start = time.time()
    for _ in range(ITERS):
        W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
    end = time.time()
    return (end - start) / ITERS

def time_decomposition():
    times = []
    for b in [1, 10, 50, 100]:
        times.append([])
        for r in range(2, 20):
            times[-1].append(get_decomposition_time(r, b))
    return times

def plot_decomposition_time(times):
    fig, ax = plt.subplots()
    for i, b in enumerate([1, 10, 50, 100]):
        ax.plot(range(2, 20), times[i], label=f"Bond dimension: {b}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.savefig("decomposition_time.png")

times = time_decomposition()
plot_decomposition_time(times)
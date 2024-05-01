import torch
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, get_permutation
import matplotlib.pyplot as plt
import tensorly as tl

tl.set_backend('pytorch')

pics_path = "../pics/"

torch.random.manual_seed(0)

device = torch.device("cpu")

def get_decomposition_time(rank, bond_dim, ITERS=3):
    N = rank * rank * rank
    W = torch.randn(N, N).to(device)
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
    ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    dims = [r**3 for r in ranks]
    for b in [1, 10, 50, 100]:
        times.append([])
        for r in ranks:
            times[-1].append(get_decomposition_time(r, b))
    return dims, times

def plot_decomposition_time(dims, times):
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title("Decomposition Time (CPU)")
    else:
        ax.set_title("Decomposition Time (GPU)")
    for i, b in enumerate([1, 10, 50, 100]):
        ax.plot(dims, times[i], label=f"Bond dimension: {b}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (s)")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}decomposition_time_cpu.png")
    else:
        plt.savefig(f"{pics_path}decomposition_time_gpu.png")

dims, times = time_decomposition()
plot_decomposition_time(dims, times)
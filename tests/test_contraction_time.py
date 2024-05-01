import torch
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_matrix, get_permutation, mpo_to_tensor
import matplotlib.pyplot as plt
import tensorly as tl


tl.set_backend('pytorch')

torch.random.manual_seed(0)

bond_dims = [1,10,20,30,40,50]
pics_path = "../pics/"

device = torch.device("cuda")

def test_contraction(A, B):
    d = len(A)
    factors = [None] * d
    shape_a = A[0].shape
    shape_b = B[0].shape
    max_time = 0
    for k in range(d):
        r_prev, i_k, j_k, r_next = A[k].shape
        d_prev, j_k, l_k, d_next = B[k].shape
        start = time.time()
        factors[k] = tl.tensordot(A[k], B[k], ([2], [1]))
        end = time.time()
        if end - start > max_time:
            max_time = end - start
            shape_a = A[k].shape
            shape_b = B[k].shape
        factors[k] = factors[k].permute(0, 3, 1, 4, 2, 5) # (r_{k-1}, d_{k-1}, i_k, l_k, r_k, d_k)
        factors[k] = factors[k].reshape(r_prev * d_prev, i_k, l_k, d_next * r_next)
    return max_time, shape_a, shape_b

def get_product_time(bond_dim, ITERS=3):
    rank = 16
    N = 4096
    W = torch.randn(N, N).to(device)
    x = torch.randn(N, N).to(device)
    W_reshaped = W.reshape([rank, rank, rank, rank, rank, rank])
    x_reshaped = x.reshape([rank, rank, rank, rank, rank, rank])
    order = get_permutation(W_reshaped, decompose=True)
    W_reshaped = W_reshaped.permute(order)
    x_reshaped = x_reshaped.permute(order)
    W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
    x_mpo = mpo_decompostion(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
    t, shape_a, shape_b = test_contraction(W_mpo, x_mpo)
    start = time.time()
    torch.matmul(W, x)
    end = time.time()
    return t, shape_a, shape_b, end - start

def time_product():
    times = []
    list_shape_a = []
    list_shape_b = []
    matmul_times = []
    for b in bond_dims:
        print(f"bond_dim = {b}")
        t, shape_a, shape_b, t_matmul = get_product_time(b)
        times.append(t)
        list_shape_a.append(shape_a)
        list_shape_b.append(shape_b)
        matmul_times.append(t_matmul)
    return times, list_shape_a, list_shape_b, matmul_times

def plot_product_time(times, matmul_times):
    # x = bond_dims, y = times
    fig, ax = plt.subplots()
    if device == torch.device("cpu"):
        ax.set_title(f"(4096,4096) @ (4096, 4096) Worst Contraction Time (CPU)")
    else:
        ax.set_title(f"(4096,4096) @ (4096, 4096) Worst Contraction Time (GPU)")
    ax.plot(bond_dims, times, label="worst contraction using torch.tensordot")
    ax.plot(bond_dims, matmul_times, label="(4096, 4096) @ (4096, 4096) using torch.matmul")
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Time")
    ax.legend()
    if device == torch.device("cpu"):
        plt.savefig(f"{pics_path}worst_contraction_time_cpu.png")
    else:
        plt.savefig(f"{pics_path}worst_contraction_time_gpu.png")

def gen_table(list_shape_a, list_shape_b):
    if device == torch.device("cpu"):
        with open(f"{pics_path}worst_contraction_shape_cpu.md", "w") as f:
            f.write("| Bond Dimension | Shape A | Shape B | Computation Scale | torch.matmul Scale |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for i, b in enumerate(bond_dims):
                shape_a = list_shape_a[i]
                shape_b = list_shape_b[i]
                scale = shape_a[0] * shape_a[1] * shape_a[2] * shape_a[3] * shape_b[0] * shape_b[2] * shape_b[3]
                matmul_scale = 4096 ** 3
                f.write(f"| {b} | {list_shape_a[i]} | {list_shape_b[i]} | {format(scale, '.2e')} | {format(matmul_scale, '.2e')} |\n")
    else:
        with open(f"{pics_path}worst_contraction_shape_gpu.md", "w") as f:
            f.write("| Bond Dimension | Shape A | Shape B | Computation Scale | torch.matmul Scale |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for i, b in enumerate(bond_dims):
                shape_a = list_shape_a[i]
                shape_b = list_shape_b[i]
                scale = shape_a[0] * shape_a[1] * shape_a[2] * shape_a[3] * shape_b[0] * shape_b[2] * shape_b[3]
                matmul_scale = 4096 ** 3
                f.write(f"| {b} | {list_shape_a[i]} | {list_shape_b[i]} | {format(scale, '.2e')} | {format(matmul_scale, '.2e')} |\n")

times, list_shape_a, list_shape_b, matmul_times = time_product()
plot_product_time(times, matmul_times)
gen_table(list_shape_a, list_shape_b)


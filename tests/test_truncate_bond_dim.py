import torch
import tensorly as tl
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, get_permutation, matrix_by_matrix, validate_product

tl.set_backend('pytorch')

torch.random.manual_seed(42)

W = torch.randn(4096, 4096)
x = torch.randn(4096, 4096)
y = torch.matmul(W, x)

W_reshaped = W.reshape([16]*6)
x_reshaped = x.reshape([16]*6)
W_reshaped = W_reshaped.permute(get_permutation(W_reshaped, decompose=True))
x_reshaped = x_reshaped.permute(get_permutation(x_reshaped, decompose=True))

bond_dim = 100
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
x_mpo = mpo_decompostion(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
print("done mpo_decomposition")
y_mpo = matrix_by_matrix(W_mpo, x_mpo, round_dim=10)
print(f"y_mpo rank: {y_mpo.rank}")

validate_product(y_mpo, y, 1e-3, 1e-3)
import torch
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, get_permutation_order, validate_decomposition

print("****** Test MPO Decomposition ******")
torch.random.manual_seed(0)
W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3
x = torch.randn(4096) # 4096 = 16^3
y = torch.matmul(W, x) # (216, 4096) @ (4096,) = (216,)

W_reshaped = W.reshape([6,6,6,16,16,16])
x_reshaped = x.reshape([16,16,16])
order = get_permutation_order(W_reshaped)
W_reshaped = W_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)

bond_dim = 100
print(f"****** Testing MPO Decomposition error ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])

validate_decomposition(W_mpo, W, 1e-3, 1e-3)

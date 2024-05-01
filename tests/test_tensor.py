import torch
import tensorly as tl
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, get_permutation, validate_decomposition

tl.set_backend('pytorch')

print("-"*50)
print("< Test MPO Decomposition >")
torch.random.manual_seed(0)
W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3

W_reshaped = W.reshape([6,6,6,16,16,16])
order = get_permutation(W_reshaped, decompose=True)
W_reshaped = W_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)

bond_dim = 100
print(f"****** Testing MPO Decomposition error ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])

validate_decomposition(W_mpo, W, 1e-3, 1e-3)
print("-"*50)

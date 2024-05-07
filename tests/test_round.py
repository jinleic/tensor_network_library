import torch
import tensorly as tl
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, get_permutation, mpo_round, validate_decomposition

tl.set_backend('pytorch')

torch.random.manual_seed(0)

W = torch.randn(4096, 4096)

W_reshaped = W.reshape([16,16,16,16,16,16])
W_reshaped = W_reshaped.permute(get_permutation(W_reshaped, decompose=True))

bond_dim = 256
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
print(W_mpo.compression_rate)
try:
    validate_decomposition(W_mpo, W, 1e-3, 1e-3)
    print("Decomposition validation passed")
except AssertionError:
    print("Decomposition validation failed")
round_dim = 255
W_rounded = mpo_round(W_mpo, rank=[1, round_dim, round_dim, 1])
try:
    validate_decomposition(W_rounded, W, 1e-3, 1e-3)
    print("Round validation passed")
except AssertionError:
    print("Round validation failed")

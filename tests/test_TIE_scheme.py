import sys
sys.path.append('../')
from tensors.TIE_scheme import *
from tensors.operations import *
from tensors.tensor import MPO
import torch
import tensorly as tl
from tensorly.decomposition import matrix_product_state

tl.set_backend('pytorch')

# TIE scheme: begin with transformed factors
print("TIE scheme: begin with transformed factors")
G = [None] * 3
G[2] = torch.tensor([
    [1,2,3],
    [0,0,0],
    [-1,-2,-3],
    [4,5,6],
    [0,0,0],
    [-4,-5,-6]
],dtype=torch.float32)
G[2] = G[2].reshape([3, 2, 3, 1])
print(f"G[2] shape: {G[2].shape} \n{G[2]}")
G[1] = torch.tensor([
    [1,2,3,4,5,6,7,8,9],
    [0,1,-4,7,-1,0,-2,-1,-3],
    [-1,-2,0,-2,0,0,0,2,0]
],dtype=torch.float32)
G[1] = G[1].reshape([3, 1, 3, 3])
print(f"G[1] shape: {G[1].shape} \n{G[1]}")
G[0] = torch.tensor([
    [2,1,-1,3,4,-3],
    [-2,-1,-1,0,0,-1]
],dtype=torch.float32)
G[0] = G[0].reshape([1, 2, 2, 3])
print(f"G[0] shape: {G[0].shape} \n{G[0]}")

X = torch.tensor([
    [
        [1,7,13],
        [3,9,15],
        [5,11,17]
    ],
    [
        [2,8,14],
        [4,10,16],
        [6,12,18]
    ]
],dtype=torch.float32)

W_mpo = MPO(G)
y_scheme = matrix_by_vector_scheme(W_mpo, X, transform_factor=False).reshape(-1)
print(f"y_scheme: {y_scheme}")

print("-"*50)

# naive scheme: turn transformed factors back to original tensor
print("naive scheme: turn transformed factors back to original tensor")
G = [None] * 3
G[2] = torch.tensor([
    [1,2,3],
    [0,0,0],
    [-1,-2,-3],
    [4,5,6],
    [0,0,0],
    [-4,-5,-6]
],dtype=torch.float32)
# G[2] = G[2].reshape([3, 2, 3, 1])

# G[2] = G[2].reshape([2, 3, 3]) # (0,1,2,3) -> (2,3,3,1)
# G[2] = G[2].permute(0,2,1) # (0,2,3,1) -> (2,3,1,3)
# G[2] = G[2].reshape([2, 3, 3, 1]) # (0,2,1,3) -> (2,3,3,1)

G[2] = G[2].reshape([2,3,3,1]) # (i_k, r_prev, j_k, r_next)
G[2] = G[2].permute(0,2,1,3) # (i_k, j_k, r_prev, r_next)
print(f"G[2] shape: {G[2].shape}")
print(G[2])
G[2] = G[2].permute(2,0,1,3) # (r_prev, i_k, j_k, r_next)
print(f"G[2] shape: {G[2].shape}")
print(G[2])

G[1] = torch.tensor([
    [1,2,3,4,5,6,7,8,9],
    [0,1,-4,7,-1,0,-2,-1,-3],
    [-1,-2,0,-2,0,0,0,2,0]
],dtype=torch.float32)
# G[1] = G[1].reshape([3, 1, 3, 3])
G[1] = G[1].reshape([1,3,3,3])
G[1] = G[1].permute(0,2,1,3)
print(f"G[1] shape: {G[1].shape}")
print(G[1])
G[1] = G[1].permute(2,0,1,3)
G[0] = torch.tensor([
    [2,1,-1,3,4,-3],
    [-2,-1,-1,0,0,-1]
],dtype=torch.float32)
# G[0] = G[0].reshape([1, 2, 2, 3])
G[0] = G[0].reshape([2, 2, 1, 3])
print(f"G[0] shape: {G[0].shape}")
print(G[0])
G[0] = G[0].permute(2,0,1,3)

W_mpo = MPO(G)

W_full = mpo_to_tensor(W_mpo)
y_full = tl.tensordot(W_full, X, ([1,3,5], [0,1,2])).reshape(-1)
print(f"y_full: {y_full}")

print("-"*50)

# TIE scheme: begin with original factors
print("TIE scheme: begin with original factors")

# transform manually

# G[2] = G[2].permute(1, 0, 2, 3)
# G[2] = G[2].reshape([3, 2, 3, 1])
# print(G[2])

# G[1] = G[1].permute(1, 0, 2, 3)
# G[1] = G[1].reshape([3, 1, 3, 3])
# print(G[1])

# G[0] = G[0].permute(1, 0, 2, 3)
# G[0] = G[0].reshape([1, 2, 2, 3])
# print(G[0])

# G[2] = tl.unfold(G[2], 1).reshape([3, 2, 3, 1])
# print(G[2])

W_mpo = MPO(G)
y_scheme = matrix_by_vector_scheme(W_mpo, X, transform_factor=True).reshape(-1)
print(f"y_scheme: {y_scheme}")

print("-"*50)

# TIE scheme: begin with full tensor
print("TIE scheme: begin with full tensor")
W = W_full.permute(get_permutation(W_full, decompose=False))
W_reshaped = W.reshape([2,1,2,2,3,3])
W_reshaped = W_reshaped.permute(get_permutation(W_reshaped, decompose=True))
bond_dim = 100
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
y_mpo = matrix_by_vector_scheme(W_mpo, X, transform_factor=True).reshape(-1)
print(f"y_mpo: {y_mpo}")

print("-"*50)

# # TIE scheme: test with random tensor

torch.random.manual_seed(42)
W = torch.randn(216,8)
x = torch.randn(8)
y = torch.matmul(W, x)

W_reshaped = W.reshape([6,6,6,2,2,2])
x_reshaped = x.reshape([2,2,2])
W_reshaped = W_reshaped.permute(get_permutation(W_reshaped, decompose=True))
bond_dim = 100
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
# validate_decomposition(W_mpo, W, 1e-3, 1e-3)
# x_mpo = matrix_product_state(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
# y_mpo = matrix_by_vector(W_mpo, x_mpo)
# validate_product(y_mpo, y, 1e-3, 1e-3)

y_scheme = matrix_by_vector_scheme(W_mpo, x_reshaped, transform_factor=True).reshape(-1)
print(f"error: {torch.max(torch.abs(y_scheme - y))}")
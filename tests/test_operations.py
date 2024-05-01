import torch
from tensorly.decomposition import matrix_product_state
import time
import sys
sys.path.append('../')
import tensorly as tl
from tensors.operations import mpo_decompostion, matrix_by_vector, matrix_by_matrix, get_permutation, validate_product, validate_stacked_product

tl.set_backend('pytorch')
# -----------------------------------------------------------
# Test matrix-by-vector product

print("-"*50)
print("< Test matrix-by-vector product >")
torch.random.manual_seed(0)
W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3
x = torch.randn(4096) # 4096 = 16^3
y = torch.matmul(W, x) # (216, 4096) @ (4096,) = (216,)
# reshape W and x
W_reshaped = W.reshape([6,6,6,16,16,16])
x_reshaped = x.reshape([16,16,16])
# permute W
order = get_permutation(W_reshaped, decompose=True)
W_reshaped = W_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)

bond_dim = 100
print(f"****** Testing matrix-by-vector product error ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
x_mpo = matrix_product_state(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
y_mpo = matrix_by_vector(W_mpo, x_mpo)
validate_product(y_mpo, y, 1e-3, 1e-3)

print("-"*50)

bond_dim = 1
print(f"****** Testing matrix-by-vector product speed ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
x_mpo = matrix_product_state(x_reshaped, rank=[1, bond_dim, bond_dim, 1])

ITERS = 1000
start = time.time()
for _ in range(ITERS):
    matrix_by_vector(W_mpo, x_mpo)
end = time.time()
print(f"Time for MPO-form matrix-by-vector product: {(end - start) / ITERS} s")

start = time.time()
for _ in range(ITERS):
    tl.dot(W, x)
end = time.time()
print(f"Time for naive matrix-by-vector product: {(end - start) / ITERS} s")
print("-"*50)

# -----------------------------------------------------------
# Test matrix-by-matrix product

print("< Test matrix-by-matrix product >")
torch.random.manual_seed(0)
W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3
x = torch.randn(4096,216) # 4096 = 16^3, 8 = 2^3
y = torch.matmul(W, x) # (216, 4096) @ (4096, 8) = (216, 8)
# reshape W and x
W_reshaped = W.reshape([6,6,6,16,16,16])
x_reshaped = x.reshape([16,16,16,6,6,6])
# permute W
order = get_permutation(W_reshaped, decompose=True)
W_reshaped = W_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)
x_reshaped = x_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)

bond_dim = 100
print(f"****** Testing matrix-by-matrix product error ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
x_mpo = mpo_decompostion(x_reshaped, rank=[1, bond_dim, bond_dim, 1])
start = time.time()
y_mpo = matrix_by_matrix(W_mpo, x_mpo)
end = time.time()
print(f"Time for MPO-form matrix-by-matrix product: {end - start} s")
validate_product(y_mpo, y, 1e-3, 1e-3)

print("-"*50)

bond_dim = 1
print(f"****** Testing matrix-by-matrix product speed ******\nBond dimension: {bond_dim}")
W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
x_mpo = mpo_decompostion(x_reshaped, rank=[1, bond_dim, bond_dim, 1])

ITERS = 1
start = time.time()
for _ in range(ITERS):
    matrix_by_matrix(W_mpo, x_mpo)
end = time.time()
print(f"Time for MPO-form matrix-by-matrix product: {(end - start) / ITERS} s")

start = time.time()
for _ in range(ITERS):
    tl.dot(W, x)
end = time.time()
print(f"Time for naive matrix-by-matrix product: {(end - start) / ITERS} s")
print("-"*50)

# -----------------------------------------------------------
# Test stacked matrix-by-vector product

# print("< Test multiple matrix-by-vector product >")
# torch.random.manual_seed(0)
# W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3
# x = torch.randn(4096,216) # 4096 = 16^3, 216=6^3
# y = torch.matmul(W, x) # (216, 4096) @ (4096, 216) = (216, 216)

# W_reshaped = W.reshape([6,6,6,16,16,16])
# order = get_permutation(W_reshaped, decompose=True)
# W_reshaped = W_reshaped.permute(order) # permute(0, 3, 1, 4, 2, 5)
# bond_dim = 100
# W_mpo = mpo_decompostion(W_reshaped, rank=[1, bond_dim, bond_dim, 1])
# x_transpose = x.permute(1,0)
# y_mpo = []
# start = time.time()
# for i in range(x_transpose.shape[0]):
#     x_vec = x_transpose[i].reshape([16,16,16])
#     x_vec_mpo = matrix_product_state(x_vec, rank=[1, bond_dim, bond_dim, 1])
#     y_vec_mpo = matrix_by_vector(W_mpo, x_vec_mpo)
#     y_mpo.append(y_vec_mpo)
# end = time.time()
# print(f"Bond dimension: {bond_dim}")
# print(f"Time for multiple matrix-by-vector product: {end - start} s")
# validate_stacked_product(y_mpo, y, 1e-3, 1e-3)

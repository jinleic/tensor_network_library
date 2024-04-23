import torch
from tensorly.decomposition import matrix_product_state
import time
import sys
sys.path.append('../')
from tensors.operations import mpo_decompostion, matrix_by_vector, get_permutation_order, validate_product

print("****** Test matrix-by-vector product ******")
torch.random.manual_seed(0)
W = torch.randn(216, 4096) # 216 = 6^3, 4096 = 16^3
x = torch.randn(4096) # 4096 = 16^3
y = torch.matmul(W, x) # (216, 4096) @ (4096,) = (216,)
# reshape W and x
W_reshaped = W.reshape([6,6,6,16,16,16])
x_reshaped = x.reshape([16,16,16])
# permute W
order = get_permutation_order(W_reshaped)
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
    torch.matmul(W, x)
end = time.time()
print(f"Time for naive matrix-by-vector product: {(end - start) / ITERS} s")

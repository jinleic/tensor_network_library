import torch
import time
import torch.nn.functional as F

ITERS = 1
W = torch.randn(216, 4096)
x = torch.randn(4096,216)
start = time.time()
for _ in range(ITERS):
    y = torch.matmul(W, x)
end = time.time()
print(f"(216, 4096) @ (4096, 216): {(end - start)/ITERS} s")

bond_dim = 100
A = torch.randn(bond_dim,6,16,bond_dim)
B = torch.randn(bond_dim,16,6,bond_dim)
start = time.time()
for _ in range(ITERS):
    C = torch.tensordot(A, B, dims=([2], [1]))
end = time.time()
print(f"tensordot: {(end - start)/ITERS} s")

A = A.reshape(-1, 16)
B = B.reshape(16, -1)
start = time.time()
for _ in range(ITERS):
    C = torch.matmul(A, B)
end = time.time()
print(f"matmul-2D: {(end - start)/ITERS} s")

A = A.reshape(-1)
B = B.reshape(-1)
start = time.time()
for _ in range(ITERS):
    C = torch.matmul(A, B)
end = time.time()
print(f"matmul-1D: {(end - start)/ITERS} s")

A = A.reshape(-1, 16)
B = B.reshape(-1, 16)
start = time.time()
for _ in range(ITERS):
    C = F.linear(A, B)
end = time.time()
print(f"linear: {(end - start)/ITERS} s")
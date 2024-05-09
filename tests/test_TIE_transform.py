import torch
import sys
sys.path.append('../')
from tensors.TIE_scheme import transform, matrix_by_vector_scheme
from tensors.tensor import MPO

# m = np.array([None, 2, 1, 2])
# n = np.array([1, 2, 3, 3])
# r = [1, 3, 3, 1]
# V = [None] * 4
# V[3] = torch.tensor([
#     [54,60,66,72,78,84],
#     [0,0,0,0,0,0],
#     [-54,-60,-66,-72,-78,-84],
#     [117,132,147,162,177,192],
#     [0,0,0,0,0,0],
#     [-117,-132,-147,-162,-177,-192],
# ])

# res = transform(V=V[3], h=3, m=m, n=n, r=r)
# print(f"V[3] transformed:\n {res}")

# V[2] = torch.tensor([
#     [-396, -882, -432, -972],
#     [756, 1674, 828, 1854],
#     [-186, -411, -204, -456]
# ])

# res = transform(V=V[2], h=2, m=m, n=n, r=r)
# print(f"V[2] transformed:\n {res}")


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
])

G = [None] * 3

G[2] = torch.tensor([
    [1,2,3],
    [0,0,0],
    [-1,-2,-3],
    [4,5,6],
    [0,0,0],
    [-4,-5,-6]
])
G[2] = G[2].reshape([3, 2, 3, 1])
G[1] = torch.tensor([
    [1,2,3,4,5,6,7,8,9],
    [0,1,-4,7,-1,0,-2,-1,-3],
    [-1,-2,0,-2,0,0,0,2,0]
])
G[1] = G[1].reshape([3, 1, 3, 3])
G[0] = torch.tensor([
    [2,1,-1,3,4,-3],
    [-2,-1,-1,0,0,-1]
])
G[0] = G[0].reshape([1, 2, 2, 3])

y = matrix_by_vector_scheme(MPO(G), X, transform_factor=False)
print(f"y:\n {y}")

"""
y:
 [2778. 6189.  426.  957.]
"""
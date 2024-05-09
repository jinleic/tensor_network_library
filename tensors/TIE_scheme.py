import tensorly as tl
import numpy as np

# 2024/5/9: still not sure what is wrong, don't use this file for now

def transform(V, h, m, n, r):
    assert h > 1, "transform is only defined for h > 1"
    V = V.transpose(1, 0)
    V = V.reshape(n[h - 1], -1)
    # split
    length = np.prod(n[1:h - 1]) * np.prod(m[h:])
    T = tl.zeros((length, n[h - 1], r[h - 1]))
    for j in range(length):
        T[j] = V[:, j * r[h - 1] : (j + 1) * r[h - 1]]
    # assemble
    V = tl.zeros((n[h - 1] * r[h - 1], length))
    for j in range(length):
        V[:, j] = T[j].reshape(-1)
    return V
    

def matrix_by_vector_scheme(W, x, transform_factor=True):
    """Perform matrix-by-matrix product as described in the TIE paper
    
    Parameters
    ----------
    W : MPO
        MPO Decomposition of the matrix
    x : tensor
        high-order tensor of the vector
    Returns
    -------
    y : tensor
        tensor of the result vector
    """
    d = len(W)
    m = np.array([None] * (d + 1))
    n = np.array([None] * (d + 1))
    r = list(W.rank)
    for i, factor in enumerate(W):
        m[i + 1] = factor.shape[1]
        n[i + 1] = factor.shape[2]

    if transform_factor:
        for i in range(d):
            original_shape = W[i].shape
            # W[i] = W[i].permute(1, 0, 2, 3)
            # W[i] = W[i].reshape(original_shape)
            # W[i] = W[i].permute(1, 2, 0, 3) # (i_k, j_k, r_prev, r_next)

            W[i] = tl.unfold(W[i], 1).reshape(original_shape)

    V = [None] * (d + 2)
    V_prime = [None] * (d + 2)
    x = x.permute(*range(d - 1, -1, -1))
    V_prime[d + 1] = x.reshape(n[d], -1) #（ n[d], n[d-1], ..., n[1] )
    for h in range(d, 0, -1):
        r_prev, i_k, j_k, r_next = W[h - 1].shape
        G = W[h - 1].reshape(-1, j_k * r_next) # (r_prev * i_k, j_k * r_next)

        V[h] = tl.dot(G, V_prime[h + 1])
        if h > 1:
            V_prime[h] = transform(V[h], h, m, n, r)

    return V[1]

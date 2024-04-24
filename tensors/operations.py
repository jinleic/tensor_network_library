import torch
import tensorly as tl
from tensorly.decomposition._tt import svd_interface
from tensorly.tt_tensor import TTTensor, tt_to_tensor
from tensors.tensor import MPO

def get_permutation(tensor, decompose=True):
    """Get the permutation for MPO Decomposition or decompression

    Parameters
    ----------
    tensor : tensor
        tensor to decompose or decompress
    decompose : bool
        if True, get the permutation for MPO Decomposition, else get the permutation for decompression
    Returns
    -------
    order : list
        permutation order
    """
    assert len(tensor.shape) % 2 == 0, "The order of the tensor should be even"
    d = len(tensor) // 2
    if decompose:
        return tuple(torch.argsort(torch.cat((torch.arange(0, 2 * d, 2), torch.arange(1, 2 * d, 2)))))
    else:
        return tuple(torch.cat((torch.arange(0, 2 * d, 2), torch.arange(1, 2 * d, 2))))
    

def get_decomposition_permutation(tensor):
    """Get the permutation for MPO Decomposition

    Parameters
    ----------
    tensor : tensor
        tensor to decompose
    Returns
    -------
    order : list
        permutation order
    """
    print("DeprecationWarning: get_decomposition_permutation is deprecated. Use get_permutation instead.")
    return get_permutation(tensor, decompose=True)
    assert len(tensor.shape) % 2 == 0, "The order of the tensor should be even"
    d = len(tensor) // 2
    return tuple(torch.argsort(torch.cat((torch.arange(0, 2 * d, 2), torch.arange(1, 2 * d, 2)))))


def get_decompression_permutation(tensor):
    """Get the permutation for decompression of MPO Decomposition
    
    Parameters
    ----------
    tensor : tensor
        tensor to decompress
    Returns
    -------
    order : list
        permutation order
    """
    print("DeprecationWarning: get_decompression_permutation is deprecated. Use get_permutation instead.")
    return get_permutation(tensor, decompose=False)
    assert len(tensor.shape) % 2 == 0, "The order of the tensor should be even"
    d = len(tensor.shape) // 2
    return tuple(torch.cat((torch.arange(0, 2 * d, 2), torch.arange(1, 2 * d, 2))))


def _validate_tt_rank(
    tensor_shape,
    rank
):
    """Returns the rank of an MPO Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : int list
    Returns
    -------
    rank : int list
        rank of the decomposition
    """

    # Check user input for potential errors
    n_dim = len(tensor_shape)
    if n_dim % 2 != 0:
        message = f"Provided tensor of order {n_dim} but TT decomposition requires a tensor of even order."
        raise (ValueError(message))
    # if n_dim + 1 != len(rank):
    if n_dim/2 + 1 != len(rank):
        message = f"Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {len(rank)} while tl.ndim(tensor) + 1  = {n_dim+1}"
        raise (ValueError(message))

    # Initialization
    if rank[0] != 1:
        message = "Provided rank[0] == {} but boundary conditions dictate rank[0] == rank[-1] == 1.".format(
            rank[0]
        )
        raise ValueError(message)
    if rank[-1] != 1:
        message = "Provided rank[-1] == {} but boundary conditions dictate rank[0] == rank[-1] == 1.".format(
            rank[-1]
        )
        raise ValueError(message)

    return list(rank)


def mpo_decompostion(input_tensor, rank):
    """Perform MPO Decomposition
    
    Parameters
    ----------
    input_tensor : tensor
        tensor to decompose
    rank : int list
        ranks of the MPO Decomposition
    Returns
    -------
    MPO : MPO
        MPO Decomposition of input_tensor
    """
    rank = _validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    # factors = [None] * n_dim
    factors = [None] * (n_dim//2)

    # Getting the TT factors up to n_dim/2 - 1
    for k in range(n_dim//2 - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[2 * k] * tensor_size[2 * k + 1])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method="truncated_svd")

        rank[k + 1] = current_rank

        # Get kth MPO factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[2 * k], tensor_size[2 * k + 1], rank[k + 1]))

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    factors[-1] = tl.reshape(unfolding, (rank[-2], tensor_size[-2], tensor_size[-1], 1))

    return MPO(factors)


def mpo_to_tensor(factors):
    """Convert MPO to tensor

    Parameters
    ----------
    factors : list
        factors of the MPO
    Returns
    -------
    tensor : tensor
        decompressed tensor of the MPO
    """
    if isinstance(factors, (float, int)):  # 0-order tensor
        return factors
    full_shape = []
    for f in factors:
        full_shape.append(f.shape[1])
        full_shape.append(f.shape[2])
    full_tensor = tl.reshape(factors[0], (factors[0].shape[1] * factors[0].shape[2], -1))
    for factor in factors[1:]:
        rank_prev, _, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)


def matrix_by_vector(matrix, vector):
    """Perform matrix-by-vector product

    Parameters
    ----------
    matrix : MPO
        MPO Decomposition of the matrix
    vector : MPO
        MPO Decomposition of the vector
    Returns
    -------
    MPO : MPO
        MPO Decomposition of the product
    """
    assert len(matrix) == len(vector), "The number of factors in the matrix and vector should be the same"
    d = len(matrix)
    factors = [None] * d
    for k in range(d):
        m_k = matrix[k] # (r_{k-1}, i_k, j_k, r_k)
        x_k = vector[k] # (d_{k-1}, j_k, d_k)
        assert len(m_k.shape) == 4, "The matrix factor should be order-4 tensor"
        assert len(x_k.shape) == 3, "The vector factor should be order-3 tensor"
        r_prev, i_k, j_k, r_next = m_k.shape
        d_prev, j_x, d_next = x_k.shape
        assert j_k == j_x, "The inner dimensions of the matrix and vector should be the same"
        m_k = m_k.transpose(0, 1, 3, 2) # (r_{k-1}, i_k, r_k, j_k)
        m_k = m_k.reshape(-1, j_k) # (r_{k-1} * i_k *r_k, j_k)
        x_k = x_k.transpose(1, 0, 2) # (j_k, d_{k-1}, d_k)
        x_k = x_k.reshape(j_k, -1) # (j_k, d_{k-1} * d_k)
        factors[k] = tl.dot(m_k, x_k) # (r_{k-1} * i_k * r_k, d_{k-1} * d_k)
        factors[k] = factors[k].reshape(r_prev, i_k, r_next, d_prev, d_next)
        factors[k] = factors[k].transpose(0, 3, 1, 2, 4)
        factors[k] = factors[k].reshape(r_prev * d_prev, i_k, d_next * r_next)
    return TTTensor(factors)


def matrix_by_matrix(A, B):
    """Perform matrix-by-matrix product
    
    Parameters
    ----------
    A : MPO
        MPO Decomposition of the first matrix
    B : MPO
        MPO Decomposition of the second matrix
    Returns
    -------
    MPO : MPO
        MPO Decomposition of the product
    """
    assert len(A) == len(B), "The number of factors in the matrix A and B should be the same"
    d = len(A)
    factors = [None] * d
    for k in range(d):
        a_k = A[k] # (r_{k-1}, i_k, j_k, r_k)
        b_k = B[k] # (d_{k-1}, j_k, l_k, d_k)
        # print(f"{k} shape: {a_k.shape}, {b_k.shape}")
        assert len(a_k.shape) == 4, "The matrix factor should be order-4 tensor"
        assert len(b_k.shape) == 4, "The vector factor should be order-4 tensor"
        assert a_k.shape[2] == b_k.shape[1], "The inner dimensions of the matrix A and B should be the same"
        r_prev, i_k, j_k, r_next = a_k.shape
        d_prev, j_k, l_k, d_next = b_k.shape
        a_k = a_k.transpose(0, 1, 3, 2) # (r_{k-1}, i_k, r_k, j_k)
        a_k = a_k.reshape(-1, j_k) # (r_{k-1} * i_k * r_k, j_k)
        b_k = b_k.transpose(1, 0, 2, 3) # (j_k, d_{k-1}, l_k, d_k)
        b_k = b_k.reshape(j_k, -1) # (j_k, d_{k-1} * l_k * d_k)
        # print(f"{k} dot: {a_k.shape}, {b_k.shape}")
        factors[k] = tl.dot(a_k, b_k) # (r_{k-1} * i_k * r_k, d_{k-1} * l_k * d_k)
        factors[k] = factors[k].reshape(r_prev, i_k, r_next, d_prev, l_k, d_next)
        factors[k] = factors[k].transpose(0, 3, 1, 4, 2, 5) # (r_{k-1}, d_{k-1}, i_k, l_k, r_k, d_k)
        factors[k] = factors[k].reshape(r_prev * d_prev, i_k, l_k, d_next * r_next)
    return MPO(factors)

def validate_stacked_product(y_mpo, y, rtol=0.00001, atol=1e-8):
    """Validate the stacked matrix-by-vector product of MPO Decomposition

    Parameters
    ----------
    y_mpo : MPO
        stacked MPO Decomposition of the product
    y : tensor
        tensor of the product
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    Raises error if the product is not valid
    """
    for i in range(len(y_mpo)):
        y_mpo[i] = torch.tensor(tt_to_tensor(y_mpo[i]))
        y_mpo[i] = y_mpo[i].reshape(-1)
    y_tensor = torch.vstack(y_mpo).transpose(1,0).reshape(-1)
    y = y.reshape(-1)
    print(f"Error of stacked MPO-form matrix-by-vector product: {torch.max(torch.abs(y_tensor - y))}")
    assert torch.allclose(y_tensor, y, rtol, atol)


def validate_product(y_mpo, y, rtol=0.00001, atol=1e-8):
    """Validate the product of MPO Decomposition

    Parameters
    ----------
    y_mpo : MPO
        MPO Decomposition of the product
    y : tensor
        tensor of the product
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    Raises error if the product is not valid
    """
    if len(y_mpo[0].shape) == 3:
        y_tensor = torch.tensor(tt_to_tensor(y_mpo))
        y_tensor = y_tensor.reshape(-1)
        y = y.reshape(-1)
        print(f"Error of MPO-form matrix-by-vector product: {torch.max(torch.abs(y - y_tensor))}")
        assert torch.allclose(y, y_tensor, rtol, atol)
    elif len(y_mpo[0].shape) == 4:
        y_tensor = torch.tensor(mpo_to_tensor(y_mpo))
        order = get_permutation(y_tensor, decompose=False)
        y_tensor = y_tensor.permute(order)
        y_tensor = y_tensor.reshape(-1)
        y = y.reshape(-1)
        print(f"Error of MPO-form matrix-by-matrix product: {torch.max(torch.abs(y - y_tensor))}")
        assert torch.allclose(y, y_tensor, rtol, atol)
    else:
        raise ValueError("The order of the MPO factors should be 3 or 4")


def validate_decomposition(W_mpo, W, rtol=0.00001, atol=1e-8):
    """Validate the MPO Decomposition

    Parameters
    ----------
    W_mpo : MPO
        MPO Decomposition of the tensor
    W : tensor
        tensor to decompose
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    Raises error if the decomposition is not valid
    """
    W_tensor = mpo_to_tensor(W_mpo)
    W_tensor = torch.tensor(W_tensor)
    order = get_permutation(W_tensor, decompose=False)
    W_tensor = W_tensor.permute(order)
    W_tensor = W_tensor.reshape(-1)
    W = W.reshape(-1)
    print(f"Error of MPO Decomposition: {torch.max(torch.abs(W - W_tensor))}")
    assert torch.allclose(W, W_tensor, rtol, atol)


import tensorly as tl
from tensorly.tt_tensor import FactorizedTensor

def _validate_mpo(factors):
    n_factors = len(factors)

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank = tl.shape(factor)[0]
        next_rank = tl.shape(factor)[-1]
        current_shape = []
        for r in tl.shape(factor)[1:-1]:
            current_shape.append(r)

        # Consecutive factors should have matching ranks
        if index and tl.shape(factors[index - 1])[-1] != current_rank:
            raise ValueError(
                "Consecutive factors should have matching ranks\n"
                " -- e.g. tl.shape(factors[0])[2]) == tl.shape(factors[1])[0])\n"
                f"However, tl.shape(factor[{index-1}])[2] == {tl.shape(factors[index - 1])[2]} but"
                f" tl.shape(factor[{index}])[0] == {current_rank} "
            )
        # Check for boundary conditions
        if (index == 0) and current_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[0].shape[0] == 1."
                f"However, got factor[0].shape[0] = {current_rank}."
            )
        if (index == n_factors - 1) and next_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[-1].shape[2] == 1."
                f"However, got factor[{n_factors}].shape[2] = {next_rank}."
            )

        # shape.append(current_shape)
        shape.extend(current_shape)
        rank.append(current_rank)

    # Add last rank (boundary condition)
    rank.append(next_rank)

    return tuple(shape), tuple(rank)

class MPO(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()

        # Will raise an error if invalid
        shape, rank = _validate_mpo(factors)

        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    def __getitem__(self, index):
        return self.factors[index]

    def __setitem__(self, index, value):
        self.factors[index] = value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        message = f"factors list : rank-{self.rank} matrix-product-state tensor of shape {self.shape} "
        return message

    def to_tensor(self):
        raise NotImplementedError

    def to_unfolding(self, mode):
        raise NotImplementedError

    def to_vec(self):
        raise NotImplementedError
import numpy as np
import sympy as sp

def tensor_to_matrix_rowmajor(t0, indices):
    # Get all indices that are going to form rows
    t0_ind_rows = [ind for ind in range(len(t0.shape)) if ind not in indices]
    # Get all indices that are going to form columns
    t0_ind_cols = list(indices)
    new_ind_order = t0_ind_rows + t0_ind_cols
    # Get number of rows
    remaining_idx_ranges = [t0.shape[ind] for ind in t0_ind_rows]
    n_rows = int(np.multiply.reduce(remaining_idx_ranges))
    n_cols = int(np.multiply.reduce([t0.shape[ind] for ind in t0_ind_cols]))
    matrix = np.ascontiguousarray(t0.transpose(new_ind_order).reshape(n_rows, n_cols))
    return matrix, t0_ind_rows, remaining_idx_ranges


def tensor_to_matrix_colmajor(t0, indices):
    # Get all indices that are going to form columns
    t0_ind_cols = [ind for ind in range(len(t0.shape)) if ind not in indices]
    # Get all indices that are going to form rows
    t0_ind_rows = list(indices)
    new_ind_order = t0_ind_cols + t0_ind_rows
    # Get number of rows
    remaining_idx_ranges = [t0.shape[ind] for ind in t0_ind_cols]
    n_rows = int(np.multiply.reduce([t0.shape[ind] for ind in t0_ind_rows]))
    n_cols = int(np.multiply.reduce(remaining_idx_ranges))
    matrix = np.ascontiguousarray(t0.transpose(new_ind_order).reshape(n_rows, n_cols))
    return matrix, t0_ind_cols, remaining_idx_ranges


def contract(t0, t1=None, indices=None):
    if t1 is None:
        # assume t1 is delta and t0 is "hyper square" then trace
        t0_range = len(t0.shape)
        t1 = np.zeros(t0_range**t0_range)
        t1 = t1.reshape([t0_range for i in range(t0_range)])
        for i in range(t0_range):
            t1[(i,) * t0_range] = 1
        # indices should just be one list, so we create another identical one
        indices = [indices, indices]
    a, a_remaining_d, a_idx_ranges = tensor_to_matrix_rowmajor(t0, indices[0])
    b, b_remaining_d, b_idx_ranges = tensor_to_matrix_colmajor(t1, indices[1])
    result = a @ b
    result = result.reshape(a_idx_ranges + b_idx_ranges)
    # The matrix is currently in this order
    current_order = a_remaining_d + list(indices[0])
    # But needs to be transposed back to its original:
    new_ind_order = [current_order.index(i) for i in range(len(result.shape))]
    result = result.transpose(new_ind_order)
    return result


##############################################
##############################################

e_2=(np.array([1, 0]), np.array([0, 1]))
e_3=(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

def canonical_reshape(psi):
    canonical_indices = sp.factorint(psi.size)
    canonical_indices = [k for k in canonical_indices.keys() for _ in range(canonical_indices[k])]
    # reshape psi into a tensor of canonical indices
    return psi.reshape(*canonical_indices)

def contract_tensors(A, i_A, B, i_B):
    if isinstance(i_A, int):
        i_A = [i_A]
    if isinstance(i_B, int):
        i_B = [i_B]

    A = np.moveaxis(A, i_A, [i for i in range(len(i_A))] )
    # reorder the indices of B so the i_B is the first index
    B = np.moveaxis(B, i_B, [i for i in range(len(i_B))])

    n_A = np.prod([A.shape[i] for i in range(-len(i_A),0,1)])
    n_B = np.prod([B.shape[i] for i in range(0,len(i_B),1)])
    return canonical_reshape(A.reshape(n_A, A.size//n_A).T @ B.reshape(n_B, B.size//n_B))

CN_m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CN_m = canonical_reshape(CN_m)
eye = canonical_reshape(np.eye(2,2))

x = contract_tensors(CN_m, [1,3], eye,[0, 1])
# contract(CN_m, np.eye(2), indices = [0,0])

# psi = np.kron(e_2[0],np.kron(e_3[0],e_2[1]))
# psi = canonical_reshape(psi)

# phi = np.kron(e_3[1],e_2[0])
# phi = canonical_reshape(phi)

# inner product of psi and phi
# A = psi.reshape(psi.size//phi.size,phi.size) @ phi.reshape(phi.size)


# psi_A = e_2[0].reshape(2,) @ (psi.reshape(2, psi.size//4, 2) @ e_2[0].reshape(2,) )

print()
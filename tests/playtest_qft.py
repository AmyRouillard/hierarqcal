from hierarqcal import (
    Qcycle,
    Qmotif,
    Qinit,
    Qmask,
    Qunmask,
    Qpermute,
    Qpivot,
    plot_circuit,
    plot_motif,
    get_tensor_as_f,
    Qunitary,
    Qhierarchy,
)
from hierarqcal.utils import product2tensor
import numpy as np
import sympy as sp
import itertools as it

# ====== Matrices
toffoli_m = np.identity(8)
toffoli_m[6, 6] = 0
toffoli_m[6, 7] = 1
toffoli_m[7, 6] = 1
toffoli_m[7, 7] = 0
cnot_m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
h_m = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
swap_m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
x_m = np.array([[0, 1], [1, 0]])
x0 = sp.symbols("x")
cp_m = sp.Matrix(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sp.exp(sp.I * x0)]]
)


# ====== Gates level 0
toffoli = Qunitary(get_tensor_as_f(toffoli_m), 0, 3)
cnot = Qunitary(get_tensor_as_f(cnot_m), 0, 2, name="CX")
hadamard = Qunitary(get_tensor_as_f(h_m), 0, 1, name="H")
swap = Qunitary(get_tensor_as_f(swap_m), 0, 2)
x = Qunitary(get_tensor_as_f(x_m), 0, 1)
cphase = Qunitary(get_tensor_as_f(cp_m), arity=2, symbols=[x0], name="CP")
cphased = Qunitary(get_tensor_as_f(cp_m), arity=2, symbols=[x0], name="CPd")

# ===== QFT
n = 4

qft = (
    Qinit(n)
    + (
        Qpivot(mapping=hadamard)
        + Qpivot(mapping=cphase, share_weights=False)
        + Qmask("1*")
    )
    * n
)

# plot_circuit(qft, plot_width=25)
# print()

# hierq = Qinit(5) + Qmask("!*")
# plot_circuit(hierq, plot_width=25)

# hierq = Qinit(5) + Qmask("!*", mapping=cnot)
# plot_circuit(hierq, plot_width=25)


# hierq = Qinit(5) + Qcycle(mapping=cnot)
# plot_circuit(hierq, plot_width=25)

# hierq = Qinit(5) + Qpivot(mapping=cnot, global_pattern="*1*")
# plot_circuit(hierq, plot_width=25)

hierq = Qinit(["x0", "y0", "z0"]) + (
    Qpivot(mapping=hadamard, global_pattern="*1")
    + Qpivot(mapping=cphase, global_pattern="*1", merge_within="01")
    + Qpivot(mapping=cnot, global_pattern="11*", merge_within="11")
    + Qpivot(mapping=cphased, global_pattern="*11", merge_within="11")
    + Qpivot(mapping=cnot, global_pattern="11*", merge_within="11")
    + Qpivot(mapping=hadamard, global_pattern="*1")
)
plot_circuit(hierq, plot_width=25)

print()

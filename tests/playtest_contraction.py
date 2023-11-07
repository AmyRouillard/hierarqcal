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
from hierarqcal.utils import product2tensor, canonical_reshape
import numpy as np
import sympy as sp
import itertools as it
import matplotlib.pyplot as plt

h_m = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
hadamard = Qunitary(get_tensor_as_f(h_m), 0, 1)


bits = "01"

nq = len(bits)

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])

tensors = [ket0 if elem == 0 else ket1 for elem in bits[::-1]]
# use the identity as input tensor
input_tensor = product2tensor(
    [np.array([[1, 0], [0, 1]], dtype=np.clongdouble)] * nq  #
)


circ = (
    Qinit(nq, tensors=input_tensor)  # Qinit(nq, tensors=product2tensor(tensors))
    + Qcycle(mapping=hadamard)
    # + Qcycle(mapping=hadamard)
)

plot_circuit(circ, plot_width=10)

print()

output_tensor = circ()

print()

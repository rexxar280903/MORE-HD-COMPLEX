"""Circuit definitions and the analytic statevector engine (pseudocode §4, §11.6; G4-01, G4-03).

Every model is defined once as a gate list (``CircuitModel``). Two executors
read the same gate list:

* ``BatchedStatevector`` -- production engine (NumPy, complex128, shots=None).
  It exploits the circuit structure ``|psi(x)> = U(theta) (|phi(x)> (x) |0..0>)``:
  the data encoding is a product state, so per objective evaluation the engine
  builds the columns of ``U(theta)`` once and applies them to all samples with a
  fixed-shape matrix product, then reads the readout Pauli expectations from the
  reduced density matrix of the readout wires (exact, one pass per sample).
* ``pennylane_reference`` -- PennyLane ``default.qubit`` QNode with the same gates,
  used by the unit tests and by the per-run backend self-check to prove the
  engine reproduces the reference simulator.

Wire convention: wire 0 is the most significant bit of the computational-basis
index (PennyLane convention). Data wires are wires ``0..n_data-1``; any remaining
wires start in |0>. Readout wires are the last wires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import (
    ARCH_MORE_HD,
    ARCH_MORE_HD_60P,
    ARCH_MORE_HD_C,
    ARCH_MORE_HD_C_FIXED_RZ,
    ARCH_MORE_REPRO,
    ENGINE_CHUNK_ROWS,
    OBSERVABLE_LABELS_15,
    OBSERVABLE_LABELS_MORE,
)
from .seeds import derive_subseed

# ---------------------------------------------------------------------------
# gate matrices (Qiskit/PennyLane conventions; identical definitions)
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
PX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI_MATRIX = {"I": I2, "X": PX, "Y": PY, "Z": PZ}
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)


def m_rx(t: float) -> np.ndarray:
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def m_ry(t: float) -> np.ndarray:
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def m_rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=np.complex128)


def m_p(t: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * t)]], dtype=np.complex128)


def m_ryy(t: float) -> np.ndarray:
    return math.cos(t / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(t / 2.0) * np.kron(PY, PY)


def m_rzz(t: float) -> np.ndarray:
    return math.cos(t / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(t / 2.0) * np.kron(PZ, PZ)


ONE_QUBIT = {"RX": m_rx, "RY": m_ry, "RZ": m_rz, "P": m_p, "H": None}
TWO_QUBIT = {"RYY": m_ryy, "RZZ": m_rzz, "CNOT": None}

# ---------------------------------------------------------------------------
# gate list representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    """One gate. ``source``: 'x' (feature), 'theta' (trainable), 'fixed', or 'none'."""

    gate: str
    wires: tuple
    source: str = "none"
    index: int = -1
    scale: float = 1.0

    def angle(self, x_row=None, theta=None, fixed=None) -> float:
        if self.source == "x":
            return self.scale * float(x_row[self.index])
        if self.source == "theta":
            return self.scale * float(theta[self.index])
        if self.source == "fixed":
            return self.scale * float(fixed[self.index])
        raise ValueError("gate has no angle")


@dataclass
class CircuitModel:
    name: str
    n_wires: int
    n_data_wires: int
    readout_wires: tuple
    encoding: list
    body: list
    n_trainable: int
    observable_labels: tuple
    fixed_values: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_fixed(self) -> int:
        return 0 if self.fixed_values is None else int(len(self.fixed_values))

    @property
    def n_observables(self) -> int:
        return len(self.observable_labels)

    def observable_terms(self) -> list:
        """[(label, {wire: pauli})] in the locked order."""
        terms = []
        for label in self.observable_labels:
            factors = {w: p for w, p in zip(self.readout_wires, label) if p != "I"}
            terms.append((label, factors))
        return terms

    def gate_counts(self) -> dict:
        counts: dict = {}
        for op in self.encoding + self.body:
            counts[op.gate] = counts.get(op.gate, 0) + 1
        return counts

    def depth(self) -> int:
        """Circuit depth (longest path in gate layers), encoding included."""
        level = [0] * self.n_wires
        for op in self.encoding + self.body:
            d = max(level[w] for w in op.wires) + 1
            for w in op.wires:
                level[w] = d
        return max(level)

    def resource_summary(self) -> dict:
        counts = self.gate_counts()
        return {
            "model": self.name,
            "n_wires": self.n_wires,
            "n_trainable_params": self.n_trainable,
            "n_fixed_params": self.n_fixed,
            "gate_count_total": sum(counts.values()),
            "gate_counts": counts,
            "cnot_count": counts.get("CNOT", 0),
            "depth": self.depth(),
            **self.meta,
        }


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------

N_DATA = 8
N_READOUT = 2


def _ry_encoding() -> list:
    return [Op("RY", (i,), "x", i, 1.0) for i in range(N_DATA)]


def _entangling_block_cnot() -> list:
    """ENTANGLING_BLOCK_CNOT (§4.3): ring on data wires, last data -> both readouts, readout corr."""
    ops = [Op("CNOT", (i, i + 1)) for i in range(N_DATA - 1)]
    ops.append(Op("CNOT", (N_DATA - 1, 0)))
    ops.append(Op("CNOT", (N_DATA - 1, N_DATA)))
    ops.append(Op("CNOT", (N_DATA - 1, N_DATA + 1)))
    ops.append(Op("CNOT", (N_DATA, N_DATA + 1)))
    return ops


def build_more_hd(n_layers: int = 3, name: str = ARCH_MORE_HD) -> CircuitModel:
    """BUILD_CIRCUIT_MORE_HD (§4.1); with n_layers=6 this is MORE-HD-60P (§11.6.1)."""
    n_wires = N_DATA + N_READOUT
    body = []
    for layer in range(n_layers):
        for q in range(n_wires):
            body.append(Op("RY", (q,), "theta", layer * n_wires + q))
        body.extend(_entangling_block_cnot())
    return CircuitModel(
        name=name,
        n_wires=n_wires,
        n_data_wires=N_DATA,
        readout_wires=(N_DATA, N_DATA + 1),
        encoding=_ry_encoding(),
        body=body,
        n_trainable=n_layers * n_wires,
        observable_labels=OBSERVABLE_LABELS_15,
        meta={"n_variational_layers": n_layers, "entangling_block_count": n_layers,
              "state_family": "real", "packing_rule": f"RY_ONLY_{n_layers}x10"},
    )


def build_more_hd_c(n_layers: int = 3) -> CircuitModel:
    """BUILD_CIRCUIT_MORE_HD_C (§4.2): per layer theta = [10 RY, 10 RZ]."""
    n_wires = N_DATA + N_READOUT
    body = []
    for layer in range(n_layers):
        base = layer * 2 * n_wires
        for q in range(n_wires):
            body.append(Op("RY", (q,), "theta", base + q))
            body.append(Op("RZ", (q,), "theta", base + n_wires + q))
        body.extend(_entangling_block_cnot())
    return CircuitModel(
        name=ARCH_MORE_HD_C,
        n_wires=n_wires,
        n_data_wires=N_DATA,
        readout_wires=(N_DATA, N_DATA + 1),
        encoding=_ry_encoding(),
        body=body,
        n_trainable=n_layers * 2 * n_wires,
        observable_labels=OBSERVABLE_LABELS_15,
        meta={"n_variational_layers": n_layers, "entangling_block_count": n_layers,
              "state_family": "complex", "packing_rule": "PER_LAYER_[10_RY,10_RZ]_x3"},
    )


def build_more_hd_c_fixed_rz(fixed_rz: np.ndarray, n_layers: int = 3) -> CircuitModel:
    """BUILD_CIRCUIT_MORE_HD_C_FIXED_RZ (§11.6.2): 30 trainable RY, 30 frozen non-zero RZ."""
    fixed_rz = np.asarray(fixed_rz, dtype=np.float64)
    n_wires = N_DATA + N_READOUT
    if fixed_rz.shape != (n_layers * n_wires,) or np.any(fixed_rz == 0.0):
        raise ValueError("fixed RZ vector must have 30 non-zero angles")
    body = []
    for layer in range(n_layers):
        for q in range(n_wires):
            body.append(Op("RY", (q,), "theta", layer * n_wires + q))
            body.append(Op("RZ", (q,), "fixed", layer * n_wires + q))
        body.extend(_entangling_block_cnot())
    return CircuitModel(
        name=ARCH_MORE_HD_C_FIXED_RZ,
        n_wires=n_wires,
        n_data_wires=N_DATA,
        readout_wires=(N_DATA, N_DATA + 1),
        encoding=_ry_encoding(),
        body=body,
        n_trainable=n_layers * n_wires,
        observable_labels=OBSERVABLE_LABELS_15,
        fixed_values=fixed_rz.copy(),
        meta={"n_variational_layers": n_layers, "entangling_block_count": n_layers,
              "state_family": "complex", "packing_rule": "RY_TRAINABLE_3x10 + RZ_FIXED_3x10"},
    )


# --- MORE (Wu et al., 2023) reproduction: model.py::build_qcnn ---------------------
# Parameter vectors in Qiskit's sorted order (EstimatorQNN weight order):
#   c2[0..23], c3[0..11], p1[0..3], p2[0..1], p3[0], с1[0..47]  (с1 uses Cyrillic 'с')
MORE_PARAM_BLOCKS = (("c2", 24), ("c3", 12), ("p1", 4), ("p2", 2), ("p3", 1), ("с1", 48))
MORE_INPUT_SCALE = 2.0      # MORE scales PCA features to [0, 2pi]; ours are in [0, pi]
MORE_ZFEATURE_ALPHA = 2.0   # Qiskit ZFeatureMap applies P(2 * x)


def _more_offsets() -> dict:
    offsets, pos = {}, 0
    for name, size in MORE_PARAM_BLOCKS:
        offsets[name] = pos
        pos += size
    return offsets


def _more_conv_kernel(ops: list, q1: int, q2: int, prefix: str, start: int, offs: dict) -> None:
    """model.py::conv_circuit: RX, RX, RYY, RZZ, RZ, RZ (6 parameters)."""
    b = offs[prefix] + start
    ops.append(Op("RX", (q1,), "theta", b + 0))
    ops.append(Op("RX", (q2,), "theta", b + 1))
    ops.append(Op("RYY", (q1, q2), "theta", b + 2))
    ops.append(Op("RZZ", (q1, q2), "theta", b + 3))
    ops.append(Op("RZ", (q1,), "theta", b + 4))
    ops.append(Op("RZ", (q2,), "theta", b + 5))


def _more_conv_layer(ops: list, wires: list, prefix: str, offs: dict) -> None:
    """model.py::conv_layer on the given wires (local qubits mapped to ``wires``)."""
    local = list(range(len(wires)))
    idx = 0
    for a, b in zip(local[0::2], local[1::2]):
        _more_conv_kernel(ops, wires[a], wires[b], prefix, idx, offs)
        idx += 6
    for a, b in zip(local[1::2], local[2::2] + [0]):
        _more_conv_kernel(ops, wires[a], wires[b], prefix, idx, offs)
        idx += 6


def _more_pool_layer(ops: list, wires: list, sources: list, sinks: list, prefix: str, offs: dict) -> None:
    """model.py::pool_layer: CX(source, sink) then RY(param) on sink; one parameter per pair."""
    for i, (s, t) in enumerate(zip(sources, sinks)):
        ops.append(Op("CNOT", (wires[s], wires[t])))
        ops.append(Op("RY", (wires[t],), "theta", offs[prefix] + i))


def build_more_repro() -> CircuitModel:
    """MORE (no loss adjuster R) reproduced from github.com/Jindi0/MORE model.py::build_qcnn.

    8 qubits, ZFeatureMap(reps=2) encoding, QCNN ansatz with 91 parameters,
    single readout qubit 7 measured with X, Y, Z. Input mapping: our features are
    MinMax-scaled to [0, pi]; MORE scales PCA features to [0, 2pi], so the encoded
    angle is P(2 * (2 * x)).
    """
    offs = _more_offsets()
    n_wires = 8
    enc = []
    scale = MORE_ZFEATURE_ALPHA * MORE_INPUT_SCALE
    for q in range(n_wires):
        enc.append(Op("H", (q,)))
        enc.append(Op("P", (q,), "x", q, scale))
        enc.append(Op("H", (q,)))
        enc.append(Op("P", (q,), "x", q, scale))
    body: list = []
    _more_conv_layer(body, list(range(8)), "с1", offs)
    _more_pool_layer(body, list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7], "p1", offs)
    _more_conv_layer(body, [4, 5, 6, 7], "c2", offs)
    _more_pool_layer(body, [4, 5, 6, 7], [0, 1], [2, 3], "p2", offs)
    _more_conv_layer(body, [6, 7], "c3", offs)
    _more_pool_layer(body, [6, 7], [0], [1], "p3", offs)
    n_params = sum(size for _, size in MORE_PARAM_BLOCKS)
    assert n_params == 91
    return CircuitModel(
        name=ARCH_MORE_REPRO,
        n_wires=n_wires,
        n_data_wires=n_wires,
        readout_wires=(7,),
        encoding=enc,
        body=body,
        n_trainable=n_params,
        observable_labels=OBSERVABLE_LABELS_MORE,
        meta={"n_variational_layers": 3, "entangling_block_count": 0, "state_family": "complex",
              "packing_rule": "QISKIT_SORTED_PARAMETER_ORDER",
              "source": "github.com/Jindi0/MORE@867d194 model.py::build_qcnn",
              "input_mapping": "ZFeatureMap(reps=2) with P(2 * (2 * x_scaled))"},
    )


def build_model(architecture: str, init_bundle: dict | None = None) -> CircuitModel:
    if architecture == ARCH_MORE_HD:
        return build_more_hd(3)
    if architecture == ARCH_MORE_HD_60P:
        return build_more_hd(6, name=ARCH_MORE_HD_60P)
    if architecture == ARCH_MORE_HD_C:
        return build_more_hd_c(3)
    if architecture == ARCH_MORE_HD_C_FIXED_RZ:
        if init_bundle is None:
            raise ValueError("MORE-HD-C-FixedRZ needs the paired initialization bundle")
        return build_more_hd_c_fixed_rz(init_bundle["rz_phase"])
    if architecture == ARCH_MORE_REPRO:
        return build_more_repro()
    raise ValueError(f"unknown architecture {architecture!r}")


# ---------------------------------------------------------------------------
# deterministic paired initialisation (§4 BUILD_PAIRED_PRIMARY_INITIALIZATION, §11.11.5)
# ---------------------------------------------------------------------------


def build_initialization_bundle(master_seed: int) -> dict:
    rng_ry = np.random.default_rng(derive_subseed(master_seed, "init:ry_core"))
    rng_rz = np.random.default_rng(derive_subseed(master_seed, "init:rz_phase"))
    rng_extra = np.random.default_rng(derive_subseed(master_seed, "ablation:init:ry_extra"))
    rng_more = np.random.default_rng(derive_subseed(master_seed, "init:more_qcnn"))
    ry_core = rng_ry.uniform(0.0, 2.0 * math.pi, size=30)
    rz_phase = rng_rz.uniform(0.0, 2.0 * math.pi, size=30)
    for i in range(30):                      # RZ = 0 is a stationary control; redraw exact zeros
        while rz_phase[i] == 0.0:
            rz_phase[i] = rng_rz.uniform(0.0, 2.0 * math.pi)
    ry_extra = rng_extra.uniform(0.0, 2.0 * math.pi, size=30)
    more_init = rng_more.random(91)          # qiskit-machine-learning default: U[0, 1)
    return {
        "ry_core": ry_core,
        "rz_phase": rz_phase,
        "ry_extra": ry_extra,
        "more_init": more_init,
        "ry_core_seed": derive_subseed(master_seed, "init:ry_core"),
        "rz_phase_seed": derive_subseed(master_seed, "init:rz_phase"),
        "ry_extra_seed": derive_subseed(master_seed, "ablation:init:ry_extra"),
        "more_init_seed": derive_subseed(master_seed, "init:more_qcnn"),
    }


def pack_ry_rz_by_layer(ry_core: np.ndarray, rz_phase: np.ndarray, n_layers: int = 3, n_qubits: int = 10) -> np.ndarray:
    ry = np.asarray(ry_core).reshape(n_layers, n_qubits)
    rz = np.asarray(rz_phase).reshape(n_layers, n_qubits)
    return np.concatenate([np.concatenate([ry[i], rz[i]]) for i in range(n_layers)])


def initial_params(architecture: str, bundle: dict) -> np.ndarray:
    if architecture == ARCH_MORE_HD:
        return bundle["ry_core"].copy()
    if architecture == ARCH_MORE_HD_C:
        return pack_ry_rz_by_layer(bundle["ry_core"], bundle["rz_phase"])
    if architecture == ARCH_MORE_HD_60P:
        return np.concatenate([bundle["ry_core"], bundle["ry_extra"]])
    if architecture == ARCH_MORE_HD_C_FIXED_RZ:
        return bundle["ry_core"].copy()
    if architecture == ARCH_MORE_REPRO:
        return bundle["more_init"].copy()
    raise ValueError(architecture)


# ---------------------------------------------------------------------------
# production engine
# ---------------------------------------------------------------------------


def _bit(index: np.ndarray, wire: int, n_wires: int) -> np.ndarray:
    return (index >> (n_wires - 1 - wire)) & 1


class BatchedStatevector:
    """Exact (shots=None) complex128 statevector simulation of a CircuitModel.

    ``outputs(X, theta)`` returns the readout Pauli expectation values, one row
    per sample, in ``model.observable_labels`` order. A sample's result does not
    depend on which other samples share the batch: rows are processed in
    fixed-shape, zero-padded chunks and every per-sample contraction is a
    separate fixed-shape product (verified in tests/test_engine.py).
    """

    def __init__(self, model: CircuitModel, chunk_rows: int = ENGINE_CHUNK_ROWS):
        self.model = model
        self.chunk_rows = int(chunk_rows)
        n = model.n_wires
        self.n = n
        self.dim = 1 << n
        self.n_data = model.n_data_wires
        self.data_dim = 1 << self.n_data
        n_ro = len(model.readout_wires)
        if tuple(model.readout_wires) != tuple(range(n - n_ro, n)):
            raise ValueError("readout wires must be the last wires")
        if any(len(op.wires) != 1 or op.wires[0] >= self.n_data for op in model.encoding):
            raise ValueError("encoding must be single-qubit gates on data wires")
        self.n_ro = n_ro
        self.ro_dim = 1 << n_ro
        self.n_executions = 0
        self._idx = np.arange(self.dim)
        self._leading, self._steps = self._compile(model.body)
        mats = []
        for label in model.observable_labels:
            m = np.array([[1.0 + 0j]])
            for p in label:
                m = np.kron(m, PAULI_MATRIX[p])
            mats.append(m)
        mats = np.stack(mats)                          # (n_obs, ro_dim, ro_dim)
        self._obs_re_t = np.ascontiguousarray(np.transpose(mats.real, (0, 2, 1)))
        self._obs_im_t = np.ascontiguousarray(np.transpose(mats.imag, (0, 2, 1)))
        self._enc_by_wire = {w: [op for op in model.encoding if op.wires[0] == w] for w in range(self.n_data)}

    # -- compilation -----------------------------------------------------------
    def _compile(self, body: list):
        """Split the body into (a) leading single-qubit gates per wire, folded into an
        initial Kronecker product, and (b) a step list: merged single-qubit gates,
        CNOT runs merged into one row permutation, and two-qubit gates."""
        leading = {w: [] for w in range(self.n)}
        touched: set = set()
        steps: list = []
        pending: dict = {}

        def flush():
            for w in sorted(pending):
                steps.append(("1q", w, tuple(pending[w])))
            pending.clear()

        for op in body:
            if len(op.wires) == 1:
                w = op.wires[0]
                if w not in touched:
                    leading[w].append(op)
                else:
                    pending.setdefault(w, []).append(op)
                continue
            flush()
            touched.update(op.wires)
            if op.gate == "CNOT":
                c, t = op.wires
                p = self._idx ^ (_bit(self._idx, c, self.n) << (self.n - 1 - t))
                if steps and steps[-1][0] == "perm":
                    steps[-1] = ("perm", steps[-1][1][p])
                else:
                    steps.append(("perm", p))
            elif op.gate in TWO_QUBIT:
                steps.append(("2q", op, self._xor_sources(*op.wires)))
            else:
                raise ValueError(op.gate)
        flush()
        return leading, steps

    def _xor_sources(self, w1: int, w2: int):
        n = self.n
        b1 = _bit(self._idx, w1, n)
        b2 = _bit(self._idx, w2, n)
        local = 2 * b1 + b2
        srcs = []
        for m in range(4):
            m1, m2 = m >> 1, m & 1
            srcs.append(self._idx ^ ((m1 << (n - 1 - w1)) | (m2 << (n - 1 - w2))))
        return local, srcs

    # -- gates -------------------------------------------------------------------
    def _op_matrix(self, op: Op, theta, fixed) -> np.ndarray:
        if op.gate == "H":
            return HADAMARD
        fn = ONE_QUBIT.get(op.gate) or TWO_QUBIT.get(op.gate)
        return fn(op.angle(theta=theta, fixed=fixed))

    def _merged(self, ops, theta, fixed) -> np.ndarray:
        g = I2
        for op in ops:
            g = self._op_matrix(op, theta, fixed) @ g
        return g

    def _apply_1q(self, t: np.ndarray, g: np.ndarray, w: int) -> np.ndarray:
        v = t.reshape(1 << w, 2, -1)
        return np.matmul(g, v).reshape(t.shape)

    @staticmethod
    def _apply_2q(t: np.ndarray, g4: np.ndarray, local: np.ndarray, srcs: list) -> np.ndarray:
        """out[row] = sum_m G[local(row), local(row) ^ m] * t[row ^ mask(m)]."""
        out = None
        for m in range(4):
            coef = g4[local, local ^ m]
            if not np.any(coef):
                continue
            term = coef[:, None] * (t if m == 0 else t[srcs[m], :])
            out = term if out is None else out + term
        return out

    def unitary_columns(self, theta: np.ndarray) -> np.ndarray:
        """Columns of U(theta) acting on |d> (x) |0..0>: shape (dim, data_dim), complex128."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.model.n_trainable,):
            raise ValueError(f"theta shape {theta.shape} != ({self.model.n_trainable},)")
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("non-finite parameter vector")
        fixed = self.model.fixed_values
        t = np.ones((1, 1), dtype=np.complex128)
        for w in range(self.n):
            g = self._merged(self._leading[w], theta, fixed)
            if w < self.n_data:
                t = np.kron(t, g)                      # column index = data basis state
            else:
                t = np.kron(t, g[:, :1])               # non-data wires start in |0>
        for step in self._steps:
            kind = step[0]
            if kind == "perm":
                t = t[step[1], :]
            elif kind == "1q":
                t = self._apply_1q(t, self._merged(step[2], theta, fixed), step[1])
            else:
                op, (local, srcs) = step[1], step[2]
                t = self._apply_2q(t, self._op_matrix(op, theta, fixed), local, srcs)
        return np.ascontiguousarray(t)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Product-state amplitudes of the data wires: (N, data_dim), real when possible."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("X must be 2-D")
        n = x.shape[0]
        phi = np.ones((n, 1), dtype=np.complex128)
        for w in range(self.n_data):
            vec = np.zeros((n, 2), dtype=np.complex128)
            vec[:, 0] = 1.0
            for op in self._enc_by_wire[w]:
                if op.gate == "H":           # elementwise (no BLAS) to stay batch-independent
                    inv = 1.0 / math.sqrt(2.0)
                    vec = np.stack([(vec[:, 0] + vec[:, 1]) * inv, (vec[:, 0] - vec[:, 1]) * inv], axis=1)
                    continue
                ang = op.scale * x[:, op.index]
                if op.gate == "RY":
                    c, s = np.cos(ang / 2.0), np.sin(ang / 2.0)
                    vec = np.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]], axis=1)
                elif op.gate == "P":
                    vec = np.stack([vec[:, 0], np.exp(1j * ang) * vec[:, 1]], axis=1)
                else:
                    raise ValueError(f"unsupported encoding gate {op.gate}")
            phi = (phi[:, :, None] * vec[:, None, :]).reshape(n, -1)
        if np.all(phi.imag == 0.0):
            return np.ascontiguousarray(phi.real)
        return np.ascontiguousarray(phi)

    def expectations(self, phi: np.ndarray, u_cols: np.ndarray) -> np.ndarray:
        n = phi.shape[0]
        out = np.empty((n, self.model.n_observables), dtype=np.float64)
        rows = self.chunk_rows
        real_phi = not np.iscomplexobj(phi)
        u_t = u_cols.T                                      # (data_dim, dim)
        u_re_t = np.ascontiguousarray(u_t.real)
        u_im_t = np.ascontiguousarray(u_t.imag)
        # Only real GEMMs (dgemm) are used: their per-row results do not depend on the
        # row's position in the fixed-shape chunk, which keeps outputs batch-independent.
        buf = np.zeros((rows, self.data_dim), dtype=np.float64)
        buf_i = None if real_phi else np.zeros((rows, self.data_dim), dtype=np.float64)
        r = self.ro_dim
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            m = stop - start
            if real_phi:
                buf[:m] = phi[start:stop]
                buf[m:] = 0.0
                a = buf @ u_re_t
                b = buf @ u_im_t
            else:
                buf[:m] = phi[start:stop].real
                buf_i[:m] = phi[start:stop].imag
                buf[m:] = 0.0
                buf_i[m:] = 0.0
                a = buf @ u_re_t - buf_i @ u_im_t
                b = buf @ u_im_t + buf_i @ u_re_t
            ab = np.concatenate(
                [a.reshape(rows, -1, r), b.reshape(rows, -1, r)], axis=2
            )                                                  # (rows, dim/r, 2r)
            gram = np.matmul(np.ascontiguousarray(ab.transpose(0, 2, 1)), ab)[:m]
            rho_re = gram[:, :r, :r] + gram[:, r:, r:]
            rho_im = gram[:, r:, :r] - gram[:, :r, r:]
            out[start:stop] = (
                np.einsum("crs,prs->cp", rho_re, self._obs_re_t)
                - np.einsum("crs,prs->cp", rho_im, self._obs_im_t)
            )
        self.n_executions += n
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite circuit output")
        return out

    def outputs(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.expectations(self.encode(x), self.unitary_columns(theta))


# ---------------------------------------------------------------------------
# PennyLane reference executor (default.qubit, shots=None)
# ---------------------------------------------------------------------------


def pennylane_reference(model: CircuitModel) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    import pennylane as qml

    dev = qml.device("default.qubit", wires=model.n_wires)
    gate_map = {
        "RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ, "P": qml.PhaseShift,
        "H": qml.Hadamard, "CNOT": qml.CNOT, "RYY": qml.IsingYY, "RZZ": qml.IsingZZ,
    }
    pauli_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    terms = model.observable_terms()
    fixed = model.fixed_values

    @qml.qnode(dev, diff_method=None)
    def circuit(x_row, theta):
        for op in model.encoding + model.body:
            fn = gate_map[op.gate]
            if op.gate in ("H", "CNOT"):
                fn(wires=list(op.wires))
            else:
                fn(op.angle(x_row=x_row, theta=theta, fixed=fixed), wires=list(op.wires))
        results = []
        for _, factors in terms:
            obs = None
            for w, p in sorted(factors.items()):
                o = pauli_map[p](w)
                obs = o if obs is None else obs @ o
            results.append(qml.expval(obs))
        return results

    def run(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.array([np.asarray(circuit(row, np.asarray(theta, dtype=np.float64)), dtype=np.float64)
                         for row in x])

    return run


def backend_self_check(model: CircuitModel, theta: np.ndarray, x: np.ndarray) -> dict:
    """Per-run cross-check of the production engine against PennyLane default.qubit."""
    import pennylane as qml

    eng = BatchedStatevector(model)
    ours = eng.outputs(x, theta)
    ref = pennylane_reference(model)(x, theta)
    diff = float(np.max(np.abs(ours - ref)))
    return {
        "reference_device": "default.qubit",
        "pennylane_version": qml.__version__,
        "n_samples": int(x.shape[0]),
        "max_abs_diff": diff,
    }

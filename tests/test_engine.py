"""G4-01 / G4-03: circuit structure, engine vs PennyLane and Qiskit, structural zeros."""

import numpy as np
import pytest

from core import constants as C
from core.circuits import (
    BatchedStatevector,
    build_initialization_bundle,
    build_model,
    initial_params,
    pack_ry_rz_by_layer,
    pennylane_reference,
)
from core.numerics import yodd_norm_fraction

ARCHS = ["MORE-HD", "MORE-HD-C", "MORE-HD-60P", "MORE-HD-C-FixedRZ", "MORE-REPRO"]
N_PARAMS = {"MORE-HD": 30, "MORE-HD-C": 60, "MORE-HD-60P": 60, "MORE-HD-C-FixedRZ": 30, "MORE-REPRO": 91}


@pytest.fixture(scope="module")
def bundle():
    return build_initialization_bundle(101)


@pytest.mark.parametrize("arch", ARCHS)
def test_parameter_count_and_output_shape(arch, bundle):
    m = build_model(arch, bundle)
    assert m.n_trainable == N_PARAMS[arch]
    assert initial_params(arch, bundle).shape == (N_PARAMS[arch],)
    out = BatchedStatevector(m).outputs(np.full((3, 8), 0.4), initial_params(arch, bundle))
    n_obs = 3 if arch == "MORE-REPRO" else 15
    assert out.shape == (3, n_obs)
    assert np.all(np.abs(out) <= 1.0 + 1e-12)
    if arch == "MORE-HD-C-FixedRZ":
        assert m.n_fixed == 30 and np.all(m.fixed_values != 0.0)


def test_observable_order_locked():
    assert build_model("MORE-HD").observable_labels == C.OBSERVABLE_LABELS_15
    assert C.YODD_INDICES_15 == (1, 5, 7, 8, 10, 13)
    assert [C.OBSERVABLE_LABELS_15[i] for i in C.YODD_INDICES_15] == ["IY", "XY", "YI", "YX", "YZ", "ZY"]
    assert build_model("MORE-REPRO").observable_labels == ("X", "Y", "Z")


@pytest.mark.parametrize("arch", ARCHS)
def test_engine_matches_pennylane_default_qubit(arch, bundle):
    rng = np.random.default_rng(7)
    m = build_model(arch, bundle)
    theta = initial_params(arch, bundle) + rng.normal(0, 0.5, m.n_trainable)
    x = rng.uniform(0, np.pi, (5, 8))
    ours = BatchedStatevector(m).outputs(x, theta)
    ref = pennylane_reference(m)(x, theta)
    assert np.max(np.abs(ours - ref)) < 1e-12


@pytest.mark.parametrize("arch", ["MORE-HD", "MORE-HD-60P"])
def test_real_family_has_exact_structural_zeros(arch, bundle):
    rng = np.random.default_rng(3)
    m = build_model(arch, bundle)
    eng = BatchedStatevector(m)
    for _ in range(3):
        out = eng.outputs(rng.uniform(0, np.pi, (50, 8)), rng.uniform(0, 2 * np.pi, m.n_trainable))
        assert np.max(np.abs(out[:, list(C.YODD_INDICES_15)])) <= 1e-15
        frac, _ = yodd_norm_fraction(out, C.YODD_INDICES_15)
        assert frac < 1e-20


@pytest.mark.parametrize("arch", ["MORE-HD-C", "MORE-HD-C-FixedRZ"])
def test_complex_family_can_activate_yodd(arch, bundle):
    rng = np.random.default_rng(4)
    m = build_model(arch, bundle)
    out = BatchedStatevector(m).outputs(rng.uniform(0, np.pi, (50, 8)), initial_params(arch, bundle))
    frac, _ = yodd_norm_fraction(out, C.YODD_INDICES_15)
    assert frac > 1e-6
    assert np.max(np.abs(out[:, list(C.YODD_INDICES_15)])) > 1e-6


@pytest.mark.parametrize("arch", ARCHS)
def test_outputs_are_batch_independent_and_deterministic(arch, bundle):
    rng = np.random.default_rng(11)
    m = build_model(arch, bundle)
    eng = BatchedStatevector(m, chunk_rows=64)
    theta = initial_params(arch, bundle)
    x = rng.uniform(0, np.pi, (150, 8))
    full = eng.outputs(x, theta)
    again = eng.outputs(x, theta)
    assert np.array_equal(full, again)
    for i in (0, 63, 64, 149):
        single = eng.outputs(x[i:i + 1], theta)
        assert np.array_equal(single[0], full[i])
    perm = rng.permutation(150)
    assert np.array_equal(eng.outputs(x[perm], theta), full[perm])


def test_paired_initialization_rules(bundle):
    a = initial_params("MORE-HD", bundle)
    b = initial_params("MORE-HD-60P", bundle)
    c = initial_params("MORE-HD-C-FixedRZ", bundle)
    d = initial_params("MORE-HD-C", bundle)
    assert np.array_equal(a, bundle["ry_core"]) and np.array_equal(c, a)
    assert np.array_equal(b[:30], a)
    assert np.array_equal(d, pack_ry_rz_by_layer(bundle["ry_core"], bundle["rz_phase"]))
    assert np.array_equal(d[0:10], a[0:10]) and np.array_equal(d[10:20], bundle["rz_phase"][0:10])
    assert np.array_equal(build_model("MORE-HD-C-FixedRZ", bundle).fixed_values, bundle["rz_phase"])
    assert np.all(bundle["rz_phase"] != 0.0)
    again = build_initialization_bundle(101)
    for key in ("ry_core", "rz_phase", "ry_extra", "more_init"):
        assert np.array_equal(again[key], bundle[key])
    assert np.all((bundle["more_init"] >= 0) & (bundle["more_init"] < 1))


def test_circuit_resources():
    a, b, d = build_model("MORE-HD"), build_model("MORE-HD-60P"), build_model("MORE-HD-C")
    assert a.resource_summary()["cnot_count"] == 33 and b.resource_summary()["cnot_count"] == 66
    assert d.resource_summary()["cnot_count"] == 33
    assert a.meta["n_variational_layers"] == 3 and b.meta["n_variational_layers"] == 6


# ---------------------------------------------------------------------------
# MORE reproduction vs the original Qiskit code (github.com/Jindi0/MORE model.py)
# ---------------------------------------------------------------------------

qiskit = pytest.importorskip("qiskit")


def _more_qiskit_circuit():
    """Verbatim structure of MORE model.py::build_qcnn (conv_circuit, conv_layer, pool_layer)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZFeatureMap

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rx(params[0], 0)
        target.rx(params[1], 1)
        target.ryy(params[2], 0, 1)
        target.rzz(params[3], 0, 1)
        target.rz(params[4], 0)
        target.rz(params[5], 1)
        return target

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 6)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index:(param_index + 6)]), [q1, q2])
            param_index += 6
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index:(param_index + 6)]), [q1, q2])
            param_index += 6
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.cx(0, 1)
        target.ry(params[0], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index:(param_index + 3)]), [source, sink])
            param_index += 1
        return qc

    fm = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
    circuit = QuantumCircuit(8)
    circuit.compose(fm, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    return circuit, fm, ansatz


def test_more_repro_matches_original_qiskit_circuit(bundle):
    import warnings

    from qiskit.quantum_info import SparsePauliOp, Statevector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit, fm, ansatz = _more_qiskit_circuit()
    weights = list(ansatz.parameters)          # EstimatorQNN weight order (sorted)
    assert len(weights) == 91
    names = [p.vector.name for p in weights]
    assert names[:24] == ["c2"] * 24 and names[-48:] == ["с1"] * 48
    rng = np.random.default_rng(5)
    model = build_model("MORE-REPRO")
    eng = BatchedStatevector(model)
    obs = [SparsePauliOp.from_list([(p + "I" * 7, 1)]) for p in "XYZ"]
    for _ in range(3):
        theta = rng.uniform(0, 2 * np.pi, 91)
        x = rng.uniform(0, np.pi, 8)
        x_more = 2.0 * x                         # MORE scales features to [0, 2pi]
        binding = {**dict(zip(fm.parameters, x_more)), **dict(zip(weights, theta))}
        sv = Statevector(circuit.assign_parameters(binding))
        ref = np.array([sv.expectation_value(o).real for o in obs])
        ours = eng.outputs(x[None, :], theta)[0]
        assert np.max(np.abs(ours - ref)) < 1e-12

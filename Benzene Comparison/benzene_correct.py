# Comparative Simulation of Benzene with VQE and ADAPT-VQE

# --- Import Libraries ---
from qiskit import Aer, execute
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit.algorithms.minimum_eigensolvers import AdaptVQE
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# --- 1. Molecule Setup ---
print("\nSetting up benzene molecule with active space...")

driver = PySCFDriver(
    atom="""C  0.0000  1.4027  0.0000; 
           H  0.0000  2.4903  0.0000; 
           C -1.2148  0.7014  0.0000; 
           H -2.1567  1.2451  0.0000; 
           C -1.2148 -0.7014  0.0000; 
           H -2.1567 -1.2451  0.0000; 
           C  0.0000 -1.4027  0.0000; 
           H  0.0000 -2.4903  0.0000; 
           C  1.2148 -0.7014  0.0000; 
           H  2.1567 -1.2451  0.0000; 
           C  1.2148  0.7014  0.0000; 
           H  2.1567  1.2451  0.0000""",
    basis="sto3g"
)

# Get electronic structure problem from the driver
print("\nRunning PySCF driver to compute integrals...")
start_time = time.time()
molecule = driver.run()
print(f"PySCF calculation completed in {time.time() - start_time:.2f} seconds")
print(f"Number of orbitals: {molecule.num_spatial_orbitals}")
print(f"Number of electrons: {molecule.num_particles}")

# Set up the Active Space
print("\nApplying Active Space Transformation (6e, 6o)...")
transformer = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=6)
molecule = transformer.transform(molecule)
print(f"Active space transformation complete.")
print(f"Number of active orbitals: {molecule.num_spatial_orbitals}")
print(f"Number of active electrons: {molecule.num_particles}")

# Map to qubit operator
print("\nMapping to qubit operator using Jordan-Wigner...")
mapper = JordanWignerMapper()
fermionic_op = molecule.hamiltonian.second_q_op()
qubit_op = mapper.map(fermionic_op)
print(f"Qubit operator created with {qubit_op.num_qubits} qubits")
print(f"Number of Pauli terms: {len(qubit_op)}")

# Get the HF energy
hf_energy = molecule.reference_energy
print(f"Hartree-Fock energy: {hf_energy:.6f} Hartree")
print(
    f"Nuclear repulsion energy: {molecule.nuclear_repulsion_energy:.6f} Hartree")

# Backend for simulation
backend = Aer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(backend)

# Define a callback to monitor progress
optimization_progress = []
pbar = None


def store_intermediate_result(eval_count, parameters, mean, std):
    global pbar
    energy = float(mean) + molecule.nuclear_repulsion_energy
    optimization_progress.append((eval_count, energy))
    if pbar is not None:
        pbar.update(1)
        pbar.set_description(f"Energy: {energy:.6f} Hartree")
    return


# --- 2. VQE Baseline Implementation ---
print("\nSetting up VQE...")

num_particles = molecule.num_particles
num_spatial_orbitals = molecule.num_spatial_orbitals
print(
    f"Using {num_spatial_orbitals} spatial orbitals with {num_particles} particles")

hf_initial_state = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper)
print(
    f"Hartree-Fock initial state prepared with {hf_initial_state.num_qubits} qubits")

print("Building UCCSD ansatz...")
uccsd_ansatz = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=hf_initial_state)
print(f"UCCSD ansatz created with {len(uccsd_ansatz.parameters)} parameters")

# Set the maximum number of iterations for SLSQP
max_iterations = 1000
optimizer_vqe = SLSQP(maxiter=max_iterations)
print(f"Using SLSQP optimizer with maximum {max_iterations} iterations")

vqe = VQE(ansatz=uccsd_ansatz,
          optimizer=optimizer_vqe,
          quantum_instance=quantum_instance,
          callback=store_intermediate_result)

print("\nRunning VQE optimization (this may take some time)...")
pbar = tqdm(total=max_iterations)
optimization_progress = []
start_time = time.time()
vqe_result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
end_time = time.time()
if pbar is not None:
    pbar.close()
    pbar = None

vqe_energy = vqe_result.eigenvalue.real + molecule.nuclear_repulsion_energy
print(f"VQE completed in {end_time - start_time:.2f} seconds")
print(f"Number of optimizer iterations: {vqe_result.optimizer_evals}")
print(f"VQE Ground-State Energy: {vqe_energy:.6f} Hartree")
print(f"Energy improvement over HF: {hf_energy - vqe_energy:.6f} Hartree")

# Plot optimization progress
if optimization_progress:
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in optimization_progress], [x[1]
             for x in optimization_progress], 'o-')
    plt.axhline(y=hf_energy, color='r', linestyle='--', label='HF Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.title('VQE Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('vqe_optimization_progress.png')
    plt.show()

# --- 3. ADAPT-VQE Implementation ---
print("\nSetting up ADAPT-VQE...")

# Set the maximum number of iterations for COBYLA
adapt_max_iterations = 500
optimizer_adapt = COBYLA(maxiter=adapt_max_iterations)
print(
    f"Using COBYLA optimizer with maximum {adapt_max_iterations} iterations per operator")

print("Building ADAPT-VQE with UCCSD operator pool...")
# Create operator pool for ADAPT-VQE
operator_pool = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper).operators

adapt_vqe = AdaptVQE(
    optimizer=optimizer_adapt,
    quantum_instance=quantum_instance,
    operator_pool=operator_pool,
)
print(
    f"ADAPT-VQE initialized with {len(adapt_vqe.operator_pool)} operators in pool")

print("\nRunning ADAPT-VQE optimization (this may take longer than standard VQE)...")
start_time = time.time()
adapt_result = adapt_vqe.compute_minimum_eigenvalue(operator=qubit_op)
end_time = time.time()

adapt_energy = adapt_result.eigenvalue.real + molecule.nuclear_repulsion_energy
print(f"ADAPT-VQE completed in {end_time - start_time:.2f} seconds")
print(f"Final circuit depth: {adapt_vqe.get_optimal_circuit().depth()}")
print(f"Number of parameters used: {len(adapt_vqe.get_optimal_cost_vector())}")
print(f"ADAPT-VQE Ground-State Energy: {adapt_energy:.6f} Hartree")
print(f"Energy improvement over HF: {hf_energy - adapt_energy:.6f} Hartree")
print(f"Energy improvement over VQE: {vqe_energy - adapt_energy:.6f} Hartree")

# --- 4. Analysis and Comparison ---
print("\nGenerating final comparison of all methods...")

energies = {
    "Hartree-Fock": hf_energy,
    "VQE": vqe_energy,
    "ADAPT-VQE": adapt_energy
}

labels = list(energies.keys())
values = list(energies.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.ylabel("Energy (Hartree)")
plt.title("Ground-State Energy Comparison")
plt.grid(True)
plt.savefig('energy_comparison.png')
plt.show()

print("\nEnergy Results:")
for name, energy in energies.items():
    print(f"{name:12s}: {energy:.6f} Hartree")

# Calculate correlation energies
print("\nCorrelation Energy Recovery:")
vqe_corr = hf_energy - vqe_energy
adapt_corr = hf_energy - adapt_energy
print(f"VQE correlation energy:     {vqe_corr:.6f} Hartree")
print(f"ADAPT-VQE correlation energy: {adapt_corr:.6f} Hartree")
if adapt_corr > 0:
    print(
        f"ADAPT-VQE improvement over VQE: {(adapt_corr - vqe_corr) / vqe_corr * 100:.2f}%")

# --- 5. Discussion: Scalability ---
print("""
Scalability Discussion:
- For larger molecules (drug discovery targets), simple UCCSD ansatz becomes intractable.
- ADAPT-VQE offers flexible circuit depths but optimization overhead grows.
- Need: smarter ansatze (e.g., qubit-ADAPT, hardware-efficient pools), error mitigation (zero-noise extrapolation, VQE + PEC).
- Larger active spaces (e.g., 20+ qubits) need quantum hardware-aware compilation.
- Preprocessing: active space selection, orbital optimization become critical.
""")

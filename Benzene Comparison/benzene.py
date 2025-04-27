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
molecule = driver.run()

# Set up the Active Space
transformer = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=6)
molecule = transformer.transform(molecule)

# Map to qubit operator
mapper = JordanWignerMapper()
fermionic_op = molecule.hamiltonian.second_q_op()
qubit_op = mapper.map(fermionic_op)

# Get the HF energy
hf_energy = molecule.reference_energy

# Backend for simulation
backend = Aer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(backend)

# --- 2. VQE Baseline Implementation ---
print("\nRunning VQE...")

num_particles = molecule.num_particles
num_spatial_orbitals = molecule.num_spatial_orbitals

hf_initial_state = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper)

uccsd_ansatz = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=hf_initial_state)

optimizer_vqe = SLSQP(maxiter=1000)

vqe = VQE(ansatz=uccsd_ansatz, optimizer=optimizer_vqe,
          quantum_instance=quantum_instance)
vqe_result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
vqe_energy = vqe_result.eigenvalue.real + molecule.nuclear_repulsion_energy

print(f"VQE Ground-State Energy: {vqe_energy:.6f} Hartree")

# --- 3. ADAPT-VQE Implementation ---
print("\nRunning ADAPT-VQE...")

optimizer_adapt = COBYLA(maxiter=500)

adapt_vqe = AdaptVQE(
    ansatz=None,
    optimizer=optimizer_adapt,
    quantum_instance=quantum_instance,
    operator_pool=UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper).operators,
)

adapt_result = adapt_vqe.compute_minimum_eigenvalue(operator=qubit_op)
adapt_energy = adapt_result.eigenvalue.real + molecule.nuclear_repulsion_energy

print(f"ADAPT-VQE Ground-State Energy: {adapt_energy:.6f} Hartree")

# --- 4. Analysis and Comparison ---

energies = {
    "Hartree-Fock": hf_energy,
    "VQE": vqe_energy,
    "ADAPT-VQE": adapt_energy
}

labels = list(energies.keys())
values = list(energies.values())

plt.figure()
plt.bar(labels, values)
plt.ylabel("Energy (Hartree)")
plt.title("Ground-State Energy Comparison")
plt.grid(True)
plt.show()

print("\nEnergy Results:")
for name, energy in energies.items():
    print(f"{name:12s}: {energy:.6f} Hartree")

# --- 5. Discussion: Scalability ---
print("""
Scalability Discussion:
- For larger molecules (drug discovery targets), simple UCCSD ansatz becomes intractable.
- ADAPT-VQE offers flexible circuit depths but optimization overhead grows.
- Need: smarter ansatze (e.g., qubit-ADAPT, hardware-efficient pools), error mitigation (zero-noise extrapolation, VQE + PEC).
- Larger active spaces (e.g., 20+ qubits) need quantum hardware-aware compilation.
- Preprocessing: active space selection, orbital optimization become critical.
""")

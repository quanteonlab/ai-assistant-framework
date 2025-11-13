# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 30)

**Starting Chapter:** Chapter 12 Quantum Computing G. He Coauthor. 12.3.1 Physics Exercise Two Entangled Dipoles

---

#### Dirac Notation in Quantum Mechanics
Background context: In quantum mechanics, states are represented using Dirac's notation. The ket $| \psi \rangle $ denotes a state vector in an abstract Hilbert space, while the bra$\langle x |$ is its dual adjoint or covector space representation. The wave function $\psi(x)$ can be obtained from the inner product of the bra and ket: 
$$\psi(x) = \langle x | \psi \rangle.$$

This inner product provides a projection of the state vector onto the basis vectors.

:p What is Dirac notation, and how does it represent quantum states?
??x
Dirac notation uses $| \psi \rangle $ to denote a state vector in an abstract Hilbert space. The corresponding dual adjoint or covector space representation is given by the bra$\langle x |$. The wave function $\psi(x)$, which gives the probability amplitude of finding the state at position $ x$, can be obtained via the inner product:
$$\psi(x) = \langle x | \psi \rangle.$$

This represents a projection of the state vector onto the basis vectors. 
??x

---

#### Qubits and Quantum Gates
Background context: In quantum computing, qubits are fundamental units of information that can exist in multiple states simultaneously (superposition). The state of $n $ qubits is represented as a vector in a Hilbert space with dimension$2^n$. Quantum gates manipulate these states. Commonly used gates include the Pauli-X gate (bit flip), Hadamard gate, and CNOT gate.

:p What are qubits, and how do they differ from classical bits?
??x
Qubits are quantum mechanical analogs of classical bits that can exist in multiple states simultaneously due to superposition. Unlike a classical bit which is either 0 or 1, a qubit can be represented as:
$$| \psi \rangle = a |0\rangle + b |1\rangle,$$where $ a $ and $ b$ are complex numbers representing the probability amplitudes.

In contrast to classical bits, where operations like NOT (X) simply flip 0 to 1 or 1 to 0, quantum gates can perform more sophisticated transformations such as the Hadamard gate which puts a qubit into a superposition state:
$$H | \psi \rangle = H(a |0\rangle + b |1\rangle) = \frac{a+b}{\sqrt{2}}|0\rangle + \frac{a-b}{\sqrt{2}}|1\rangle.$$??x

---

#### Operators and Inner Products
Background context: Operators in Dirac notation, such as $O = |\phi\rangle \langle \psi|$, are represented by matrices. The inner product of two states is denoted as:
$$\langle \phi | \psi \rangle.$$

The scalar or inner product between the states $|\phi\rangle $ and$|\psi\rangle$ is given by:
$$\langle \phi | \psi \rangle = (\phi, \psi) = \langle \psi | \phi \rangle^*,$$where $*$ denotes complex conjugation.

:p What are the properties of operators and inner products in Dirac notation?
??x
Operators in Dirac notation are represented as matrices. For example:
$$O = |\phi\rangle \langle \psi| = [a b; c d] \begin{bmatrix} x \\ y \end{bmatrix},$$where $|\phi\rangle $ and$|\psi\rangle$ are vectors.

The inner product between two states is denoted as:
$$\langle \phi | \psi \rangle.$$

Properties include:

- The scalar or inner product of the states $|\phi\rangle $ and$|\psi\rangle$:
  $$\langle \phi | \psi \rangle = (\phi, \psi) = \langle \psi | \phi \rangle^*,$$where the asterisk denotes complex conjugation.

- An operator like $O$ changes one state into another:
$$O|\psi\rangle = |\phi\rangle.$$??x

---

#### Example of a Simple Quantum Program
Background context: Here is an example using Python and Cirq to create a simple quantum circuit with two qubits. We will apply the Hadamard gate to both qubits, followed by a CNOT gate.

:p How can we write a simple quantum program in Python using Cirq?
??x
Here's a simple quantum program using Python and Cirq:

```python
import cirq

# Create a Quantum Circuit with two qubits.
q = cirq.LineQubit.range(2)
circuit = cirq.Circuit()

# Apply Hadamard gates to both qubits.
circuit.append([cirq.H(q[0]), cirq.H(q[1])])

# Add a CNOT gate between the first and second qubits.
circuit.append(cirq.CNOT(q[0], q[1]))

print("Circuit:")
print(circuit)
```

This circuit prepares both qubits in a superposition state using Hadamard gates and then applies a controlled-not (CNOT) operation to entangle them.

??x

---

#### IBM Quantum Computer Example
Background context: To demonstrate the application of quantum computing, let's use the physical IBM Quantum Computer. This involves executing circuits on real hardware and obtaining results.

:p How can we execute a quantum program using an IBM Quantum Computer?
??x
To execute a quantum program using an IBM Quantum Computer:

1. **Create and compile your circuit**: Ensure it fits within the constraints of the available hardware.
2. **Upload to IBM Q Experience**: Use the IBM Q Experience platform or API.

Example code snippet in Python (using qiskit):

```python
from qiskit import QuantumCircuit, transpile, Aer, execute

# Create a quantum circuit with 2 qubits and 1 classical bit for measurement.
qc = QuantumCircuit(2, 2)

# Apply Hadamard gates to both qubits and a CNOT gate.
qc.h([0, 1])
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

print("Quantum Circuit:")
print(qc)

# Compile the circuit for the target backend
compiled_circuit = transpile(qc, backend=Aer.get_backend('qasm_simulator'), optimization_level=3)

# Execute the compiled circuit
job = execute(compiled_circuit, Aer.get_backend('qasm_simulator'), shots=1024)
result = job.result()

print("Result:")
print(result.get_counts(qc))
```

This code creates a simple quantum circuit, compiles it for execution on the QASM simulator (for demonstration purposes), and measures the results.

??x

#### Qubits and Quantum States
In quantum computing, information is stored using qubits, which are quantum bits. Unlike classical bits that can be either 0 or 1, a single qubit state is expressed as a linear combination of two basis states $|0⟩$ and $|1⟩$:
$$|𝜓⟩ = u|0⟩ + v|1⟩$$where $ u $ and $ v$ are complex numbers satisfying the normalization condition:
$$|u|^2 + |v|^2 = 1.$$

The state can also be represented geometrically on a Bloch sphere, where the angle $\theta $ and phase$\phi$ determine its position.
:p What is a qubit and how is it different from a classical bit?
??x
A qubit is a quantum mechanical system that stores information in superposition states of $|0⟩$ and $|1⟩$, allowing for complex combinations represented by the equation:
$$|𝜓⟩ = u|0⟩ + v|1⟩$$where $ u $ and $ v$ are complex numbers. This is different from a classical bit, which can only be 0 or 1.
x??

---
#### Bloch Sphere Representation
The state of a qubit can be represented geometrically using the Bloch sphere, where:
$$|𝜓⟩ = \cos(\frac{\theta}{2})|0⟩ + e^{i\phi}\sin(\frac{\theta}{2})|1⟩$$with $\theta \in [0, \pi]$ and $\phi \in [0, 2\pi)$.

This representation shows that a pure $|0⟩$ state lies on the $+z$ axis and a pure $|1⟩$ state on the $-z$ axis.
:p How can the qubit state be represented using the Bloch sphere?
??x
The qubit state can be represented on the Bloch sphere with:
$$|𝜓⟩ = \cos(\frac{\theta}{2})|0⟩ + e^{i\phi}\sin(\frac{\theta}{2})|1⟩$$where $\theta $ is the polar angle and$\phi$ is the azimuthal angle. This geometric representation helps visualize the state of a qubit.
x??

---
#### Composite Qubits
When combining two qubits, their states are represented in an expanded Hilbert space created by the tensor product:
$$H_{AB} = H_A \otimes H_B$$

The combined state is given by:
$$|\psi_{AB}⟩ = |\psi_A⟩ \otimes |\psi_B⟩.$$

For example, if $|\psi_A⟩ = u_1|0⟩ + v_1|1⟩$ and $|\psi_B⟩ = u_2|0⟩ + v_2|1⟩$, then:
$$|\psi_{AB}⟩ = (u_1|0⟩ + v_1|1⟩) \otimes (u_2|0⟩ + v_2|1⟩)$$which expands to a four-dimensional state vector.

:p How are multiple qubits combined in quantum computing?
??x
Multiple qubits can be combined using the tensor product of their individual Hilbert spaces. For instance, if:
$$|\psi_A⟩ = u_1|0⟩ + v_1|1⟩$$and$$|\psi_B⟩ = u_2|0⟩ + v_2|1⟩$$then their combined state is:
$$|\psi_{AB}⟩ = (u_1|0⟩ + v_1|1⟩) \otimes (u_2|0⟩ + v_2|1⟩)$$which expands to:
$$|\psi_{AB}⟩ = u_1u_2|00⟩ + u_1v_2|01⟩ + v_1u_2|10⟩ + v_1v_2|11⟩.$$

This results in a four-dimensional state vector.
x??

---

#### Direct Product of Vectors and States
Background context explaining how vectors and states can be combined using direct products. This is crucial for understanding composite quantum systems, specifically two-qubit systems.

:p What are the basis vectors for a two-qubit system?
??x
The basis vectors for a two-qubit system are $|0A⟩$,$|1A⟩$,$|0B⟩$, and $|1B⟩$. These can be represented as:
$$|0A⟩ = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1A⟩ = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
$$|0B⟩ = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1B⟩ = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

These vectors can form the basis for a 4-dimensional Hilbert space, such as $|Ψ⟩ = [a \; b] ⊗ [c \; d]$.

??x
The answer with detailed explanations.
```java
// Example of creating two-qubit states in Java using arrays to represent vectors
public class QuantumVector {
    public static double[][] getBasisVectors() {
        return new double[][]{
            {1, 0}, // |0A⟩
            {0, 1}, // |1A⟩
            {1, 0}, // |0B⟩
            {0, 1}  // |1B⟩
        };
    }
}
```
x??

---

#### Separable and Entangled States
Background context explaining the difference between separable and entangled states. A state is separable if it can be expressed as a direct product of individual states; otherwise, it is entangled.

:p How do you prove that the Bell states are entangled using the definition of separability?
??x
To prove that the Bell states are entangled, we use the fact that a state is separable if and only if $wz = xy$. The Bell states are:
$$|β00⟩ = \frac{1}{\sqrt{2}}(|00⟩ + |11⟩)$$
$$|β01⟩ = \frac{1}{\sqrt{2}}(|01⟩ + |10⟩)$$
$$|β10⟩ = \frac{1}{\sqrt{2}}(|00⟩ - |11⟩)$$
$$|β11⟩ = \frac{1}{\sqrt{2}}(|01⟩ - |10⟩)$$

These states cannot be written as a direct product of individual qubit states, indicating they are entangled.

??x
The answer with detailed explanations.
```java
// Example checking separability in Java using the definition
public class EntanglementCheck {
    public static boolean isEntangled(String state) {
        if (state.equals("β00") || state.equals("β10")) {
            return true; // These are known to be entangled
        }
        return false;
    }
}
```
x??

---

#### Density Matrix and Entanglement
Background context explaining the density matrix, which is used to describe quantum states without resorting to wave functions. It's particularly useful for ensembles of pure states.

:p What is a density matrix and how is it defined?
??x
A density matrix $\rho$ describes the state of an ensemble of pure states. It is defined as:
$$\rho = \sum_i p_i |\psi_i⟩⟨\psi_i|$$where $ p_i $ is the probability that the pure state $|\psi_i⟩$ is present in the ensemble.

:p Show how separability can be checked using a density matrix.
??x
To check if a system is separable, we examine its density matrix. A 4-dimensional system (two qubits) is separable if and only if:
$$\rho_{12} = \rho_A \otimes \rho_B$$where $\rho_A $ and$\rho_B$ are the reduced density matrices for subsystems A and B.

For example, a state $|\psi⟩ = [a \; b] ⊗ [c \; d]$ would be separable if:
$$w z = x y$$??x
The answer with detailed explanations.
```java
// Example of calculating the density matrix in Java using matrices
public class DensityMatrix {
    public static double[][] calculateDensityMatrix(double[] vec1, double[] vec2) {
        int size = 4; // 2 qubits
        double[][] rhoAB = new double[size][size];
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                rhoAB[i][j] += vec1[i] * vec2[j]; // Direct product of vectors
            }
        }
        
        return rhoAB;
    }
}
```
x??

---

#### Two Entangled Dipoles and Hamiltonian
Background context explaining the interaction between two magnetic dipoles, represented by their Pauli matrices. The Hamiltonian for this system is given.

:p Show that the direct product of states forms a basis in $\mathbb{C}^4$.
??x
The basis vectors for the 4-dimensional Hilbert space formed by the direct product of two qubits are:
$$|00⟩ = [1, 0] ⊗ [1, 0],$$
$$|01⟩ = [1, 0] ⊗ [0, 1],$$
$$|10⟩ = [0, 1] ⊗ [1, 0],$$
$$|11⟩ = [0, 1] ⊗ [0, 1].$$

These states can be written in matrix form as:
$$|00⟩ = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad |01⟩ = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix},$$
$$|10⟩ = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \quad |11⟩ = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}.$$??x
The answer with detailed explanations.
```java
// Example of creating basis vectors for two-qubit states in Java using arrays to represent vectors
public class BasisVectors {
    public static double[][] getBasisVectors() {
        return new double[][]{
            {1, 0, 0, 0}, // |00⟩
            {0, 1, 0, 0}, // |01⟩
            {0, 0, 1, 0}, // |10⟩
            {0, 0, 0, 1}  // |11⟩
        };
    }
}
```
x??

---

---

#### Quantum Entanglement and Eigenstates
Background context: The provided text discusses eigenstates of a quantum system, specifically focusing on entangled states. The eigenstates mentioned are $\phi_3 $ and$\phi_4 $, where $\phi_3 = |00\rangle $ is a separable state and$\phi_4 = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ is an entangled state.

:p Identify which eigenstates are separable and which are entangled.
??x
The eigenstate $\phi_3 = |00\rangle $ is separable, while$\phi_4 = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ is entangled.

Explanation: A state is separable if it can be written as a product of individual states. Here,$|00\rangle $ is simply the tensor product of two qubits in the state$|0\rangle $. On the other hand, $\phi_4 = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ cannot be written as a simple tensor product and represents an entangled state.

??x
To confirm this, consider the form of the states:
```python
# Example in Python to show separability
from qiskit import QuantumCircuit

# Create a separable state
qc_separable = QuantumCircuit(2)
qc_separable.iden([0])
qc_separable.iden([1])

print(qc_separable)

# Create an entangled state
qc_entangled = QuantumCircuit(2)
qc_entangled.h(0)  # Apply Hadamard to the first qubit
qc_entangled.cx(0, 1)  # Apply CNOT with the first qubit as control

print(qc_entangled)

# Both circuits can be checked for separability using their circuit structure.
```
x??

---

#### Hamiltonian Matrix Evaluation
Background context: The text discusses evaluating a Hamiltonian matrix in a given basis of eigenstates. The Hamiltonian is expressed in matrix form and should ideally diagonalize if the states are correctly chosen.

:p Calculate the Hamiltonian matrix $H$ for the given eigenstates.
??x
To calculate the Hamiltonian matrix $H $, we need to evaluate the expectation values $\langle \phi_i | H | \phi_j \rangle$ for all pairs of basis states. The provided states are:

- $\phi_1 = |00\rangle $-$\phi_2 = |01\rangle $-$\phi_3 = |10\rangle $-$\phi_4 = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ Assuming $H$ is a simple matrix, we can represent it as:

```python
import numpy as np

# Define the Hamiltonian matrix H (example 2x2 for simplicity)
H = np.array([[1, 0], [0, -1]])

# Define the eigenstates in bra-ket notation
phi_1_bra = np.array([1, 0])
phi_2_bra = np.array([0, 1])
phi_3_bra = np.array([0, 0])  # This is not a full set of states for this example

# Calculate the matrix elements <phi_i|H|phi_j>
H_matrix = np.zeros((4, 4), dtype=complex)

for i in range(4):
    for j in range(4):
        H_ij = (np.conj(phi_1_bra[i]) * phi_1_bra[j] + 
                np.conj(phi_2_bra[i]) * phi_2_bra[j] +
                np.conj(phi_3_bra[i]) * phi_3_bra[j])
        
        H_matrix[i, j] = sum(H[k, l] for k in range(2) for l in range(2))

H_matrix
```
x??

---

#### Controlled NOT (CNOT) Gate Effect
Background context: The CNOT gate is a fundamental two-qubit quantum gate that flips the target qubit if the control qubit is $|1\rangle$.

:p Determine the effect of the CNOT gate on the states $|10\rangle, |01\rangle, |00\rangle,$ and $|11\rangle$.
??x
The CNOT gate operates as follows:
- If the control qubit is 0, it does not change the target qubit.
- If the control qubit is 1, it flips the target qubit.

Let's calculate the effect of CNOT on each state:

- $|10\rangle $: Control qubit is 1, so target qubit is flipped. Result: $\text{CNOT} |10\rangle = |11\rangle$.
- $|01\rangle $: Control qubit is 0, so target qubit remains unchanged. Result: $\text{CNOT} |01\rangle = |01\rangle$.
- $|00\rangle $: Control qubit is 0, so target qubit remains unchanged. Result: $\text{CNOT} |00\rangle = |00\rangle$.
- $|11\rangle $: Control qubit is 1, so target qubit is flipped. Result: $\text{CNOT} |11\rangle = |10\rangle$.

```python
from qiskit import QuantumCircuit

# Define the CNOT gate and apply it to each state
qc_cnot = QuantumCircuit(2)
qc_cnot.cnot([1, 0])

print(qc_cnot)

# Apply CNOT on different states
states = ['|10>', '|01>', '|00>', '|11>']
results = []

for state in states:
    if state == '|10>':
        qc = QuantumCircuit(2)
        qc.x([1])
    elif state == '|00>':
        qc = QuantumCircuit(2)
    else:
        qc = QuantumCircuit(2)
        qc.x([0])

    qc.cnot([1, 0])

    # Print the resulting state
    results.append(qc)

for result in results:
    print(result)
```
x??

---

#### Controlled Z (CZ) Gate Effect
Background context: The controlled-Z gate applies a phase shift of $-1 $ to the$|11\rangle$ state, leaving other states unchanged.

:p Verify the effect of the CZ gate on the states $|00\rangle, |01\rangle, |10\rangle,$ and $|11\rangle$.
??x
The controlled-Z (CZ) gate operates as follows:
- If the control qubit is 0, it does not change any state.
- If the control qubit is 1, it applies a phase shift of $-1$ to the target qubit.

Let's calculate the effect of CZ on each state:

- $|00\rangle $: Control qubit is 0, so no change. Result: $\text{CZ} |00\rangle = |00\rangle$.
- $|01\rangle $: Control qubit is 0, so no change. Result: $\text{CZ} |01\rangle = |01\rangle$.
- $|10\rangle $: Control qubit is 1, so apply a phase shift to the target qubit. Result: $\text{CZ} |10\rangle = -|10\rangle$.
- $|11\rangle $: Control qubit is 1, so apply a phase shift to the target qubit. Result: $\text{CZ} |11\rangle = -|11\rangle$.

```python
from qiskit import QuantumCircuit

# Define the CZ gate and apply it to each state
qc_cz = QuantumCircuit(2)
qc_cz.cz([0, 1])

print(qc_cz)

# Apply CZ on different states
states = ['|00>', '|01>', '|10>', '|11>']
results = []

for state in states:
    if state == '|10>' or state == '|11>':
        qc = QuantumCircuit(2)
        qc.z([1])
    else:
        qc = QuantumCircuit(2)

    qc.cz([0, 1])

    # Print the resulting state
    results.append(qc)

for result in results:
    print(result)
```
x??

---

#### Bell State Creation with Gates
Background context: The provided text describes creating entangled Bell states using quantum gates. Specifically, it mentions using a Hadamard gate followed by a CNOT gate to create the $\frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ state.

:p Explain how to create an entangled state $|\beta_{00}\rangle$ using the given quantum circuit.
??x
To create the entangled Bell state $|\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$, we can use a Hadamard gate followed by a CNOT gate. Here's how:

1. Start with the initial state $|00\rangle$.
2. Apply the Hadamard gate to the first qubit, transforming it into $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.
3. Use a CNOT gate where the first qubit is the control and the second qubit is the target.

This sequence of gates will produce the desired entangled state:

```python
from qiskit import QuantumCircuit, transpile

# Create a quantum circuit for creating the Bell state
qc = QuantumCircuit(2)
qc.h(0)  # Apply Hadamard to the first qubit
qc.cx(0, 1)  # Apply CNOT with the first qubit as control and second as target

print(qc)

# Transpile the circuit for better visualization (optional)
transpiled_circuit = transpile(qc, basis_gates=['u', 'cx'])
print(transpiled_circuit)
```
x??

#### Hadamard Gate Introduction
Background context: The Hadamard gate is a fundamental single-qubit gate used to create superposition states. It transforms eigenstates of the Z operator into eigenstates of the X operator. The Hadamard matrix, which performs this transformation, can be represented as:
$$H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$:p What is a Hadamard gate and what does it do?
??x
The Hadamard gate creates superposition states from base states. For example, applying the Hadamard gate to $|0\rangle$ results in:
$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

This state is a superposition of $|0\rangle $ and$|1\rangle$.
x??

---

#### Two Hadamard Gates Acting as Identity Operator
Background context: Applying the Hadamard gate twice on any qubit should return it to its original state, effectively acting as an identity operator. This can be demonstrated by applying two Hadamard gates in succession.

:p How do two consecutive Hadamard gates behave?
??x
Two consecutive Hadamard gates applied to a qubit act as the identity operator because each Hadamard gate transforms between eigenstates of Z and X, and applying it twice brings back the original state. For instance:
$$H(H|0\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \rightarrow H(\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)) = |0\rangle$$x??

---

#### X and Hadamard Gates
Background context: The combination of an X gate, which flips the state from $|0\rangle $ to$|1\rangle $, followed by a Hadamard gate creates an eigenstate of the X operator. This is because applying an X gate to $|0\rangle $ results in$|1\rangle$.

:p How do you create an eigenstate of the X operator using X and H gates?
??x
To create an eigenstate of the X operator, first apply an X gate to a qubit initially in state $|0\rangle$ and then apply a Hadamard gate. The resulting state is:
$$XH|0\rangle = X\left(\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)\right) = \frac{1}{\sqrt{2}}(|1\rangle - |0\rangle)$$

This is an eigenstate of the X operator.
x??

---

#### Measurement Operator with Cirq
Background context: A measurement operation in quantum computing provides a classical output representing the probability distribution of a particular measurement. Non-unitary operations, like measurements, are not considered gates.

:p How does a measurement operator work in Cirq?
??x
A measurement operator in Cirq is used to perform a measurement on a qubit and returns a classical bit as output. The result can be visualized using probability histograms. For example:
```python
import cirq
import matplotlib.pyplot as plt

circuit = cirq.Circuit()
a = cirq.NamedQubit('a')
circuit.append(cirq.X(a))
circuit.append(cirq.H(a))
circuit.append(cirq.measure(a, key='result'))

s = cirq.Simulator()
results = s.run(circuit, repetitions=1000)
print(results)
```
This code applies an X gate and a Hadamard gate to qubit `a`, then measures the state. The output is a series of 0's and 1's representing the measurement outcomes.
x??

---

#### CNOT Gate with Two Qubits
Background context: The controlled-NOT (CNOT) gate is a two-qubit gate where the target qubit is flipped if the control qubit is $|1\rangle$.

:p How does the CNOT gate operate?
??x
The CNOT gate operates such that it performs:
$$\text{CNOT}(q_0, q_1) = \begin{cases} (q_0, q_1), & \text{if } q_0 = 0 \\ (q_0, 1 - q_1), & \text{if } q_0 = 1 \end{cases}$$

For instance:
- If $q_0 = 0 $ and$q_1 = 0 $, the output is $(0, 0)$.
- If $q_0 = 1 $ and$q_1 = 0 $, the output is$(1, 1)$.

Circuit Example:
```python
import cirq

circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2)
circuit.append(cirq.X(q0))
circuit.append(cirq.Z(q1))
circuit.append(cirq.CNOT(q0, q1))

s = cirq.Simulator()
results = s.simulate(circuit)
print(results)
```
This code creates a CNOT gate between $q_0 $ and$q_1 $, flips$ q_0 $using an X gate, and applies Z to$ q_1 $. The output state is$(1, 1)$.
x??

---

#### Toffoli Gate with Three Qubits
Background context: The Toffoli or CCNOT gate is a three-qubit controlled-controlled-NOT gate. It inverts the third qubit if the first two qubits are both $|1\rangle$.

:p What does the Toffoli (CCNOT) gate do?
??x
The Toffoli gate performs:
$$\text{Toffoli}(q_0, q_1, q_2) = \begin{cases} (q_0, q_1, q_2), & \text{if } q_0 \neq 1 \text{ or } q_1 \neq 1 \\ (q_0, q_1, 1 - q_2), & \text{if } q_0 = 1 \text{ and } q_1 = 1 \end{cases}$$

For example:
- If $q_0 = 1 $, $ q_1 = 1 $, and$ q_2 = 0 $, the output is$(1, 1, 1)$.
- If any of $q_0 $ or$q_1 $ are not$1$, the state remains unchanged.

Circuit Example:
```python
import cirq

circuit = cirq.Circuit()
q0, q1, q2 = cirq.LineQubit.range(3)
circuit.append(cirq.X(q0))
circuit.append(cirq.Z(q2))
circuit.append(cirq.Toffoli(q0, q1, q2))

s = cirq.Simulator()
results = s.simulate(circuit)
print(results)
```
This code creates a Toffoli gate between $q_0 $, $ q_1 $, and$ q_2 $, flips$ q_0 $using an X gate, applies Z to$ q_2 $, and the output state is$(1, 0, 1)$.
x??

--- 

These flashcards cover the key concepts of quantum gates and operations using Cirq. Each card provides a clear explanation of the concept along with relevant code examples for better understanding.


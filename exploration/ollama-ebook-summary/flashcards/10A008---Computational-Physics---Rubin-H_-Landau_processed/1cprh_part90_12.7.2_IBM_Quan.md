# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 90)

**Starting Chapter:** 12.7.2 IBM Quantum Exercises

---

#### Quantum Computing Overview
Background context explaining quantum computing and its relevance. The IBM Quantum device is used to simulate a Bell state, which is a specific type of entangled qubit pair. The code snippet provided uses Qiskit to run a quantum circuit on the IBM Quantum device and measures the results.

:p What does the provided code snippet demonstrate in terms of quantum computing?
??x
The code demonstrates how to use Qiskit to execute a quantum circuit on the IBM Quantum device for measuring a Bell state. Specifically, it initializes a Bell state using qubits, runs the circuit with 8192 shots (measurements), and plots the histogram of the measurement results.

```python
# Import necessary libraries from Qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Define the Bell state circuit
bell_state = QuantumCircuit(2)
bell_state.h(0)  # Apply Hadamard gate to first qubit
bell_state.cx(0, 1)  # Apply CNOT gate with first qubit as control and second qubit as target

# Transpile the circuit for the IBM Quantum device (assuming 'device' is already defined)
tranpiled_circuit = transpiler(bell_state, backend=device)

# Run the circuit on the IBM Quantum device
job = device.run(tranpiled_circuit , shots=8192) 
job_monitor(job, interval=2)

# Get and plot the results
result = job.result()
counts = result.get_counts(bell_state)
plot_histogram(counts)
```

x??

---

#### Full Adder Circuit Overview
Background context on full adders in classical computing. The text describes how to implement a full adder using quantum circuits, specifically focusing on two qubit addition with the help of ancillary bits.

:p What is the purpose of the `adder_circuit` function provided in the code?
??x
The `adder_circuit` function creates a quantum circuit that implements a full adder for adding two single-bit binary numbers (x and y) using three qubits. The function initializes the necessary qubits, applies controlled operations to compute the sum and carry bit, and measures the results.

```python
def adder_circuit(x_in: int, y_in: int) -> QuantumCircuit:
    # Initialize a 5-qubit quantum circuit with 3 classical bits for measurement
    qc = QuantumCircuit(5, 3)
    
    # Convert input integers to binary strings and pad them to two bits each
    s = f"0{y_in:02b}{x_in:02b}"
    
    # Initialize the qubits with the binary inputs
    qc.initialize(s, [4, 3])
    
    # Apply the first Toffoli gate for carry propagation
    qc.ccx(0, 1, 4)
    
    # Apply a CNOT gate to copy the result into an ancillary bit
    qc.cx(0, 1)
    
    # Reset qubit 0 and apply another Toffoli and CNOT for further processing
    qc.reset(0)
    qc.ccx(1, 2, 0)
    
    return qc
```

x??

---

#### Quantum Circuit for Adding 01 + 10

:p What does the quantum circuit shown in Figure 12.8 do?
??x
The quantum circuit shown in Figure 12.8 performs a full adder operation to add two single-bit binary numbers (01 and 10), resulting in their sum. The circuit uses three qubits, where the first two represent the input bits x and y, and the third acts as an ancillary carry bit. It employs Toffoli (CCX) gates and CNOT gates to perform the addition logic.

x??

---

#### IBM Quantum Device Simulation
Background context on using IBM Quantum devices for simulation purposes. The text explains how to run a quantum circuit on the IBM Quantum device, measure the results, and visualize them with Qiskit's histogram plotting function.

:p How does the provided code snippet run the Bell state measurement on an IBM Quantum device?
??x
The provided code runs a quantum circuit designed to measure a Bell state on the IBM Quantum device. It initializes the necessary qubits for a Bell state, transpiles the circuit for the specific backend (IBM Quantum), executes the job with 8192 shots, monitors the job's progress, retrieves the results, and plots the histogram of the measurement outcomes.

```python
# Import necessary libraries from Qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Define the Bell state circuit
bell_state = QuantumCircuit(2)
bell_state.h(0)  # Apply Hadamard gate to first qubit
bell_state.cx(0, 1)  # Apply CNOT gate with first qubit as control and second qubit as target

# Transpile the circuit for the IBM Quantum device (assuming 'device' is already defined)
tranpiled_circuit = transpiler(bell_state, backend=device)

# Run the circuit on the IBM Quantum device
job = device.run(tranpiled_circuit , shots=8192) 
job_monitor(job, interval=2)

# Get and plot the results
result = job.result()
counts = result.get_counts(bell_state)
plot_histogram(counts)
```

x??

---

#### Quantum Fourier Transform Overview
Background context: The quantum Fourier transform (QFT) is a fundamental quantum algorithm that transforms the state of N qubits from the computational basis to a superposition of states. It is widely used in various quantum algorithms, such as Shor's algorithm for integer factorization and the Quantum Phase Estimation.

:p What is the quantum Fourier transform (QFT)?
??x
The QFT transforms an N-qubit state into a superposition of states, essentially performing a discrete Fourier transform on a quantum register. It is crucial in many quantum algorithms due to its ability to convert between the computational basis and the Fourier basis.
x??

---

#### CZ Gate Effects
Background context: The controlled-Z (CZ) gate is a two-qubit quantum gate that applies a phase shift of $-1 $ to the$|11\rangle$ state. It is often used in various quantum algorithms and circuits.

:p Prove that $CZ|00⟩=|00⟩$,$ CZ|01⟩=|01⟩$,$ CZ|10⟩=|10⟩$, and $ CZ|11⟩=−|11⟩$.
??x
The CZ gate applies a phase shift of $-1 $ only to the state$|11\rangle$. For other states, it leaves them unchanged. Specifically:
- For $|00⟩$: The CZ gate does not apply any operation.
  - Therefore, $CZ|00⟩=|00⟩$.
- For $|01⟩$: The control qubit is in the state $|0\rangle$, so the target qubit remains unchanged.
  - Therefore, $CZ|01⟩=|01⟩$.
- For $|10⟩$: The control qubit is in the state $|1\rangle$, but the target qubit is not part of the phase shift.
  - Therefore, $CZ|10⟩=|10⟩$.
- For $|11⟩$: Both qubits are in the state $|1\rangle $, so a phase shift of $-1$ is applied to this state.
  - Therefore,$CZ|11⟩=−|11⟩$.

This can be verified by considering the action of the CZ gate on each basis state. :
```python
# Pseudocode for verifying the effects of the CZ gate
def cz_effect(state):
    if state == "00":
        return "00"
    elif state == "01":
        return "01"
    elif state == "10":
        return "10"
    elif state == "11":
        return "-11"
```
x??

---

#### CNOT Gate Effects
Background context: The controlled-NOT (CNOT) gate is a two-qubit quantum gate that flips the target qubit if the control qubit is in the state $|1\rangle$. It is one of the most commonly used gates in quantum circuits.

:p Determine the effect of the CNOT gate on the states $|10⟩$,$|01⟩$,$|00⟩$, and $|11⟩$.
??x
The CNOT gate flips the target qubit if the control qubit is $|1\rangle$. Let's examine each state:
- For $|10⟩$: The control qubit is in the state $|1\rangle$, so the target qubit is flipped.
  - Therefore, $CNOT|10⟩=|11⟩$.
- For $|01⟩$: The control qubit is in the state $|0\rangle$, so no operation is performed on the target qubit.
  - Therefore, $CNOT|01⟩=|01⟩$.
- For $|00⟩$: The control qubit is in the state $|0\rangle$, so no operation is performed on the target qubit.
  - Therefore, $CNOT|00⟩=|00⟩$.
- For $|11⟩$: The control qubit is in the state $|1\rangle$, so the target qubit is flipped.
  - Therefore, $CNOT|11⟩=|10⟩$.

This can be verified by considering the action of the CNOT gate on each basis state. :
```python
# Pseudocode for verifying the effects of the CNOT gate
def cnot_effect(state):
    if state == "10":
        return "11"
    elif state == "01":
        return "01"
    elif state == "00":
        return "00"
    elif state == "11":
        return "10"
```
x??

---

#### Bell State Creation
Background context: A Bell state is a maximally entangled quantum state of two qubits. The Bell state $|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01⟩ - |10⟩)$ is one such example, and it can be created using a series of quantum gates.

:p Create a quantum circuit for creating the entangled Bell state $|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01⟩ - |10⟩)$.
??x
To create the Bell state $|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01⟩ - |10⟩)$, we can use a series of quantum gates as follows:
- Apply an X gate to the second qubit to get $|10\rangle $ from the initial$|00\rangle$ state.
- Apply a Hadamard (H) gate to the first qubit to put it in a superposition state.
- Apply a CNOT gate with the first qubit as control and the second qubit as target.

The circuit can be represented by the following steps:
1. Initialize:$|00\rangle $2. Apply H on q0:$\frac{1}{\sqrt{2}}(|00⟩ + |01⟩)$3. Apply X on q1:$\frac{1}{\sqrt{2}}(|01⟩ + |11⟩)$4. Apply CNOT (q0 as control, q1 as target):$\frac{1}{\sqrt{2}}(|01⟩ - |10⟩)$ This circuit creates the desired Bell state.
```python
# Pseudocode for creating the Bell state using quantum gates
def create_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply H on q0
    qc.x(1)  # Apply X on q1
    qc.cx(0, 1)  # Apply CNOT with q0 as control and q1 as target
```
x??

---

#### Half Adder Circuit
Background context: A half adder is a basic combinational logic circuit that takes two binary inputs (qubits in this case) and produces their sum and carry. The sum output is the XOR of the two input qubits, while the carry output is 1 if both inputs are 1.

:p Create a quantum circuit for a half-adder that adds the qubits $q0 $ and$q1 $, outputs the sum in $ y $, and sets the carry bit$ q2 = 1 $if$ q0 = q1 = 1 $, else$ q2 = 0$. Verify the addition for 1 + 1, 1 + 0, and 0 + 1.
??x
To create a half adder circuit in quantum computing, we need to:
- Perform an XOR operation on the inputs $q0 $ and$q1 $ to get the sum bit$y$.
- Use a CNOT gate with $q0 $ as control and$q2$ as target to set the carry bit.

The steps are as follows:
1. Initialize qubits:$|q0⟩|q1⟩|q2⟩ = |000⟩$2. Apply an XOR operation on $ q0$and $ q1$ using a CNOT gate.
3. Use a CNOT gate with $q0 $ as control and$q2$ as target to set the carry bit.

The circuit can be represented by:
```python
# Pseudocode for creating a half adder circuit
def half_adder_circuit():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)  # XOR (sum) operation
    qc.cx(0, 2)  # CNOT for carry bit

    return qc
```
To verify the addition:
- For $q0 = 1 $ and$q1 = 1 $: The sum is $0 $ and carry is$1$.
- For $q0 = 1 $ and$q1 = 0 $: The sum is$1 $ and carry is$0$.
- For $q0 = 0 $ and$q1 = 1 $: The sum is$1 $ and carry is$0$.

These can be verified by running the circuit with different input states. :
```python
# Example verification for q0=1, q1=1
qc = half_adder_circuit()
qc.draw(output='mpl')
```
x??

---

#### 1-Qubit Quantum Fourier Transform (QFT)
Background context: The QFT for a single qubit is computed using a simple formula involving the complex variable Z, where $Z = e^{-2\pi i/N}$ for $N=2^1 = 2$. This results in transformations that can be implemented with just one qubit and a few operations.
:p What does the QFT do to a single qubit state?
??x
The QFT transforms a single-qubit state into another form. Specifically, for a qubit $|q\rangle$, the transformation is given by:
$$QFT2|q⟩ = \frac{1}{\sqrt{2}}( |0\rangle + Z^{-q} |1\rangle ) $$where $ Z = e^{-i\pi/2}$ and represents a phase shift.

This means that for any input state, the output is a linear combination of basis states with appropriate phases.
??x
The QFT on a single qubit state maps it to another form. Given an input state $|q\rangle$, the output is:
$$\frac{1}{\sqrt{2}}( |0\rangle + e^{-i\pi q/2} |1\rangle )$$

This effectively performs a Hadamard transform followed by a phase shift, as illustrated in Qiskit using the Hadamard gate and the controlled phase gate.

Code example:
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.h(0)  # Apply Hadamard gate
qc.p(pi/2, 0)  # Apply a π/2 phase shift (controlled by the first bit)
```
x??

---

#### 2-Qubit Quantum Fourier Transform (QFT)
Background context: For two qubits, the QFT computes four components. The transformation involves complex exponentials and can be implemented with specific quantum gates in Qiskit.
:p How does the QFT work for two qubits?
??x
The QFT for two qubits transforms a state $|y\rangle $ into another form represented by$|Y\rangle$. Mathematically, it is expressed as:
$$|Y\rangle = \frac{1}{2} \sum_{k=0}^{3} \sum_{l=0}^{3} y_k Z^{-kl} |k\rangle$$

In Qiskit, this can be implemented using a series of Hadamard gates and controlled phase shift gates.
??x
The 2-qubit QFT transforms an input state $|y\rangle $ into another form represented by$|Y\rangle$:
$$|Y\rangle = \frac{1}{2} \left[ (y_0 Z^{0} + y_1 Z^{0} + y_2 Z^{0} + y_3 Z^{0}) |0\rangle \right.$$
$$+ (y_0 Z^{0} + y_1 Z^{-1} + y_2 Z^{-2} + y_3 Z^{-3}) |1\rangle$$
$$+ (y_0 Z^{0} + y_1 Z^{-2} + y_2 Z^{-4} + y_3 Z^{-6}) |2\rangle$$
$$+ \left. (y_0 Z^{0} + y_1 Z^{-3} + y_2 Z^{-6} + y_3 Z^{-9}) |3\rangle \right]$$

This can be implemented in Qiskit using the following code:
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)  # Apply Hadamard gate to first qubit
qc.cx(0, 1)  # Apply CNOT gate with control on first and target on second qubit
qc.p(pi/4, 1)  # Apply a π/4 phase shift on the second qubit
```
x??

---

#### n-Qubit Quantum Fourier Transform (QFT)
Background context: The QFT for $n $ qubits generalizes to transform an$n$-dimensional state. It involves direct products and complex exponentials, resulting in a transformation that can be applied using quantum gates.
:p How is the n-qubit QFT implemented?
??x
The n-qubit QFT transforms an input state $|k\rangle$ into another form represented by:
$$|Y\rangle = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} e^{-2\pi i k/N} |i\rangle$$

This can be implemented in Qiskit using a sequence of Hadamard gates and controlled phase shift gates. The implementation involves binary fraction notation for the exponents.

Code example:
```python
from qiskit import QuantumCircuit

def qft(circ, n):
    """Apply the QFT to the first n qubits in circ"""
    # Apply hadamard gates
    for i in range(n):
        circ.h(i)
        # Add controlled phase shift gates
        for k in range(i):
            circ.cu1(pi/2**(i-k), k, i)

# Example usage:
qc = QuantumCircuit(4)
qft(qc, 4)
```
x??

---

#### Oracle + Diffuser = Grover's Search Algorithm
Background context: In Grover’s search algorithm, an oracle is used to mark the target state and a diffuser is used to amplify the amplitude of the marked state. Together, they allow for faster search in unsorted databases.
:p What are the steps involved in Grover’s search algorithm?
??x
Grover's search algorithm involves the following key steps:
1. **Initialization**: Start with $n $ qubits initialized to$|0\rangle$.
2. **Superposition**: Apply Hadamard gates to create a superposition of all states.
3. **Oracle Application**: Use an oracle to mark the target state by flipping its sign.
4. **Diffuser Operation**: Apply a diffuser operator to amplify the amplitude of the marked state.

Mathematically, this involves transforming the initial uniform superposition:
$$|ψ\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} |k\rangle$$
into a form that allows for faster search by increasing the probability amplitude of the target state.

Code example (Grover oracle):
```python
from qiskit import QuantumCircuit

def grover_oracle(qc, index):
    """Create an Oracle circuit"""
    # Apply NOT gate to the target state
    qc.x(index)
    # Apply Hadamard gates
    for i in range(len(qc.qubits)):
        qc.h(i)
    # Apply controlled phase shift gates (negate the target state)
    for i in range(index):
        qc.ccx(i, index, index+1)  # Controlled-to-ancilla CNOTs
    # Reapply Hadamard gates to revert
    for i in range(len(qc.qubits)-1, -1, -1):
        qc.h(i)
    # Apply NOT gate back (revert target state)
    qc.x(index)

# Example usage:
qc = QuantumCircuit(4)
grover_oracle(qc, 3)
```
x??

---

#### Oracle Circuit for Grover’s Search Algorithm
Background context: The oracle circuit is a critical component of Grover's search algorithm. It marks the target state by applying a phase flip (controlled by the index of the marked state).
:p What does an Oracle circuit in Qiskit do?
??x
The Oracle circuit in Qiskit marks the target state by applying a controlled phase shift, effectively flipping the sign of the desired state in the superposition.

Code example:
```python
from qiskit import QuantumCircuit

def grover_oracle(qc, index):
    """Create an Oracle circuit"""
    # Apply NOT gate to the target state
    qc.x(index)
    # Apply Hadamard gates to prepare for phase flip
    for i in range(len(qc.qubits)):
        qc.h(i)
    # Apply controlled phase shift (negate the target state)
    for i in range(index):
        qc.ccx(i, index, index+1)  # Controlled-to-ancilla CNOTs
    # Reapply Hadamard gates to revert
    for i in range(len(qc.qubits)-1, -1, -1):
        qc.h(i)
    # Apply NOT gate back (revert target state)
    qc.x(index)

# Example usage:
qc = QuantumCircuit(4)
grover_oracle(qc, 3)  # Mark the third qubit as the target
```
x??

--- 

These flashcards cover the key concepts in Grover’s search algorithm and QFT for quantum computing. Each card provides a detailed explanation of the concept, relevant formulas, and example code with logic explanations.


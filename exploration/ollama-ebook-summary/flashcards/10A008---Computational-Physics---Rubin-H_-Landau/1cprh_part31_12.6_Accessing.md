# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 31)

**Starting Chapter:** 12.6 Accessing the IBM Quantum Computer

---

#### Half Adder Circuit Implementation

Background context: A half adder is a digital circuit that performs addition of two single binary digits. It produces a sum and a carry output. The formula for a half adder can be represented as:
- Sum (S) = q0 XOR q1
- Carry (C) = AND(q0, q1)

The objective here is to use quantum computing principles to implement this functionality using Cirq.

:p How does the provided code in C/CirqHalfAdder.py represent a half adder?
??x
The code uses three qubits to simulate the half adder logic. Here's a detailed explanation:
- `q0` and `q1` are input qubits representing the two bits being added.
- `q2` is used for storing the carry output.

Initially, X gates (`cirq.X`) flip the states of both qubits to 1, simulating inputs of '1'. Then a Toffoli gate (`cirq.Toffoli`) and a CNOT gate are applied:
- The Toffoli gate acts as an AND operation between `q0` and `q1`, setting `q2` to 1 if both qubits are 1.
- The subsequent CNOT gates ensure the correct output for the sum.

This setup effectively implements the half adder logic using quantum operations.

```python
# CirqHalfAdder.py: Cirq circuit for half adder
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)  # Create 3 qubits
circuit = cirq.Circuit()  # Build circuit

circuit.append(cirq.X(q0))  # Append X to q0 (set input to '1')
circuit.append(cirq.X(q1))  # Append X to q1 (set input to '1')

circuit.append(cirq.Toffoli(q0, q1, q2))  # Append Toffoli gate for AND operation
circuit.append(cirq.CNOT(q0, q1))  # Append CNOT to ensure correct output

print(circuit)  # Output circuit
```
x??

---

#### Full Adder Circuit Implementation

Background context: A full adder is an extension of the half adder that accounts for a carry-in bit. It takes three inputs (q0, q1, and Cin) and produces two outputs (sum and carry-out).

The objective here is to implement a full adder using Cirq.

:p How does the provided code in FullAdder.py represent a full adder?
??x
The code uses four qubits to simulate the full adder logic. Here's a detailed explanation:
- `q0` and `q1` are input qubits representing the two bits being added.
- `q2` is used for storing the sum output.
- `q3` is used for storing the carry-out.

The code initializes and appends gates to simulate the full adder logic:
- Initially, X gates (`cirq.X`) set inputs to '1'.
- A Toffoli gate acts as an AND operation between `q0`, `q1`, and `q2`.
- Another Toffoli gate is applied with different parameters.
- CNOT gates ensure the correct output for both sum and carry-out.

This setup effectively implements the full adder logic using quantum operations.

```python
# FullAdder.py: Cirq q0+q1 full adder program
import cirq

circuit = cirq.Circuit()  # Build circuit
q0, q1, q2, q3 = cirq.LineQubit.range(4)  # Create 4 qubits

circuit.append(cirq.X(q0))  # Append X to q0 (set input to '1')
circuit.append(cirq.X(q1))  # Append X to q1 (set input to '1')

circuit.append(cirq.Toffoli(q0, q1, q2))  # First Toffoli gate for AND operation
circuit.append(cirq.CNOT(q0, q1))  # CNOT to ensure correct output

circuit.append(cirq.Toffoli(q1, q2, q3))  # Second Toffoli gate for carry-out
```
x??

---

#### Verifying Additions with Half Adder

Background context: To verify the functionality of a half adder, we need to simulate and test various input scenarios.

The objective here is to use Cirq to simulate different addition cases (1+1, 1+0, 0+1).

:p How can you verify that the half adder works correctly for inputs 1+1, 1+0, and 0+1?
??x
To verify the correctness of the half adder implementation, we need to run simulations with different input configurations. Here's how it can be done:

For `q0 = 1` and `q1 = 1`, the expected outputs are sum = 0 (binary) and carry-out = 1.

For `q0 = 1` and `q1 = 0`, the expected outputs are sum = 1 (binary) and carry-out = 0.

For `q0 = 0` and `q1 = 1`, the expected outputs are also sum = 1 (binary) and carry-out = 0.

Here's how you can implement this in Cirq:

```python
# CirqHalfAdder.py: Cirq circuit for half adder
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)  # Create 3 qubits
circuit = cirq.Circuit()  # Build circuit

def simulate_addition(q0_input, q1_input):
    """Simulate and print the results for given inputs."""
    if q0_input == 1:
        circuit.append(cirq.X(q0))
    else:
        circuit.append(cirq.I(q0))  # Identity to keep it as '0'

    if q1_input == 1:
        circuit.append(cirq.X(q1))
    else:
        circuit.append(cirq.I(q1))

    simulator = cirq.Simulator()
    results = simulator.simulate(circuit)
    print(results)

# Test the half adder
simulate_addition(1, 1)  # Simulate 1 + 1
simulate_addition(1, 0)  # Simulate 1 + 0
simulate_addition(0, 1)  # Simulate 0 + 1
```

By running these simulations, you can check if the half adder produces the correct outputs for each test case.

x??

---

#### Quantum Circuit Operation Using Cirq
Background context: The provided Python code snippet demonstrates how to create and simulate a simple quantum circuit using the Cirq library. Specifically, it includes appending controlled-NOT (CNOT) gates between qubits \(q1\) and \(q2\), as well as between \(q0\) and \(q1\). This operation is common in quantum computing circuits.
:p What is the purpose of the code snippet provided?
??x
The code snippet aims to create a simple quantum circuit using Cirq, append specific gates (CNOT) to manipulate qubits, and simulate this circuit. The purpose here is to familiarize with basic operations on qubits and how they can be represented programmatically.
```python
# Import necessary libraries from Cirq
import cirq

# Define the qubits
q0, q1, q2 = cirq.LineQubit.range(3)

# Create a quantum circuit
circuit = cirq.Circuit()

# Append CNOT gates to the circuit
circuit.append(cirq.CNOT(q1, q2))  # CNOT gate between q1 and q2
circuit.append(cirq.CNOT(q0, q1))  # CNOT gate between q0 and q1

# Print the circuit for verification
print(circuit)

# Initialize a simulator
s = cirq.Simulator()

# Simulate the circuit
results = s.simulate(circuit)

# Output simulation results
print("Simulate the circuit: ")
print(results)
```
x??

#### IBM Quantum Access and Account Creation
Background context: The text explains how to access and use the IBM Quantum platform, including creating an account if necessary. It provides step-by-step instructions on navigating to the login page, using a cell phone’s QR reader for account creation, and accessing tutorials and programming tools.
:p What are the steps mentioned in the text for creating an IBM Quantum account?
??x
The steps mentioned include:
1. Go to `QUANTUM-COMPUTING.IBM.COM/LOGIN`.
2. Create an IBMid using a cell phone with its QR reader (this might not work for all countries).
3. Follow instructions on the page to create and authenticate your account.
4. Alternatively, use your Google or GitHub accounts to log into IBMQuantum if available.

These steps are necessary to gain access to IBM Quantum’s resources, including running quantum programs.
x??

#### IBM Quantum Composer Overview
Background context: The IBM Quantum Composer is a graphical tool for creating quantum circuits by dragging and dropping operators. It can be run on the IBM Quantum system or as a simulator. The text provides a screenshot of its layout and an example circuit used to generate a Bell state \(|\beta_{00}\rangle\).
:p What does the IBM Quantum Composer allow users to do?
??x
The IBM Quantum Composer allows users to graphically create quantum circuits by dragging and dropping operators, which can then be run on the IBM Quantum system or as a simulator. It provides an intuitive interface for designing and visualizing quantum circuits.

For example, using the Composer, you can:
- Create qubits and a classical 4-bit register.
- Drag and drop different quantum gates to design circuits.
- Run these circuits on real hardware or simulate them locally.

The provided screenshot (Figure 12.5) shows an example circuit used to generate the Bell state \(|\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)\), along with its generated computational basis states in both histogram and numerical forms.
x??

---

#### Quantum Circuit Basics
In quantum computing, a circuit is constructed using various quantum gates to manipulate qubits. The Hadamard gate \(H\) and controlled-NOT (CNOT) gate are commonly used for this purpose.

:p What is the sequence of operations performed on the qubits in the provided circuit?
??x
The sequence involves applying the Hadamard gate (\(H\)) to qubit 0, followed by a CNOT gate with qubit 0 as the control and qubit 1 as the target.
```python
# Applying H to q[0]
circuit.h(q[0])

# Then CNOT with q[0] as control and q[1] as target
circuit.cx(q[0], q[1])
```
x??

---

#### Reversed Dirac Notation for IBM Quantum
IBM Quantum uses a reversed order of qubits in their notation. Specifically, \(|qn-1…q1q0⟩\) is used instead of the conventional \(|q0q1…qn-1⟩\).

:p How does the IBM Quantum represent the state \(|01⟩\)?
??x
In IBM's notation, the state \(|01⟩\) corresponds to \(||q1=0⟩ \otimes ||q0=1⟩ = |01⟩\), which can be represented as:
\[ [1 0] \otimes [0 1] = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}. \]
x??

---

#### Qiskit Setup and Installation
Qiskit is an open-source SDK for quantum computing that can be used both on simulators and real quantum hardware, such as the IBM Quantum.

:p How do you set up a Qiskit environment in Anaconda?
??x
First, create a new conda environment with Python and necessary packages:
```bash
conda create --name qiskit python jupyter notebook
```
Activate the environment and install Qiskit along with its visualization tools:
```bash
conda activate qiskit
pip install qiskit[visualization] qiskit_ibm_provider
```
x??

---

#### Accessing IBM Quantum via API Token
To access the IBM Quantum service from a local machine, you need an API token to authenticate your external use.

:p How do you save and retrieve your API token in Qiskit?
??x
Save the API token by running:
```python
from qiskit_ibm_provider import IBMProvider

IBMProvider.save_account(api_token)
```
To load the token for future use, you can run:
```python
provider = IBMProvider()
```
x??

---

#### Using Qiskit for Circuit Simulation and Execution
Qiskit allows running circuits on both simulators and real quantum hardware. The following steps demonstrate how to execute a circuit using both methods.

:p How do you create and draw a simple 2-qubit quantum circuit in Qiskit?
??x
Create and configure the quantum circuit:
```python
from qiskit import QuantumCircuit

# Create a 2-qubit circuit
circuit = QuantumCircuit(2)
```
Apply gates to the circuit:
```python
# Apply H gate to qubit 0, then CNOT with control 0 and target 1
circuit.h(0)
circuit.cx(0, 1)

# Draw the circuit
circuit.draw('mpl')
```
x??

---

#### Comparing Simulation Results on Aer and IBM Quantum
The Qiskit Aer simulator can be used to run circuits and simulate their states. The results from the simulator are compared with those obtained from running the same circuit on an actual quantum device.

:p How do you execute a quantum circuit using the statevector_simulator backend in Qiskit?
??x
Set up the Aer backend for statevector simulation:
```python
from qiskit import Aer

backend = Aer.get_backend("statevector_simulator")
```
Run the job and retrieve the result:
```python
job = backend.run(circuit)
result = job.result()
statevector = result.get_statevector(circuit, decimals=3)
statevector.draw(output="latex")
plot_state_city(statevector)
```
x??

---

#### Executing on IBM Quantum Device
To run a circuit on an IBM Quantum device, you need to find the least busy device and transpile the circuit for execution.

:p How do you select the least busy quantum device using Qiskit?
??x
First, initialize the provider:
```python
from qiskit_ibm_provider import IBMProvider

provider = IBMProvider()
```
Find the least busy device with at least 3 qubits:
```python
device = least_busy(provider.backends(
    filters=lambda x: int(x.configuration().n_qubits) >= 3 and not x.configuration().simulator and x.status().operational is True))
print(f"Running on current least busy device: {device}")
```
x??

---

#### Measuring Qubits in the Circuit
Measuring qubits is crucial for obtaining classical information from a quantum circuit.

:p How do you add measurements to the quantum circuit?
??x
Add measurement operations to all qubits:
```python
circuit.measure_all()
```
Transpile the circuit for execution on the selected device:
```python
transpiled_circuit = transpile(circuit, device)
```
x??

---


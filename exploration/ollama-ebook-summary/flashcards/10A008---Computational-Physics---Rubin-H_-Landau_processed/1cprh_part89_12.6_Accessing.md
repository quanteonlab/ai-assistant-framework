# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 89)

**Starting Chapter:** 12.6 Accessing the IBM Quantum Computer

---

#### Half Adder Circuit Implementation

Background context: A half adder is a digital circuit that performs addition of two single-bit binary numbers. It outputs the sum and carry bit. The operation can be mathematically represented as:
- Sum = $q0 \oplus q1 $- Carry =$ q0 \land q1$

However, in this implementation using quantum circuits, we only focus on generating the sum output using a Toffoli gate followed by a CNOT gate. The carry bit is not explicitly generated but can be inferred from the state of the qubits.

:p What is the role of the Toffoli and CNOT gates in the half adder circuit?
??x
The Toffoli gate acts as a controlled-controlled-NOT (CCNOT) gate, which flips the target qubit if both control qubits are |1>. In this case, it effectively performs the AND operation on $q0 $ and$q1 $. The CNOT gate then performs an XOR operation between$ q0 $and$ q1$ to get the sum.

```python
# CirqHalfAdder.py: Cirq circuit for half adder
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)  # Create 3 qubits

circuit = cirq.Circuit()  # Build circuit
circuit.append(cirq.X(q0))  # Append X to q0
circuit.append(cirq.X(q1))  # Append X to q1
circuit.append(cirq.Toffoli(q0, q1, q2))  # Append Toffoli gate
circuit.append(cirq.CNOT(q0, q1))  # Append CNOT to q0 & q1

print(circuit)  # Output circuit

s = cirq.Simulator()  # Initialize Simulator
results = s.simulate(circuit)  # Run simulator
print(results)
```

x??

---

#### Full Adder Circuit Implementation

Background context: A full adder is an extension of a half adder that adds three bits (two inputs and one carry input). It produces a sum bit and a carry-out bit. The operation can be mathematically represented as:
- Sum =$q0 \oplus q1 \oplus Cin $- Carry Out =$(q0 \land q1) \lor (q0 \land Cin) \lor (q1 \land Cin)$ However, in this implementation using quantum circuits, we only focus on generating the sum and carry-out bits. The code provided uses Toffoli gates to implement these operations.

:p How does the Full Adder circuit use Toffoli gates?
??x
The Full Adder circuit uses a Toffoli gate for the first stage where $q0 $ and$q1 $ are used as control qubits to generate an intermediate carry-out bit. This is followed by another Toffoli gate using$q1$ and the intermediate carry-out bit as controls to generate the final carry-out bit.

```python
# FullAdder.py: Cirq full adder program
import cirq

q0, q1, q2, q3 = cirq.LineQubit.range(4)  # Create 4 qubits

circuit = cirq.Circuit()  # Build circuit
circuit.append(cirq.X(q0))  # Append X to q0
circuit.append(cirq.X(q1))  # Append X to q1
circuit.append(cirq.Toffoli(q0, q1, q2))  # Append Toffoli gate
circuit.append(cirq.CNOT(q0, q1))  # Append CNOT to q0 & q1
circuit.append(cirq.Toffoli(q1, q2, q3))  # Append Toffoli gate

print(circuit)  # Output circuit

s = cirq.Simulator()  # Initialize Simulator
results = s.simulate(circuit)  # Run simulator
print(results)
```

x??

---

#### Verifying Additions with Half and Full Adders

Background context: To verify the operations of half and full adders, we can input different combinations of $q0 $ and$q1 $(and optionally $ Cin$) and observe the output.

:p How would you verify the addition 1 + 1 using the provided half adder circuit?
??x
To verify the addition 1 + 1 with the half adder, we set both qubits $q0 $ and$q1 $ to |1>. The expected sum is |0> (since$1 \oplus 1 = 0 $) and a carry-out of |1> (since$1 \land 1 = 1$). Running the circuit with these inputs should reflect this.

```python
# CirqHalfAdder.py: Cirq circuit for half adder
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)  # Create 3 qubits

circuit = cirq.Circuit()  # Build circuit
circuit.append(cirq.X(q0))  # Set X on q0 to represent |1>
circuit.append(cirq.X(q1))  # Set X on q1 to represent |1>

s = cirq.Simulator()  # Initialize Simulator
results = s.simulate(circuit)  # Run simulator

print("Simulate the circuit output vector:", results.final_state_vector)
```

x??

---

#### Exercise: Adding Different Values with Half Adder

Background context: The exercise involves testing different inputs to ensure that the half adder correctly performs addition.

:p What values of $q0 $ and$q1$ would you test for adding 1 + 0 in the half adder circuit?
??x
For adding 1 + 0, we set $q0 $ to |1> and$q1 $ to |0>. The expected sum is |1> (since$1 \oplus 0 = 1$) and a carry-out of |0> (since no carry is generated).

```python
# CirqHalfAdder.py: Cirq circuit for half adder
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)  # Create 3 qubits

circuit = cirq.Circuit()  # Build circuit
circuit.append(cirq.X(q0))  # Set X on q0 to represent |1>
circuit.append(cirq.X(q1))  # Set X on q1 to represent |0>

s = cirq.Simulator()  # Initialize Simulator
results = s.simulate(circuit)  # Run simulator

print("Simulate the circuit output vector:", results.final_state_vector)
```

x??

---

#### Exercise: Adding Different Values with Full Adder

Background context: The exercise involves testing different inputs to ensure that the full adder correctly performs addition.

:p What values of $q0 $, $ q1 $, and$ Cin$ would you test for adding 1 + 0 with a carry-in of 1 in the full adder circuit?
??x
For adding 1 + 0 with a carry-in of 1, we set $q0 $ to |1>,$q1 $ to |0>, and$Cin $(corresponding to$ q2 $) to |1>. The expected sum is |0> (since$1 \oplus 0 \oplus 1 = 0 $) and a carry-out of |1> (since the intermediate carry is generated from$1 \land 0 = 0 $ but $0 \land 1 = 0$ and $1 \land 1 = 1$).

```python
# FullAdder.py: Cirq full adder program
import cirq

q0, q1, q2, q3 = cirq.LineQubit.range(4)  # Create 4 qubits

circuit = cirq.Circuit()  # Build circuit
circuit.append(cirq.X(q0))  # Set X on q0 to represent |1>
circuit.append(cirq.X(q1))  # Set X on q1 to represent |0>
circuit.append(cirq.X(q2))  # Set X on q2 (Cin) to represent |1>

s = cirq.Simulator()  # Initialize Simulator
results = s.simulate(circuit)  # Run simulator

print("Simulate the circuit output vector:", results.final_state_vector)
```

x??

---

#### CNOT Gate Implementation
Background context explaining the concept of a CNOT gate and its implementation. The CNOT (Controlled-NOT) gate is one of the most fundamental gates in quantum computing, which flips the state of the target qubit if the control qubit is in the $|1\rangle$ state.

The logic of a CNOT gate can be represented as follows:
- If the control qubit ($q_0 $) is $|1\rangle $, then the target qubit ($ q_1$) will flip its state.
- Otherwise, if the control qubit is $|0\rangle$, the target qubit remains unchanged.

In the provided code snippet:
```python
circuit.append(cirq.CNOT(q1, q2)) # Append CNOT to q1 , q2 
circuit.append(cirq.CNOT(q0, q1)) # Append CNOT to q0 , q1 
```
Two CNOT gates are being applied in the circuit. The first one targets $q_2 $ with$q_1 $ as control, and the second one targets$ q_1 $ with $q_0$ as control.

:p What is a CNOT gate and how does it work?
??x
A CNOT (Controlled-NOT) gate in quantum computing flips the state of the target qubit if the control qubit is in the $|1\rangle$ state. The gates are applied to the circuit by specifying which qubits act as controls and targets.

Here's how the logic works:
```python
# Example of applying CNOT gates using a hypothetical circuit object
circuit = ...  # Initialize your quantum circuit here

# Apply CNOT gate with q1 as control and q2 as target
circuit.append(cirq.CNOT(q1, q2)) 

# Apply another CNOT gate with q0 as control and q1 as target
circuit.append(cirq.CNOT(q0, q1))
```
x??

---

#### Quantum Circuit Simulation Output
Background context explaining the output of a quantum circuit simulation. The provided code initializes a simulator for running quantum circuits and outputs the result.

The snippet includes:
```python
s = cirq.Simulator() # Initialize Simulator 
print('Simulate the circuit: ') 
results = s.simulate(circuit) # Run simulator 
print(results)
```
After initializing the simulator, it runs the given circuit `circuit` and prints the simulation results.

:p What does the provided code simulate in quantum computing?
??x
The provided code initializes a quantum circuit simulator and simulates the execution of the specified quantum circuit to obtain its output state. Here's how the simulation is performed:

```python
# Initialize a Simulator object
s = cirq.Simulator()

# Print a message indicating that we are about to simulate the circuit
print('Simulate the circuit: ')

# Run the circuit through the simulator and store the results in 'results'
results = s.simulate(circuit)

# Output the results of the simulation, which include the output state vector
print(results)
```

The `s.simulate(circuit)` function runs the quantum circuit on a simulated quantum computer, giving us insight into its final state. This is useful for verifying the correctness of theoretical models and algorithms.

x??

---

#### IBM Quantum Composer Interface
Background context explaining the IBM Quantum Composer interface and how to access it. The IBM Quantum Composer allows users to graphically create quantum circuits by dragging and dropping gates onto a circuit diagram. 

The process involves:
1. Logging into the IBM Quantum platform.
2. Navigating through the dashboard until you find the 'Launch Composer' button, which opens the composer interface.

:p How do you access the IBM Quantum Composer?
??x
To access the IBM Quantum Composer, follow these steps:

1. **Login**: Go to [IBM Quantum](https://quantum-computing.ibm.com/login) and log in using your credentials.
2. **Create an Account**: If you don't have an account, use your cell phone with its QR reader to create a new IBMid account by clicking on 'Create an IBMid account'.
3. **Navigate to Composer**: Once logged in, navigate through the dashboard until you find the button labeled 'Launch Composer'. Clicking this will bring up the IBM Quantum Composer interface.

The composer is a graphical tool where you can drag and drop quantum gates to create circuits. Here's a simple example of using it:
```plaintext
H q[0] q[1]
```
This creates a Hadamard gate on $q[0]$ and another on $q[1]$.

x??

---

#### Generating Bell State Using IBM Quantum Composer
Background context explaining the generation of a specific quantum state, the Bell state ($\left| \beta_{00} \right\rangle$), using the IBM Quantum Composer. A Bell state is an entangled state which can be created using operations like Hadamard and CNOT gates.

The provided example shows how to generate the $\left| \beta_{00} \right\rangle$ state in the IBM Quantum Composer:
- It starts with a circuit creation process.
- Eliminates unused qubits and registers.
- Applies necessary quantum gates (Hadamard, CNOT) to achieve the desired state.

:p How is the Bell state ($\left| \beta_{00} \right\rangle$) generated using IBM Quantum Composer?
??x
The Bell state $\left| \beta_{00} \right\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ is generated using the IBM Quantum Composer by following these steps:

1. **Initialize Circuit**: Start a new circuit with four qubits and one classical register.
   ```plaintext
   File/New
   ```
   This gives you `q[0]`, `q[1]`, `q[2]`, and `q[3]` along with a 4-bit classical register.

2. **Remove Unnecessary Qubits**: Since only two qubits are needed for the Bell state, remove unused qubits:
   ```plaintext
   q[2], q[3] -> removed
   ```

3. **Apply Gates**:
   - Apply Hadamard (H) gate to `q[0]` to create superposition.
     ```plaintext
     H q[0]
     ```
   - Apply CNOT gates with appropriate controls and targets to entangle the qubits.
     ```plaintext
     CNOT q[0], q[1]
     ```

4. **Final State**: The resulting circuit will generate the Bell state $\left| \beta_{00} \right\rangle$.

Here's a step-by-step process in text format:
```plaintext
Hadamard gate on q[0]:
H q[0]

CNOT gates to entangle qubits:
CNOT q[0], q[1]
```

The output shows the circuit and the final state vector, which confirms that the Bell state has been successfully generated.

x??

---

#### Quantum Circuit Basics

Background context explaining the concept. The Hadamard gate $H $ and controlled-NOT (CNOT) gate are fundamental quantum gates used to manipulate qubits. In this example, a 2-qubit circuit is created where$q[0]$ undergoes a Hadamard transformation followed by a CNOT operation with $q[1]$ as the target.

:p What does the sequence of gates applied in the circuit do?
??x
The sequence first applies a Hadamard gate to qubit 0, which puts it into a superposition state. Then, a controlled-NOT (CNOT) gate is applied with qubit 0 as the control and qubit 1 as the target, entangling the two qubits.
x??

---

#### IBM Quantum's Reversed Dirac Notation

Background context explaining the concept. IBM Quantum uses a reversed Dirac notation where qubits are ordered from right to left.

:p What is the significance of the order in IBM Quantum’s notation?
??x
In IBM Quantum's notation, a state like $|01\rangle$ means that the first qubit (rightmost) is in state 0 and the second qubit (leftmost) is in state 1. This differs from traditional Dirac notation where states are ordered from left to right.
x??

---

#### Setting Up Qiskit Environment

Background context explaining the concept. The example shows how to set up a Qiskit environment on an Anaconda window in MS Windows.

:p How do you activate and install necessary packages for Qiskit?
??x
To activate and install the required packages, follow these steps:
1. Create a new conda environment with Python, Jupyter Notebook: 
   ```bash
   conda create --name qiskit python jupyter notebook
   ```
2. Activate the environment:
   ```bash
   conda activate qiskit
   ```
3. Install Qiskit and its visualization tools:
   ```bash
   pip install qiskit[visualization] qiskit_ibm_provider
   ```
x??

---

#### Quantum Circuit Creation with Qiskit

Background context explaining the concept. The example demonstrates creating a 2-qubit quantum circuit using Qiskit.

:p What are the steps to create and visualize a simple 2-qubit quantum circuit?
??x
The steps include:
1. Import necessary packages: 
   ```python
   from qiskit import QuantumCircuit
   ```
2. Create a Quantum Circuit with two qubits:
   ```python
   circuit = QuantumCircuit(2)
   ```
3. Apply the Hadamard gate to qubit 0 and CNOT gate with qubit 1 as target:
   ```python
   circuit.h(0) 
   circuit.cx(0, 1)
   ```
4. Draw the circuit using matplotlib:
   ```python
   circuit.draw('mpl')
   ```
x??

---

#### Running Quantum Circuits on Simulators and IBM Quantum

Background context explaining the concept. The example shows how to run a quantum circuit on both Qiskit’s statevector simulator and an IBM Quantum device.

:p How do you use Qiskit's Aer provider to run a quantum circuit on its simulator?
??x
To run the circuit on Qiskit's statevector simulator:
1. Import necessary packages:
   ```python
   from qiskit import Aer
   ```
2. Get the backend for running the simulation:
   ```python
   backend = Aer.get_backend('statevector_simulator')
   ```
3. Run the circuit and get the statevector result:
   ```python
   job = backend.run(circuit)
   result = job.result()
   statevector = result.get_statevector(circuit, decimals=3)
   ```
4. Visualize the statevector using Qiskit’s tools:
   ```python
   plot_state_city(statevector)
   ```
x??

---

#### Finding and Running on Least Busy IBM Quantum Device

Background context explaining the concept. The example illustrates how to find and run a quantum circuit on the least busy device available from an IBM Quantum account.

:p How do you determine and use the least busy IBM Quantum device?
??x
To determine and use the least busy IBM Quantum device:
1. Import necessary packages:
   ```python
   from qiskit_ibm_provider import IBMProvider
   ```
2. Get a provider instance:
   ```python
   provider = IBMProvider(instance="ibm-q/open/main")
   ```
3. Find the least busy backend with at least 3 qubits, not simulator and operational:
   ```python
   device = least_busy(provider.backends(filters=lambda x: int(x.configuration().n_qubits) >= 3 and not x.configuration().simulator and x.status().operational is True))
   ```
4. Print the chosen device:
   ```python
   print("Running on current least busy device:", device)
   ```
x??

---

#### Transpiling Quantum Circuits

Background context explaining the concept. The example shows how to transpile a quantum circuit for execution on an IBM Quantum device.

:p What is transpilation and why is it necessary in Qiskit?
??x
Transpilation, or translation, involves converting a high-level quantum circuit into a form that can be run on a specific backend (device) with constraints like the number of qubits and gates. It optimizes the circuit to fit within these constraints.

In Qiskit:
1. Import necessary packages:
   ```python
   from qiskit import transpile
   ```
2. Transpile the circuit for the chosen device:
   ```python
   transpiled_circuit = transpile(circuit, device)
   ```
3. Optionally, monitor the job status to ensure it is running:
   ```python
   job_monitor(transpiled_circuit)
   ```
x??


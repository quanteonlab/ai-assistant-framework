# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 32)

**Starting Chapter:** 12.7.2 IBM Quantum Exercises

---

#### Quantum Computing Introduction
Background context explaining the basics of quantum computing, including states like |0⟩ and |1⟩, and how they differ from classical bits. The IBM Quantum computer is mentioned as a real-world example.

:p What are some differences between qubits and classical bits?
??x
Qubits can exist in a superposition state represented by both |0⟩ and |1⟩ simultaneously, whereas classical bits can only be either 0 or 1 at any given time. This allows for parallel processing capabilities.
x??

---

#### Bell State Experiment
Background context on the experiment using Qiskit to create and measure a Bell state, a fundamental entangled quantum state.

:p What is a Bell state, and how was it measured in this example?
??x
A Bell state is an entangled pair of qubits that can be represented as |𝜓⟩ = (|00⟩ + |11⟩)/√2. In the example, Qiskit was used to create and measure this state on the IBM Quantum computer, resulting in the histogram showing mostly |00⟩ with some experimental errors.
x??

---

#### Full Adder Circuit
Background context on creating a full adder circuit for adding two bits using quantum gates like TOFFOLI (CCX) and CNOT.

:p What is a full adder, and how was it implemented in this example?
??x
A full adder adds two bits along with a carry bit. In the example, a three-qubit circuit was used to perform addition of x and y. The implementation involved using CCX gates for the main logic and CNOT gates as controlled-NOT operations.
x??

---

#### Quantum Adder Logic
Background context on constructing a quantum adder circuit step-by-step, including initialization and usage of quantum gates.

:p How does the `adder_circuit` function work in this example?
??x
The `adder_circuit` function initializes qubits representing x, y, and c (carry bit) and uses CCX and CNOT gates to perform addition. It starts by initializing the state, then applies a CCX gate for the main logic and a CNOT gate as an intermediate step. The final carry result is stored in one of the classical bits.
x??

---

#### Quantum Adder Code
Background context on the code provided for constructing the quantum adder circuit.

:p What does this function do?
??x
This function constructs a quantum circuit that adds two bit values x and y using CCX (Toffoli) gates and CNOT gates. It initializes qubits, applies necessary gates, and returns the resulting QuantumCircuit object.
```python
def adder_circuit(x_in: int, y_in : int) -> QuantumCircuit:
    # Initialize qubits for x, y, and c
    s = f"0{y_in:02b}{x_in:02b}"
    qc = QuantumCircuit(5, 3)
    
    # Apply initialization state
    qc.initialize(s)
    
    # Apply CCX gate for main logic
    qc.ccx(0, 2, 4)
    
    # Apply CNOT gate as intermediate step
    qc.cx(0, 2)
    
    # Reset qubit and apply another CCX for carry bit
    qc.reset(0)
    qc.ccx(1, 3, 0)
    
    return qc
```
x??

---

#### Quantum Adder Implementation
Background context on the implementation details of a quantum adder using multiple T OFFOLI (CCX) and CNOT gates.

:p What is the purpose of the more advanced IBM Quantum implementation shown in the text?
??x
The more advanced IBM Quantum implementation uses three CCX (Toffoli) gates, three CNOT gates, and additional measurement operations to perform the same addition logic. It includes resetting qubits and performing carry-bit handling.
x??

---

#### Measurement Operations
Background context on measuring states in quantum computing and the impact of experimental errors.

:p Why are measurement operations important in this context?
??x
Measurement operations determine the final state of a quantum system, converting superposition to definite classical bits. Experimental errors can introduce small counts for unexpected states due to imperfections in physical quantum devices.
x??

---

#### Quantum Fourier Transform Overview
The Quantum Fourier Transform (QFT) is a quantum version of the Discrete Fourier Transform, used extensively in quantum algorithms such as Shor's algorithm. It transforms an N-qubit quantum state into its frequency components, which are represented by complex numbers.

:p What is QFT and how does it differ from DFT?
??x
The Quantum Fourier Transform (QFT) is a quantum analog of the Discrete Fourier Transform (DFT). While DFT operates on classical data points in the time domain, transforming them into frequency components, QFT operates on qubits and transforms an N-qubit state into its spectral representation. Unlike classical DFT, which has a complexity of \(O(N^2)\), QFT can be performed using only \(O(N \log N)\) quantum gates.

The QFT is defined as:
\[ \text{QFT} |x\rangle = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{-2\pi i x y / N} |y\rangle \]

For a 2-qubit system, the QFT can be implemented using Hadamard and controlled phase gates. The general formula for the QFT of \(n\) qubits is:
\[ \text{QFT}|x\rangle = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{-2\pi i x y / N} |y\rangle \]
where \(x\) and \(y\) are binary representations of the qubit state.

```python
from qiskit import QuantumCircuit

def qft(n):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)  # Apply Hadamard gates
        for k in range(q+1, n):
            qc.cu1(-np.pi/float(2**(k-q)), q, k)
    return qc
```

x??

---

#### Controlled-Z (CZ) Gate Analysis
The CZ gate is a two-qubit quantum gate that performs the operation \(|00\rangle \rightarrow |00\rangle\), \(|01\rangle \rightarrow |01\rangle\), \(|10\rangle \rightarrow |10\rangle\), and \(|11\rangle \rightarrow -|11\rangle\).

:p What is the effect of the CZ gate on different input states?
??x
The Controlled-Z (CZ) gate operates as follows:
- It leaves the state \(|00\rangle\) unchanged.
- It leaves the state \(|01\rangle\) unchanged.
- It leaves the state \(|10\rangle\) unchanged.
- It applies a phase factor of \(-1\) to the state \(|11\rangle\).

This can be mathematically represented as:
\[ \text{CZ} |x\rangle = -1^{\langle x| 11\rangle} |x\rangle \]
where \(\langle x| 11\rangle\) is the inner product of \(|x\rangle\) with the state \(|11\rangle\).

```python
from qiskit import QuantumCircuit

def cz_gate():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)  # Apply CZ gate between qubits 0 and 1
    return qc
```

x??

---

#### CNOT Gate Operation on Different States
The CNOT (Controlled-NOT) gate is a fundamental two-qubit quantum logic gate. It flips the target qubit if the control qubit is in state \(|1\rangle\). The effect of CNOT on different states can be summarized as:
- \(|00\rangle \rightarrow |00\rangle\)
- \(|01\rangle \rightarrow |01\rangle\)
- \(|10\rangle \rightarrow |11\rangle\)
- \(|11\rangle \rightarrow |10\rangle\)

:p What is the effect of CNOT on different input states?
??x
The CNOT gate operates as follows:
- It leaves the state \(|00\rangle\) unchanged.
- It leaves the state \(|01\rangle\) unchanged.
- It flips the target qubit if the control qubit is in state \(|1\rangle\), so \(|10\rangle \rightarrow |11\rangle\) and \(|11\rangle \rightarrow |10\rangle\).

This can be represented by:
\[ \text{CNOT} |x\rangle = |0x\rangle + (-1)^{\langle x| 1\rangle}|1\overline{x}\rangle \]
where \(\langle x| 1\rangle\) is the inner product of \(|x\rangle\) with the state \(|1\rangle\).

```python
from qiskit import QuantumCircuit

def cnot_gate():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)  # Apply CNOT gate between qubits 0 (control) and 1 (target)
    return qc
```

x??

---

#### Creating Bell State with Qiskit
The Bell state \(|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)\) can be created using a quantum circuit. This involves applying an H gate to the first qubit and then a CNOT between the two qubits.

:p How do you create the Bell state \(|\beta_{11}\rangle\)?
??x
To create the Bell state \(|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)\), you can follow these steps:
1. Apply a Hadamard (H) gate to the first qubit.
2. Apply a CNOT gate between the two qubits, where the first qubit is the control and the second qubit is the target.

This can be implemented in Qiskit as follows:

```python
from qiskit import QuantumCircuit

def create_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard gate to first qubit
    qc.cx(0, 1)  # Apply CNOT between the two qubits
    return qc
```

x??

---

#### Halfadder Circuit with Qiskit
A halfadder circuit adds two single-bit binary numbers and produces a sum bit and a carry bit. The Qiskit implementation of a halfadder involves:
- Applying an H gate to both input qubits.
- Applying CNOT gates between the input qubits and output qubits.

The logic is that if both bits are 1, there will be no carry, but the sum will be 0; otherwise, the circuit behaves as expected with a sum bit and carry bit.

:p How do you create a halfadder circuit in Qiskit?
??x
To create a halfadder circuit in Qiskit, follow these steps:
1. Apply an H gate to both input qubits (q0 and q1).
2. Apply CNOT gates between the input qubits and output qubit for the sum.
3. Apply another CNOT gate between the same input qubits but target a carry bit.

Here is the Qiskit implementation:

```python
from qiskit import QuantumCircuit

def halfadder_circuit():
    qc = QuantumCircuit(3)  # Three qubits: two inputs and one output (sum)
    qc.h([0, 1])  # Apply Hadamard gates to input qubits
    qc.cx(0, 2)  # First CNOT for sum bit
    qc.cx(1, 2)  # Second CNOT for carry bit
    return qc
```

x??

---

#### 1-Qubit Quantum Fourier Transform (QFT)
Background context: The QFT is a quantum algorithm that transforms a quantum state representing a signal into another state. For one qubit, it computes \(N = 2^1\) components using Equation 12.77.

:p What is the purpose of the 1-qubit Quantum Fourier Transform (QFT)?
??x
The 1-qubit QFT transforms a single qubit state representing a signal into another state by computing two components. This transformation can be seen as equivalent to the Hadamard gate, which is used for creating superposition states.

Example: The QFT for one qubit can be expressed using Equation (12.78).

```java
// Pseudocode for 1-qubit QFT implementation
public class OneQubitQft {
    public static void applyQft(Qureg qubit) {
        double sqrtTwo = Math.sqrt(2);
        
        // Apply Hadamard gate to the first and second states
        qubit.apply(HadamardGate.H, 0);
        qubit.apply(HadamardGate.H, 1);

        // Apply phase shift gates for each state
        qubit.apply(PhaseShiftGate.PiByTwo(0), 1);
        qubit.apply(PhaseShiftGate.PiByTwo(1), 1);
    }
}
```
x??

---

#### 2-Qubit Quantum Fourier Transform (QFT)
Background context: The QFT for two qubits computes \(N = 2^2\) components using Equation 12.80, resulting in a transformation matrix given by Equation 12.81.

:p What does the QFT do with two qubits?
??x
The 2-qubit QFT transforms a quantum state representing a signal into another state by computing four components. This is represented by the matrix \(QFT_4\) defined in Equation (12.81).

Example: The QFT for two qubits can be expressed as:

```java
// Pseudocode for 2-qubit QFT implementation
public class TwoQubitQft {
    public static void applyQft(Qureg qubits) {
        // Apply Hadamard and phase shift gates in a specific sequence to achieve the QFT matrix
        qubits.apply(HadamardGate.H, 0);
        qubits.apply(PhaseShiftGate.PiByTwo(qubits.getIntValueAt(1)), 1);
        qubits.apply(SwapGate.SWAP, new int[]{0, 1});
    }
}
```
x??

---

#### General n-Qubit Quantum Fourier Transform (QFT)
Background context: The QFT for \(n\) qubits computes \(N = 2^n\) components. It uses the binary representation of the state to apply a series of Hadamard and phase shift gates as shown in Equation 12.95.

:p How does the n-qubit QFT work?
??x
The n-qubit QFT works by transforming an \(n\)-qubit state into another state that represents the Fourier components using a sequence of Hadamard and phase shift gates, as described in Equations (12.88) to (12.95).

Example: The Python-Qiskit implementation of this n-qubit QFT is given by:

```python
# Pseudocode for n-qubit QFT implementation using Qiskit
from qiskit.circuit.library import QFT

def apply_nqubit_qft(qc, num_qubits):
    qft = QFT(num_qubits)
    qc.append(qft.inverse(), range(num_qubits))
```
x??

---

#### Oracle + Diffuser = Grover's Search Algorithm
Background context: The goal is to search through a database of \(N = 2^n\) elements using quantum computing. This involves initializing the system, applying an oracle, and then using a diffuser (amplifier) to increase the amplitude of the target state.

:p What are the steps in Grover's Search Algorithm?
??x
Grover's Search Algorithm involves the following steps:
1. Initialize the \(n\)-qubit system into the ground state.
2. Apply Hadamard gates to create a superposition state.
3. Use an oracle to mark the target state by flipping its phase.
4. Apply a diffuser (amplifier) to increase the amplitude of the marked state.

Example: The implementation in Qiskit involves:

```python
# Pseudocode for Grover's Search Algorithm using Qiskit
def grovers_search(num_qubits, oracle):
    qc = QuantumCircuit(num_qubits)
    
    # Step 1: Apply Hadamard gates to create a superposition state
    for qubit in range(num_qubits):
        qc.h(qubit)
    
    # Step 2: Apply the oracle to mark the target state
    qc.append(oracle, range(num_qubits))
    
    # Step 3: Apply the diffuser (amplifier) multiple times
    for _ in range(int(np.pi/4 * np.sqrt(2**num_qubits))):
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        qc.append(oracle, range(num_qubits))
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))

    # Step 4: Measure the state
    qc.measure_all()
    
    return qc
```
x??

--- 

Each flashcard should cover a single concept or step from the provided text. The objective is to understand the context, formulas, and logic behind each topic without pure memorization. Use code examples where relevant to illustrate the concepts clearly.


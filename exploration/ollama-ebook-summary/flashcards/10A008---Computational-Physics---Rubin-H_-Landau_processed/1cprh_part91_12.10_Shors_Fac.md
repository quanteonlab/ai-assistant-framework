# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 91)

**Starting Chapter:** 12.10 Shors Factoring

---

#### Quantum Oracle and Grover's Algorithm

Background context: Grover’s algorithm is a quantum search algorithm that uses an oracle to amplify the amplitude of a desired state. The key components are the oracle $\mathcal{O}$ which marks the target state, and the diffuser operator $U_\omega$ which reflects amplitudes about the average.

Relevant formulas:
- $U_\psi = 2|\psi\rangle\langle\psi| - I $-$ U_\psi \sum_k \alpha_k |k\rangle = \sum_k [\alpha + (\alpha - \alpha_k)] |k\rangle$

:p What is the role of the diffuser operator in Grover’s algorithm?
??x
The diffuser operator $U_\omega$ is crucial as it amplifies the amplitude of the target state by reflecting amplitudes about their average. It ensures that states with higher probability amplitudes are reduced, while those closer to zero gain more probability.

Code example (pseudocode):
```python
def diffuser(n_qubits):
    # Initialize the circuit
    qc = QuantumCircuit(n_qubits)
    
    # Apply Hadamard gates
    for qubit in range(n_qubits):
        qc.h(qubit)
        
    # Apply controlled-Z operations to perform reflection about |ψ⟩
    # Here, |ψ⟩ is the state vector representing the average amplitude
    # This operation effectively creates a diffuser
    
    # Reflect amplitudes
    qc.x(range(n_qubits))
    qc.cz(0, n_qubits-1)
    for qubit in range(n_qubits - 2):
        qc.cz(qubit + 1, qubit + 2)
    qc.x(range(n_qubits))
    
    return qc

# Example of applying the diffuser
diffuser_circuit = diffuser(4)
```
x??

---

#### Grover's Oracle Implementation with Qiskit

Background context: In Grover’s algorithm, an oracle is a quantum circuit that marks the target state. For a 4-qubit system and $i=15 $(binary representation $|1111\rangle$), we need to flip the sign of this state.

Relevant formulas:
- Oracle for $|i\rangle$: A controlled-Z gate sandwiched between two Hadamard gates on each target qubit.
- Example: For $i=9 $(binary representation $|1001\rangle$), add X-gates before and after the Z-gate.

:p How is the oracle for a specific state implemented in Grover’s algorithm?
??x
The oracle for a specific state is implemented by placing a controlled-Z gate between two Hadamard gates on the corresponding qubits. For $i=15 $(binary representation $|1111\rangle$), this means applying three H gates, a Z gate, and then three more H gates.

Code example:
```python
from qiskit import QuantumCircuit

def create_oracle(i):
    # Create the oracle circuit
    qc = QuantumCircuit(4)
    
    if bin(i)[2:] == '1111':  # Check if i is 15 (binary representation of |1111⟩)
        # Apply controlled-Z gate to mark the target state
        qc.cz(0, 3)  # Controlled-Z between qubit 0 and qubit 3
        
    return qc

# Example of creating an oracle for |i=9⟩ (binary: 1001)
oracle = create_oracle(9)
```
x??

---

#### Shor’s Factoring Algorithm

Background context: Shor’s algorithm is a quantum factoring algorithm that exploits the difficulty of classical factorization to efficiently find prime factors of large integers. It relies on finding the period $T $ of a function defined as$f(x) = r^x \mod N$.

Relevant formulas:
- Period-finding step: Find the smallest $T $ such that$rT \equiv 1 \mod N$.
- Phase estimation for period finding.

:p What is Shor’s algorithm used for?
??x
Shor’s algorithm is a quantum algorithm designed to efficiently factorize large integers, which has significant implications in cryptography. It exploits the periodicity of functions defined over modulo operations and uses quantum phase estimation to find this period.

Code example (pseudocode):
```python
def shors_algorithm(N, r):
    # Initialize the circuit with necessary qubits for QFT and other operations
    qc = QuantumCircuit()
    
    # Perform QFT on the state |s⟩ where 2t𝜙=s
    qc.qft(range(n_qubits))
    
    # Apply controlled-U operations to find the period T
    for t in range(1, max_iterations):
        qc.append(controlled_U(r, N), [qc.qubits])
        
        # Perform inverse QFT
        qc.qft.inverse(range(n_qubits))
        
    return qc

# Example of running Shor's algorithm on 15 with r=7
shors_circuit = shors_algorithm(15, 7)
```
x??

---

#### Quantum Phase Estimation

Background context: Quantum phase estimation is a key subroutine in Shor’s factoring algorithm. It estimates the phase $\phi$ of an eigenvalue by measuring the state after applying quantum Fourier transform (QFT).

Relevant formulas:
- QFT on basis state |s⟩: 
  $$\text{QFT}_{2t}|s\rangle = \frac{1}{\sqrt{2^t}} \sum_{k=0}^{2^t - 1} e^{2\pi i s k / 2^t} |k\rangle$$- Inverse QFT to recover the phase:
$$|\phi\rangle = \text{QFT}_{2t}^{-1} [|\psi\rangle] \Rightarrow \phi = \frac{s}{2^t}$$:p What is quantum phase estimation used for?
??x
Quantum phase estimation is a technique used to determine the eigenvalues of unitary operators. In Shor’s algorithm, it helps in finding the period $T $ by estimating the phase$\phi = S/T $, where $ S $ is an integer between 0 and $ T-1$.

Code example (pseudocode):
```python
def quantum_phase_estimation(U, state):
    # Initialize the circuit with necessary qubits for QFT and other operations
    qc = QuantumCircuit()
    
    # Apply controlled-U operations to find the phase
    for t in range(1, max_iterations):
        qc.append(controlled_U(U), [qc.qubits])
        
        # Perform inverse QFT
        qc.qft.inverse(range(n_qubits))
        
        # Measure and get the phase
        result = qc.run()
    
    return result

# Example of running phase estimation on a unitary operator U
phase_estimation_result = quantum_phase_estimation(U, state)
```
x??

--- 

Note: The code examples provided are simplified pseudocode to illustrate the logic. Actual implementation would require using Qiskit or similar libraries and might involve more complex operations.

#### Entangled Quantum States and Hamiltonian Calculation

**Background context:**
In quantum computing, entangled states are a fundamental concept where the state of one particle is directly related to the state of another. The Hamiltonian represents the total energy of a system and can be used to describe how the system evolves over time.

The provided code `Entangle.py` computes the Hamiltonian for entangled quantum states using numpy, calculates its eigenvalues and eigenvectors, and then re-evaluates the Hamiltonian in the new basis formed by these eigenvectors.

:p What is the purpose of the `Entangle.py` script?
??x
The primary purpose of the `Entangle.py` script is to calculate the Hamiltonian for entangled quantum states, compute its eigenvalues and eigenvectors, and then transform the Hamiltonian into a new basis formed by these eigenvectors. This process helps in understanding how the energy levels change under certain transformations.

Code example:
```python
# Import necessary libraries
from numpy import *
from numpy.linalg import *

nmax = 4

H = zeros((nmax,nmax), float)

XAXB = array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
YAYB = array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
ZAZB = array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

SASB = XAXB + YAYB + ZAZB - 3*ZAZB

# Print the Hamiltonian without the mu^2/r^3 factor
print('Hamiltonian without muˆ2/rˆ3 factor', SASB)

# Compute eigenvalues and eigenvectors of the Hamiltonian
es, ev = eig(SASB)

# Extract vectors
phi1 = (ev[0,0], ev[1,0], ev[2,0], ev[3,0])
phi4 = (ev[0,1], ev[1,1], ev[2,1], ev[3,1])
phi3 = (ev[0,2], ev[1,2], ev[2,2], ev[3,2])
phi2 = (ev[0,3], ev[1,3], ev[2,3], ev[3,3])

basis = [phi1, phi2, phi3, phi4]

# Hamiltonian in the new basis
for i in range(0,nmax):
    for j in range(0, nmax):
        term = dot(SASB, basis[i])
        H[i,j] = dot(basis[j], term)

print('Hamiltonian in Eigenvector Basis', H)
```
x??

---

#### 2-Qubit Quantum Fourier Transform (QFT) Using Qiskit

**Background context:**
The Quantum Fourier Transform (QFT) is a key operation in quantum computing that transforms the state of a qubit register into its frequency domain representation. The provided code `QFT4.py` demonstrates how to implement a 2-qubit QFT using Qiskit.

:p What does the `QFT4.py` script do?
??x
The `QFT4.py` script creates and prints out a 2-qubit Quantum Fourier Transform (QFT) circuit. It uses Qiskit to define and execute the QFT, which is essential for many quantum algorithms including Shor's algorithm.

Code example:
```python
# Import necessary libraries
from math import pi
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

def qft2(inverse=False) -> QuantumCircuit:
    angle = pi/2
    if inverse is True: 
        angle = -angle
    qc = QuantumCircuit(2)
    
    # Apply Hadamard gate on second qubit
    qc.h(1)
    # Apply controlled phase shift
    qc.cu1(angle, 0, 1)
    
    return qc

# Print the QFT circuit for verification
print(qft2())
```
x??

---

#### General Diffuser Circuit Implementation in Qiskit

**Background context:**
A diffuser circuit is a component used in quantum algorithms like Grover's algorithm. It helps amplify the amplitude of a target state, making it more likely to be measured.

The provided code `diffuser.py` demonstrates how to implement a general diffuser circuit with `n_qubits` qubits using Qiskit.

:p What does the `diffuser.py` script do?
??x
The `diffuser.py` script defines a generic diffuser circuit that operates on a register of `n_qubits`. This circuit is used in quantum algorithms such as Grover's search algorithm to amplify the amplitude of a target state, making it more probable during measurement.

Code example:
```python
# Import necessary libraries
from qiskit import QuantumCircuit

def diffuser(n_qubits: int):
    # Create a circuit with n_qubits
    qc = QuantumCircuit(n_qubits)
    
    # Map |psi> to |0...0>
    qc.h(range(n_qubits))
    # Map |0...0> to |1...1>
    qc.x(range(n_qubits))
    # Multiply controlled-Z gates (Z gate) to flip the sign for |1...1>
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    # Map back from |1...1> to |0...0>
    qc.x(range(n_qubits))
    # Map back to |psi>
    qc.h(range(n_qubits))

    return qc
```
x??

---

#### Shor's Algorithm Quantum Circuit Implementation

**Background context:**
Shor's algorithm is a quantum algorithm for integer factorization, which is exponentially faster than the best known classical algorithms. The provided code `Shor.py` starts to outline the implementation of Shor's algorithm.

:p What does the `Shor.py` script aim to do?
??x
The `Shor.py` script aims to provide a starting point for implementing Shor's algorithm, which is used to factorize large integers. This involves defining the quantum circuit and setting up the necessary operations to perform the quantum Fourier transform and other steps of the algorithm.

Code example:
```python
# Import necessary libraries
from qiskit import QuantumCircuit

def shors_algorithm():
    # Define the number of qubits
    n = 5
    
    # Create a quantum circuit with `n` qubits
    qc = QuantumCircuit(n)
    
    # Apply Hadamard gates to create superposition
    for i in range(n):
        qc.h(i)
    
    # Add other operations required for Shor's algorithm (to be completed)
    pass
    
    return qc

# Print the quantum circuit for verification
print(shors_algorithm())
```
x??

---

#### Grover's Algorithm Quantum Circuit Implementation on IBMQ

**Background context:**
Grover's algorithm is a quantum search algorithm that provides a quadratic speedup over classical algorithms. The provided code `Grover.py` demonstrates how to implement and run Grover's algorithm using both the Aer simulator and an actual IBMQ device.

:p What does the `Grover.py` script do?
??x
The `Grover.py` script implements Grover's search algorithm, first running it on the Aer simulator for verification purposes. It then uses an IBMQ device to run the same algorithm, providing a practical demonstration of how quantum circuits can be executed on real hardware.

Code example:
```python
# Import necessary libraries
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.ibmq import least_busy

def grovers_algorithm(cap_n=4):
    # Create a 4-qubit quantum circuit
    qc = QuantumCircuit(cap_n)
    
    # Apply Hadamard gates to create superposition
    for i in range(cap_n):
        qc.h(i)
    
    # Define the oracle and diffusion steps (to be completed)
    pass
    
    return qc

# Print the quantum circuit for verification
print(grovers_algorithm())

# Transpile and run on Aer simulator
backend = Aer.get_backend('qasm_simulator')
transpiled_circuit = transpile(grovers_algorithm(), backend)
job = backend.run(transpiled_circuit)

result = job.result()
histogram = result.get_counts()
print(histogram)
```
x??

--- 

These flashcards cover the key concepts and code examples from the provided text, focusing on explaining the context, background, and implementation details. Each card has a single question that prompts understanding of the concept described.

#### Shor's Algorithm Introduction
Shor’s algorithm is a quantum algorithm for integer factorization. It was published by Peter Shor in 1994 and can efficiently factorize large integers, which has significant implications for cryptography. The algorithm leverages the power of quantum computing to find the factors of an integer $N$ exponentially faster than the best-known classical algorithms.

:p What is Shor's Algorithm used for?
??x
Shor’s Algorithm is primarily used for integer factorization, a problem that is computationally difficult on classical computers but can be solved much more efficiently using quantum computers. This algorithm has profound implications for cryptography, particularly public-key encryption systems like RSA.
x??

---

#### Function `amod15`
The function `amod15` takes an input $a $ and performs certain operations based on the value of$a$. It creates a Quantum Circuit that applies a series of swaps to simulate modular exponentiation modulo 15.

:p What does the `amod15` function do?
??x
The `amod15` function constructs a quantum circuit that simulates the operation $a^{2^k} \mod 15 $ by applying controlled NOT (CNOT) gates and swaps. The specific operations are based on the value of$a$.

For example:
- If $a = 2, 13$, it performs specific swap operations to simulate the modular exponentiation.
- It sets up the quantum circuit with four qubits and applies these operations.

The function is designed for use in Shor’s algorithm where such modular arithmetic is crucial.

```python
def amod15(a_in: int,p_in : int) -> QuantumCircuit:
    if a_in not in [2,4,7,8,11,13,14]:
        raise ValueError("a_in must be 2,4,7,8,11,13 or 14")
    quantum_circuit = QuantumCircuit(4)
    
    for iteration in range(p_in):
        if a_in in [2, 13]:
            # Swap operations
            quantum_circuit.swap(2, 3)
            quantum_circuit.swap(1, 2)
            quantum_circuit.swap(0, 1)
        
        if a_in in [7, 8]:
            quantum_circuit.swap(0, 1)
            quantum_circuit.swap(1, 2)
            quantum_circuit.swap(2, 3)
        
        if a_in in [4, 11]:
            # Swap operations
            quantum_circuit.swap(1, 3)
            quantum_circuit.swap(0, 2)

        if a_in in [7, 11, 13, 14]: 
            for i in range(4):
                quantum_circuit.x(i)
    
    quantum_circuit.name = "percentiˆpercenti mod 15" + str(a_in) + "^{" + str(p_in) + "}mod 15"
    percent(a_in,p_in)
    return quantum_circuit
```
x??

---

#### Function `qpe`
The function `qpe` builds a Phase Estimation circuit, which is essential for the Shor’s algorithm. This function takes a list of Quantum Circuits and returns an estimated phase.

:p What does the `qpe` function do?
??x
The `qpe` function constructs a quantum circuit to perform phase estimation on a given unitary operator $U $ raised to powers of 2, up to$2^{t-1}$. The function uses controlled versions of the unitary gates and applies an inverse Quantum Fourier Transform (QFT) at the end.

The code snippet below provides a detailed view of how `qpe` works:

```python
def qpe(u_list: List[QuantumCircuit]) -> float:
    # Build phase circuit, u_list : list of QuantumCircuits [U^(2^0), U^(2^1), ... , U^(2^(t-1))]
    t = len(u_list)
    num_qubits_u = u_list[0].num_qubits  # N qubits for cap_U gate

    qc = QuantumCircuit(t + num_qubits_u, t)

    # Put the first t_count qubits into superposition
    for i in range(t):
        qc.h(i) 

    # Put the last n_u qubit into |1> state
    qc.x(t)

    # Add contr-U^(2^j) gates
    for i in range(t): 
        qc.append(u_list[i].to_gate().control(), [i] +[ j + t for j in range(num_qubits_u)])

    # Inverse QFT
    qc.append(QFT(t, inverse=True).to_gate(), range(t))

    # Measure
    qc.measure(range(t), range(t))
    
    simulator = Aer.get_backend("aer_simulator")
    q_obj = assemble(transpile(qc, simulator), shots=1)
    result = simulator.run(q_obj, memory=True).result()
    readings = result.get_memory()

    print("Register reading: " + readings[0])
    phase = int(readings[0], 2) / (2 ** t)
    print("Corresponding phase: ", phase)

    return phase
```
x??

---

#### Main Function
The main function of the script is responsible for running Shor's algorithm. It randomly selects an integer $a$ and uses the `amod15` and `qpe` functions to find factors of 15.

:p What does the main part of the code do?
??x
The main part of the script runs Shor’s algorithm to factorize the number 15. It iteratively selects random integers $a $ between 2 and 14, checks if$gcd(a, 15) = 1$, and uses `amod15` and `qpe` functions to estimate the phase. Once a non-trivial factor is found, it prints the result.

```python
if __name__ == "__main__":
    cap_n = 15
    factor_found = False
    attempt = 0

    while not factor_found:
        print("Attempt #", attempt)
        attempt += 1

        a = random.randint(2, cap_n - 1)
        print("Random a =", a)

        k = gcd(a, cap_n)
        if k == 1:
            factor_found = True
            print("Found factor:", k)
        
        else:
            p = qpe([amod15(a, 2 ** j) for j in range(8)])
            print("Phase:", p)

            fraction = Fraction(p).limit_denominator(cap_n)
            s, r = fraction.numerator, fraction.denominator
            print("r =", r)

            if r % 2 == 0:
                guesses = [gcd(a ** (r // 2) + 1, cap_n), gcd(a ** (r // 2) - 1, cap_n)]

                for g in guesses:
                    if g not in [1, cap_n] and (cap_n % g) == 0:
                        print("Found factor:", g)
                        factor_found = True
```
x??

---

#### Quantum Eigenvalues for Arbitrary Potentials
Background context: In quantum mechanics, particles are described by wave functions and their energy levels. The time-independent Schrödinger equation (13.1) is used to find these wave functions and corresponding energies.

The problem involves finding the eigenvalues of the Hamiltonian operator in a potential that confines the particle within an atomic distance. For bound states ($E < 0 $), the wave function $\psi(x)$ must be normalizable, implying it decays exponentially at infinity (13.4).

:p What is the Schrödinger equation for a particle with energy $E $ and potential$V(x)$?
??x
The time-independent Schrödinger equation for a one-dimensional system is given by:
$$-\frac{\hbar^2}{2m} \frac{d^2\psi(x)}{dx^2} + V(x) \psi(x) = E \psi(x)$$where $\psi(x)$ is the wave function,$ V(x)$ is the potential energy, and $E$ is the particle's total energy.

The equation can be rewritten as:
$$\frac{d^2\psi(x)}{dx^2} - 2m V(x) \frac{\psi(x)}{\hbar^2} = -\frac{\hbar^2}{2m} E$$

For bound states, the wave function $\psi(x)$ must satisfy certain boundary conditions at infinity. This turns the ODE into an eigenvalue problem.

x??

---

#### Model: Nucleon in a Box
Background context: A simple model of a particle (nucleon) confined within a finite potential well is introduced to illustrate solving the Schrödinger equation numerically. The potential $V(x)$ for this box model is given by:
$$V(x) = \begin{cases}
-83 \text{ MeV}, & |x| \leq 2 \text{ fm} \\
0, & |x| > 2 \text{ fm}
\end{cases}$$
where the typical values for nuclear states are used.

:p What is the form of the Schrödinger equation inside and outside the well?
??x
Inside the well ($|x| \leq a$), the Schrödinger equation becomes:
$$\frac{d^2\psi(x)}{dx^2} + 2m \left( -\frac{\hbar^2 V_0}{2m} - \frac{\hbar^2 E}{2m} \right) \psi(x) = 0$$or equivalently:
$$\frac{d^2\psi(x)}{dx^2} + (2m \left( -\frac{\hbar^2 V_0}{2m} - \frac{\hbar^2 E}{2m} \right)) \psi(x) = 0$$

Simplifying further:
$$\frac{d^2\psi(x)}{dx^2} + (2m \left( -V_0 - \frac{E}{c^2} \right)) \psi(x) = 0$$

Outside the well ($|x| > a$), it simplifies to:
$$\frac{d^2\psi(x)}{dx^2} + \frac{\hbar^2 E}{2m} \psi(x) = 0$$where $ c$ is the speed of light.

x??

---

#### Algorithm: ODE Solver + Search
Background context: The algorithm combines a numerical integration (rk4ODE solver) with a search for an eigenvalue that satisfies both boundary conditions at infinity. This involves integrating from $-\infty $ and$\infty$ towards the well, ensuring continuity of the wave function.

:p What are the steps to solve the eigenvalue problem numerically?
??x
1. **Start at $x = -x_{\infty}$(extreme left)**: Assume an initial wave function that satisfies the boundary condition $\psi_L(x = -x_\infty) = e^{-\psi x} = e^{\psi x_\infty}$.

2. **Integrate towards origin from $x = -x_{\infty}$**: Use the rk4ODE solver to integrate step-by-step, ensuring you are integrating over an increasing function.

3. **Match at matching radius $x_m $**: Integrate up to a point just beyond the right edge of the potential well ($ a$).

4. **Start at $x = +x_{\infty}$(extreme right)**: Assume an initial wave function that satisfies $\psi_R(x = x_\infty) = e^{-\psi x} = e^{-\psi x_\infty}$.

5. **Integrate towards origin from $x = +x_{\infty}$**: Use the rk4ODE solver to integrate step-by-step, ensuring you are integrating over an increasing function.

6. **Match at matching radius $x_m $**: Integrate up to a point just beyond the right edge of the potential well ($ a$).

The solution is found by trial and error, adjusting the energy until both wave functions match at $x = x_m$.

x??

---

#### Concept of Normalization
Background context: For bound states, $\psi(x)$ must be normalizable. This implies that as $|x|$ approaches infinity ($x \to \pm \infty $), the potential$ V $goes to zero, and$\psi(x)$ should decay exponentially.

:p What are the boundary conditions for a wave function in an infinite well?
??x
For an infinite square well, the boundary conditions on the wave function at infinity are:
1. At $x = -\infty $, $\psi_L(x = -\infty) = e^{\pm \psi x} = e^{-\psi x_\infty}$.
2. At $x = +\infty $, $\psi_R(x = +\infty) = e^{\pm \psi x} = e^{-\psi x_\infty}$.

These conditions ensure that the wave function decays exponentially as it approaches infinity, maintaining normalizability.

x??

---

#### Concept of Matching
Background context: The solution is found by integrating from both sides and matching the wave functions at a point $x_m$ just beyond the well's edge. This ensures continuity and satisfies the boundary conditions.

:p What does it mean to "match" the wave function at $x_m$?
??x
Matching the wave function at $x_m $ means ensuring that the value of$\psi(x)$ from integrating towards the origin from both the left ($-\infty $) and right ($+\infty$) sides are equal at this point. This continuity condition is crucial for the solution to be valid within the well.

The wave function integrated from the left must match the wave function integrated from the right at $x_m$.

x??

---


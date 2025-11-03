# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 88)

**Starting Chapter:** Chapter 12 Quantum Computing G. He Coauthor. 12.3.1 Physics Exercise Two Entangled Dipoles

---

#### Quantum State Representation Using Dirac Notation

Background context: In quantum mechanics, states are represented using a formalism introduced by Paul Dirac. This notation uses kets (vectors) and bras (covectors), which together form a powerful way to describe quantum states.

The ket \( |\psi\rangle \) represents the state vector of a quantum system in an abstract Hilbert space, while the bra \( \langle x| \) is its dual adjoint. The inner product between two vectors, say \( |x\rangle \) and \( |\psi\rangle \), is denoted by \( \langle x|\psi\rangle \).

Formula: 
\[ \psi(x) = \langle x|\psi\rangle \]

This formula represents the wavefunction of a state in position space.

:p What does the equation \( \psi(x) = \langle x|\psi\rangle \) represent?
??x
The equation \( \psi(x) = \langle x|\psi\rangle \) represents the wavefunction or probability amplitude of finding the quantum state \( |\psi\rangle \) at position \( x \). It is essentially the inner product between the bra \( \langle x| \) and the ket \( |\psi\rangle \).

Example in Python:
```python
# Define a function to compute the wavefunction
def wavefunction(x, psi_ket):
    # Assume psi_ket is a vector representing the quantum state
    return np.dot(np.conj(psi_ket), x_vector)
```
x??

---

#### Dirac Notation for Qubits

Background context: Quantum bits or qubits are the fundamental units of information in quantum computing. They can exist in multiple states simultaneously, represented by kets and bras.

Formula:
\[ \langle\psi| = |\psi\rangle^\dagger \]
Here \( \langle\psi| \) is a bra (conjugate transpose of the ket \( |\psi\rangle \)).

:p What does the formula \( \langle\psi| = |\psi\rangle^\dagger \) signify?
??x
The formula \( \langle\psi| = |\psi\rangle^\dagger \) signifies that the bra \( \langle\psi| \) is the conjugate transpose of the ket \( |\psi\rangle \). This relationship ensures a 1:1 correspondence between kets and bras.

Example in Python:
```python
# Define a function to compute the adjoint (bra)
def create_bra(ket):
    # Assume ket is a vector representing the quantum state
    return np.conj(np.transpose(ket))
```
x??

---

#### Inner Product of Quantum States

Background context: The inner product between two states \( |\phi\rangle \) and \( |\psi\rangle \) is denoted by \( \langle\phi|\psi\rangle \). This operation provides a scalar result, often referred to as the projection of one state onto another.

Formula:
\[ \langle\phi|\psi\rangle = (\phi,\psi) = \langle\psi|\phi\rangle^* \]

:p What does the formula \( \langle\phi|\psi\rangle = (\phi,\psi) = \langle\psi|\phi\rangle^* \) represent?
??x
The formula \( \langle\phi|\psi\rangle = (\phi,\psi) = \langle\psi|\phi\rangle^* \) represents the inner product between two quantum states, where \( \langle\phi|\psi\rangle \) is a scalar value obtained by taking the conjugate transpose of one ket and multiplying it with another bra. The result is complex-conjugated if you switch the order.

Example in Python:
```python
# Define a function to compute the inner product
def inner_product(phi_bra, psi_ket):
    # Assume phi_bra is a vector representing the bra state and psi_ket is a vector representing the ket state
    return np.dot(np.conj(phi_bra), psi_ket)
```
x??

---

#### Operator Representation Using Dirac Notation

Background context: Operators in quantum mechanics are represented using bras and kets. An operator \( O \) acting on a ket \( |\psi\rangle \) results in another ket \( |O\psi\rangle \).

Formula:
\[ O|ψ⟩ = |𝜙⟩ = |Oψ⟩ \]

:p What does the formula \( O|\psi⟩ = |\phi⟩ = |O\psi⟩ \) signify?
??x
The formula \( O|\psi⟩ = |\phi⟩ = |O\psi⟩ \) signifies that applying an operator \( O \) to a ket \( |\psi⟩ \) results in another ket state \( |\phi⟩ \), which is the same as writing \( |O\psi⟩ \).

Example in Python:
```python
# Define a function to apply an operator on a quantum state
def apply_operator(operator_matrix, psi_ket):
    # Assume operator_matrix is a 2x2 matrix representing the operator and psi_ket is a vector representing the ket state
    return np.dot(operator_matrix, psi_ket)
```
x??

---

#### Qubit States in Spin1/2 Space

Background context: In spin-1/2 systems, qubits can be represented by kets corresponding to spin-up or spin-down states. These are commonly denoted as \( |0⟩ \) and \( |1⟩ \).

Formulas:
\[ |+\rangle = \left|\frac{1}{2}\right\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = |0⟩ \]
\[ |-\rangle = \left| -\frac{1}{2} \right\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = |1⟩ \]

:p What are the spin-1/2 states \( |+\rangle \) and \( |-\rangle \) represented by in Dirac notation?
??x
The spin-1/2 states \( |+\rangle \) and \( |-\rangle \) are represented as follows:
\[ |+\rangle = \left| +\frac{1}{2} \right\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = |0⟩ \]
\[ |-\rangle = \left| -\frac{1}{2} \right\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = |1⟩ \]

These states represent the up and down spin states, where \( |0⟩ \) corresponds to an "up" state and \( |1⟩ \) corresponds to a "down" state.

Example in Python:
```python
# Define spin-1/2 states as kets
plus_state = np.array([1, 0])
minus_state = np.array([0, 1])
```
x??

---

#### Operator Representation Using Direct Product

Background context: Operators can be represented using direct products of kets and bras. For example, the operator \( O \) applied to a ket \( |\psi\rangle \) results in another ket.

Formula:
\[ O = |ψ⟩⟨𝜙| \]

:p What does the formula \( O = |ψ⟩⟨𝜙| \) represent?
??x
The formula \( O = |ψ⟩⟨𝜙| \) represents an operator \( O \) constructed from a bra and a ket. Specifically, it is formed by taking the outer product of the bra \( ⟨𝜙| \) with the ket \( |\psi\rangle \). This results in an operator that maps states into other states.

Example in Python:
```python
# Define an operator using direct product (outer product)
def create_direct_product_operator(phasor):
    # Assume phasor is a vector representing the state |𝜙⟩
    return np.outer(np.conj(phasor), phasor)
```
x??

---

#### Qubits and Quantum States
Background context: In quantum computing (QC), information is stored using qubits, which are quantum mechanical states. These qubits can be in a superposition of two basic states |0⟩ and |1⟩, similar to bits but with additional properties due to their quantum nature.
:p What is the fundamental difference between classical bits and qubits?
??x
In classical computing, information is stored using binary digits (bits) that are either 0 or 1. In contrast, qubits can exist in a superposition of both states |0⟩ and |1⟩ simultaneously. This property allows quantum computers to process a vast amount of data much faster than classical computers.
x??

---
#### Quantum State Representation
Background context: A single qubit is represented by a linear combination of the basis states |0⟩ and |1⟩, which are defined using unit vectors in a two-dimensional complex vector space. The state of a qubit can be described as \(|ψ⟩ = u|0⟩ + v|1⟩\), where \(u\) and \(v\) are complex numbers that satisfy the normalization condition: \(|u|^2 + |v|^2 = 1\).
:p How is a single-qubit state represented mathematically?
??x
A qubit's state is represented as a linear combination of the basis states:
\[ |ψ⟩ = u|0⟩ + v|1⟩ \]
where \(u\) and \(v\) are complex numbers, and the normalization condition ensures that the total probability amplitude squared equals 1.
```java
// Pseudocode for representing a qubit state
public class Qubit {
    private ComplexNumber u;
    private ComplexNumber v;

    public Qubit(ComplexNumber u, ComplexNumber v) {
        this.u = u;
        this.v = v;
    }

    // Method to check if the qubit is normalized
    public boolean isNormalized() {
        return Math.abs(u.norm() + v.norm() - 1.0) < tolerance; // tolerance is a small value like 1e-6
    }
}
```
x??

---
#### Bloch Sphere Representation
Background context: The state of a qubit can also be visualized using the Bloch sphere, which provides a geometric representation of two-level quantum systems. The state \(|ψ⟩ = u|0⟩ + v|1⟩\) can be represented on this sphere in terms of its polar angles \(\theta\) and \(\phi\).
:p How is a qubit state expressed on the Bloch sphere?
??x
A qubit state \(|ψ⟩ = u|0⟩ + v|1⟩\) can be expressed using the Bloch sphere coordinates:
\[ |ψ⟩ = \cos(\frac{\theta}{2})|0⟩ + e^{i\phi}\sin(\frac{\theta}{2})|1⟩ \]
where \(\theta \in [0, \pi]\) and \(\phi \in [0, 2\pi)\).
```java
// Pseudocode for converting qubit state to Bloch sphere coordinates
public class Qubit {
    // ... existing fields ...

    public Point3D toBlochSphere() {
        double theta = 2 * Math.acos(Math.sqrt(v.norm())); // Calculate theta from v's norm
        double phi = ComplexNumber.arg(v); // Calculate phi from the phase of v
        return new Point3D(sin(theta), cos(theta) * cos(phi), cos(theta) * sin(phi));
    }
}
```
x??

---
#### Multiple Qubit States and Tensor Products
Background context: When combining states in multiple qubits, the tensor product is used to create a composite state. For two separate qubits \(|ψ_A⟩\) and \(|ψ_B⟩\), their combined state is expressed as:
\[ |ψ_{AB}⟩ = |ψ_A⟩ \otimes |ψ_B⟩ \]
This operation results in a vector with four components for each pair of basis states.
:p How does the tensor product work when combining two qubits?
??x
The tensor product combines two qubit states into a single state in an expanded Hilbert space. For example, if we have:
\[ |ψ_A⟩ = u_1|0⟩ + v_1|1⟩ \]
and
\[ |ψ_B⟩ = u_2|0⟩ + v_2|1⟩ \]
their combined state is:
\[ |ψ_{AB}⟩ = (u_1|0⟩ + v_1|1⟩) \otimes (u_2|0⟩ + v_2|1⟩) = u_1u_2|00⟩ + u_1v_2|01⟩ + v_1u_2|10⟩ + v_1v_2|11⟩ \]
where \(|00⟩\), \(|01⟩\), \(|10⟩\), and \(|11⟩\) are basis states in the combined Hilbert space.
```java
// Pseudocode for calculating tensor product of two qubits
public class Qubit {
    // ... existing fields ...

    public static Qubit[] tensorProduct(Qubit qa, Qubit qb) {
        double u1 = qa.u.norm(), v1 = qa.v.norm();
        double u2 = qb.u.norm(), v2 = qb.v.norm();

        return new Qubit[]{
                new Qubit(u1 * u2, 0), // |00⟩
                new Qubit(u1 * v2, Math.PI / 2), // |01⟩
                new Qubit(v1 * u2, -Math.PI / 2), // |10⟩
                new Qubit(v1 * v2, 0) // |11⟩
        };
    }
}
```
x??

---
#### Entangled and Separable States
Background context: In QC, the concept of entanglement is crucial. Two qubits are said to be entangled if their combined state cannot be factored into separate states. For example:
\[ |ψ_{AB}⟩ = u_1u_2|00⟩ + u_1v_2|01⟩ + v_1u_2|10⟩ + v_1v_2|11⟩ \]
is an entangled state, whereas
\[ |ψ_A⟩⊗|ψ_B⟩ = (u_1|0⟩+v_1|1⟩)(u_2|0⟩+v_2|1⟩) \]
represents a separable or product state.
:p How do you determine if two qubits are entangled?
??x
To determine if two qubits are entangled, check if their combined state can be factored into the tensor product of individual states. If it cannot, then the qubits are entangled.

For example:
\[ |ψ_{AB}⟩ = u_1u_2|00⟩ + u_1v_2|01⟩ + v_1u_2|10⟩ + v_1v_2|11⟩ \]
is an entangled state because it cannot be written as a product of individual states.

```java
// Pseudocode for checking if a qubit state is separable
public class Qubit {
    // ... existing fields ...

    public boolean isSeparable() {
        double[][] coefficients = tensorProduct(this, this).toMatrix();
        return isSquareFree(coefficients); // Function to check square-free condition
    }
}
```
x??

---

#### Definition of Entangled and Separable States
The states formed using a direct product, such as those mentioned in equation (12.16), are called separable. For example, a qubit in a \(|0\rangle\) state and a different qubit also in a \(|0\rangle\) state form the separable state \(|0_A\rangle \otimes |0_B\rangle\), which is usually written as just \(|00\rangle\). However, if two interacting systems are otherwise isolated but cannot be expressed as the direct product of their individual states, these qubits are entangled.
:p What distinguishes separable and entangled states?
??x
Entangled states refer to quantum states that cannot be factored into a simple tensor product of individual states. This means that the state of one particle affects the state of another, even if they are separated by large distances. For instance, spin-up (\( |0\rangle \)) and spin-down (\( |1\rangle \)) states can be physically far from each other yet still entangled.
??x

---

#### Bell States as Examples of Entanglement
The Bell states, defined in equations (12.22) and (12.23), are a famous example of entanglement:
\[|\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)\]
\[|\beta_{01}\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)\]
\[|\beta_{10}\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)\]
\[|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)\]

:p Prove that the Bell states are entangled.
??x
To prove that the Bell states are entangled, we need to show that they cannot be written as a simple tensor product of individual qubit states. For instance, consider \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
Suppose this state could be written as a separable state:
\[ |\beta_{00}\rangle = |u_1\rangle \otimes |v_1\rangle + |u_2\rangle \otimes |v_2\rangle \]
This would mean that \(|00\rangle\) and \(|11\rangle\) must each be separable, which is not possible since they involve both qubits together.
??x

---

#### Density Matrix Representation
The density matrix or operator, defined in equation (12.24), can be used to describe the quantum state of a system without resorting to wavefunctions:
\[ \rho = \sum_i p_i | \psi_i \rangle \langle \psi_i| \]
where \(p_i\) is the probability of the pure state \(|\psi_i\rangle\) being present in the ensemble. For a single pure state, \(p_1 = 1\).

:p Explain how to use the density matrix for an ensemble of states.
??x
The density matrix allows us to describe mixed states, which are combinations of pure states with certain probabilities. It is particularly useful when dealing with ensembles of pure states. For example, if we have two pure states \(| \psi_1 \rangle\) and \(| \psi_2 \rangle\) with probabilities \(p_1 = 0.6\) and \(p_2 = 0.4\), the density matrix would be:
\[ \rho = p_1 | \psi_1 \rangle \langle \psi_1 | + p_2 | \psi_2 \rangle \langle \psi_2 | \]
This representation helps in calculating observables without needing to explicitly use wavefunctions.
??x

---

#### Two Entangled Dipoles Hamiltonian
The interaction Hamiltonian between two interacting magnetic dipoles \(\sigma_A\) and \(\sigma_B\), separated by a distance \(r\), is given by equation (12.25):
\[ H = \mu^2 r^{-3} (\sigma_A \cdot \sigma_B - 3 \sigma_A \cdot \hat{r} \sigma_B \cdot \hat{r}) \]
where \(\hat{r}\) is the unit vector in the direction from \(A\) to \(B\).

:p Derive the Hamiltonian for two interacting dipoles with \(\hat{r} = \hat{k}\).
??x
Given that \(\sigma_A = X_A \hat{i} + Y_A \hat{j} + Z_A \hat{k}\) and \(\sigma_B = X_B \hat{i} + Y_B \hat{j} + Z_B \hat{k}\), the Hamiltonian in equation (12.25) becomes:
\[ H = \mu^2 r^{-3} (\sigma_A \cdot \sigma_B - 3 \sigma_A \cdot \hat{r} \sigma_B \cdot \hat{r}) \]
When \(\hat{r} = \hat{k}\), the dot product simplifies to:
\[ \sigma_A \cdot \hat{r} = Z_A, \quad \sigma_B \cdot \hat{r} = Z_B \]
Thus, the Hamiltonian becomes:
\[ H = \mu^2 r^{-3} (X_A X_B + Y_A Y_B + Z_A Z_B - 3 Z_A Z_B) \]

This can be written in matrix form as:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B + Z_A \otimes Z_B - 3 Z_A \otimes Z_B) \]
??x

---

#### Pauli Matrices and Direct Products
The Pauli matrices \(X\), \(Y\), and \(Z\) are operators that transform states, as shown in equations (12.27):
\[ X = \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad Y = \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}, \quad Z = \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \]

:p Derive the direct product of Pauli matrices \(X_A\) and \(X_B\).
??x
The direct product of two Pauli matrices, such as \(X_A\) and \(X_B\), is given by:
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
This can be expanded as:
\[ X_A \otimes X_B = \begin{bmatrix} 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \\ 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
??x

---

#### Direct Product of Operators
The direct product of operators in separate Hilbert spaces is a fundamental operation. For instance, the direct product \(X_A \otimes X_B\) can be expressed as:
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]

:p Explain how to compute the direct product of two operators.
??x
The direct product of two operators \(A\) and \(B\), denoted as \(A \otimes B\), is computed by placing each element of operator \(A\) in every position, multiplying it with corresponding elements from operator \(B\). For example:
\[ X_A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
The direct product is:
\[ X_A \otimes X_B = \begin{bmatrix} 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \\ 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
??x

--- 
#### Eigenvalues and Eigenvectors of the Hamiltonian
The eigenvalues and eigenvectors of the Hamiltonian are derived in equation (12.40). For instance, for \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]

:p Find the eigenvalues and eigenvectors of the Hamiltonian.
??x
The eigenvalues and eigenvectors of the Hamiltonian \(H\) are found by solving:
\[ H \phi = E \phi \]
For example, for \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
The eigenvalues and eigenvectors are:
\[ \phi_1 = |01\rangle + |10\rangle, \quad E_1 = 0 \]
\[ \phi_2 = |11\rangle, \quad E_2 = -\mu^2 r^{-3} \]
The eigenvalue \(E_1 = 0\) corresponds to the state where there is no energy difference, and \(E_2 = -\mu^2 r^{-3}\) represents the non-zero energy level.
??x

--- 
#### Separable States vs. Entangled States
The separable states are those that can be expressed as a direct product of individual qubit states, while entangled states cannot.

:p Differentiate between separable and entangled states.
??x
A separable state is one where the overall state of the system can be factored into separate parts corresponding to individual subsystems. For example:
\[ |0_A\rangle \otimes |0_B\rangle = |00\rangle \]
In contrast, an entangled state involves correlations between different qubits that cannot be separated by any tensor product of individual states. An example is the Bell state \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
This cannot be written as a simple direct product of individual qubit states.
??x

--- 
#### Direct Product and Tensor Operations
The direct product operation is used to combine operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Explain how to compute the direct product of two Pauli matrices \(X_A\) and \(X_B\).
??x
To compute the direct product of two Pauli matrices \(X_A\) and \(X_B\):
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
This is done by placing each element of \(X_A\) in every position and multiplying it with the corresponding elements from \(X_B\):
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
??x

--- 
#### Bell States and Entanglement
Bell states, such as \(|\beta_{00}\rangle\), are entangled states that cannot be factored into individual qubit states.

:p Explain why the Bell state \(|\beta_{00}\rangle\) is entangled.
??x
The Bell state \(|\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)\) is entangled because it cannot be written as a tensor product of individual states. Trying to separate it into:
\[ |\beta_{00}\rangle = |u_1\rangle \otimes |v_1\rangle + |u_2\rangle \otimes |v_2\rangle \]
Leads to contradictions, indicating that the state is inherently entangled.
??x

--- 
#### Density Matrix and Ensemble States
The density matrix can represent both pure and mixed states of a quantum system.

:p Explain how to construct a density matrix for an ensemble of states.
??x
To construct a density matrix \(\rho\) for an ensemble of states, where the probability distribution over the states is given by \(p_i\):
\[ \rho = \sum_i p_i | \psi_i \rangle \langle \psi_i| \]
For example, if we have two pure states \(| \psi_1 \rangle\) and \(| \psi_2 \rangle\) with probabilities \(p_1 = 0.6\) and \(p_2 = 0.4\), the density matrix is:
\[ \rho = p_1 | \psi_1 \rangle \langle \psi_1 | + p_2 | \psi_2 \rangle \langle \psi_2 | \]
This representation helps in calculating observables and probabilities without explicitly using wavefunctions.
??x

--- 
#### Interaction Hamiltonian of Dipoles
The interaction between two dipoles can be described by the Hamiltonian, which includes terms for direct interactions and directional effects.

:p Derive the simplified form of the dipole interaction Hamiltonian when \(\hat{r} = \hat{k}\).
??x
Given the interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (\sigma_A \cdot \sigma_B - 3 \sigma_A \cdot \hat{r} \sigma_B \cdot \hat{r}) \]
When \(\hat{r} = \hat{k}\), the dot products simplify to:
\[ \sigma_A \cdot \hat{r} = Z_A, \quad \sigma_B \cdot \hat{r} = Z_B \]
Thus, the Hamiltonian becomes:
\[ H = \mu^2 r^{-3} (X_A X_B + Y_A Y_B + Z_A Z_B - 3 Z_A Z_B) \]
This can be rewritten as:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
??x

--- 
#### Pauli Matrices and Tensor Products
The Pauli matrices are fundamental operators in quantum mechanics, and their tensor products represent combined effects on multiple qubits.

:p Compute the direct product of \(X\) with itself.
??x
To compute the direct product of the Pauli matrix \(X\) with itself:
\[ X \otimes X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
??x

--- 
#### Separable and Entangled States in Quantum Mechanics
Separable states can be written as a tensor product of individual qubit states, while entangled states cannot.

:p Differentiate separable and entangled states.
??x
A separable state is one where the overall state \(|\psi\rangle\) can be factored into separate parts corresponding to individual subsystems. For example:
\[ |0_A\rangle \otimes |0_B\rangle = |00\rangle \]
In contrast, an entangled state involves correlations between different qubits that cannot be separated by any tensor product of individual states. An example is the Bell state \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
This cannot be written as a simple direct product of individual qubit states.
??x

--- 
#### Hamiltonian and Energy Levels
The Hamiltonian describes the total energy of the system, and its eigenvalues give the possible energy levels.

:p Determine the eigenvalues and eigenvectors of the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues and eigenvectors of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
The eigenvalues \(E\) and eigenvectors \(\phi\) are found by solving the equation:
\[ H \phi = E \phi \]
For example, for \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
The eigenvalues and eigenvectors are:
\[ \phi_1 = |01\rangle + |10\rangle, \quad E_1 = 0 \]
\[ \phi_2 = |11\rangle, \quad E_2 = -\mu^2 r^{-3} \]
The eigenvalue \(E_1 = 0\) corresponds to the state where there is no energy difference, and \(E_2 = -\mu^2 r^{-3}\) represents the non-zero energy level.
??x

--- 
#### Separable States and Tensor Products
Separable states can be represented as tensor products of individual qubit states.

:p Explain how separable states are formed from individual qubits.
??x
A separable state is one where the overall state \(|\psi\rangle\) can be factored into separate parts corresponding to individual subsystems. For example, if we have two qubits, a separable state would be:
\[ |\psi\rangle = |0_A\rangle \otimes |0_B\rangle = |00\rangle \]
In this case, the state \(|00\rangle\) is formed by taking the tensor product of individual states \(|0_A\rangle\) and \(|0_B\rangle\).
??x

--- 
#### Entangled States and Tensor Products
Entangled states cannot be represented as tensor products of individual qubit states.

:p Explain why entangled states are not separable.
??x
An entangled state involves correlations between different qubits that cannot be separated by any tensor product of individual states. For example, the Bell state \(|\beta_{00}\rangle\) is entangled:
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
Trying to separate this into a form where it can be written as:
\[ |\psi\rangle = |u_1\rangle \otimes |v_1\rangle + |u_2\rangle \otimes |v_2\rangle \]
Leads to contradictions, indicating that the state is inherently entangled and cannot be factored into individual qubit states.
??x

--- 
#### Hamiltonian Eigenvalues and Energy Levels
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Determine the eigenvalues for the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
The possible energy levels are found by solving:
\[ H |\psi\rangle = E |\psi\rangle \]
For example, the eigenvalues for \(|\beta_{00}\rangle\) are:
\[ E_1 = 0 \quad \text{(for } |01\rangle + |10\rangle\text{)} \]
\[ E_2 = -\mu^2 r^{-3} \quad \text{(for } |11\rangle\text{)} \]
These eigenvalues represent the different energy states of the system.
??x

--- 
#### Tensor Products and Quantum Mechanics
Tensor products are used to combine operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Explain how tensor products are computed for Pauli matrices.
??x
To compute the tensor product of two Pauli matrices, say \(X_A\) and \(X_B\):
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
This is done by placing each element of \(X_A\) in every position and multiplying it with the corresponding elements from \(X_B\):
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
Thus, the tensor product results in a \(4 \times 4\) matrix.
??x

--- 
#### Quantum States and Tensor Products
Quantum states can be represented as tensor products of individual qubits.

:p Explain how to represent a quantum state using tensor products.
??x
A quantum state can be represented using tensor products by combining the states of multiple subsystems. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |01\rangle \]
This tensor product indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\).
??x

--- 
#### Eigenvalues of a Hamiltonian
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Determine the eigenvalues for the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
We solve the equation \(H |\psi\rangle = E |\psi\rangle\). For example, for the Bell state \(|\beta_{00}\rangle\):
\[ H |\beta_{00}\rangle = \frac{1}{\sqrt{2}} (|00\rangle - |11\rangle) \]
The eigenvalues are:
\[ E_1 = 0 \quad \text{(for } |01\rangle + |10\rangle\text{)} \]
\[ E_2 = -\mu^2 r^{-3} \quad \text{(for } |11\rangle\text{)} \]
These eigenvalues represent the energy levels of the system.
??x

--- 
#### Separable vs. Entangled States
Separable states can be written as tensor products, while entangled states cannot.

:p Explain the difference between separable and entangled states.
??x
A separable state is one where the overall state \(|\psi\rangle\) can be factored into separate parts corresponding to individual subsystems:
\[ |\psi\rangle = |0_A\rangle \otimes |1_B\rangle \]
In contrast, an entangled state involves correlations between different qubits that cannot be separated by any tensor product of individual states. For example, the Bell state \(|\beta_{00}\rangle\):
\[ |\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \]
Cannot be written as a simple direct product of individual qubit states.
??x

--- 
#### Tensor Product and Quantum Mechanics
Tensor products are used to combine operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Explain how tensor products are used in quantum mechanics.
??x
In quantum mechanics, tensor products are used to combine operators from separate Hilbert spaces. For example, if we have two qubits and their respective Pauli matrices \(X_A\) and \(X_B\), the tensor product of these operators is:
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
This results in a \(4 \times 4\) matrix:
\[ X_A \otimes X_B = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \]
This tensor product operator acts on a combined state of the two qubits.
??x

--- 
#### Quantum State Representation
Quantum states can be represented using tensor products of individual qubits.

:p Explain how to represent a quantum state as a tensor product.
??x
A quantum state in multiple qubits can be represented as a tensor product of their individual states. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |01\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\).
??x

--- 
#### Tensor Product of Pauli Matrices
The tensor product of Pauli matrices combines operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Compute the tensor product of two Pauli X matrices.
??x
To compute the tensor product of two Pauli \(X\) matrices, we take:
\[ X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
The tensor product \(X \otimes X\) is given by:
\[ X \otimes X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} & 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \\ 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} & 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix} \]
Thus, the tensor product \(X \otimes X\) is:
\[ X \otimes X = \begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix} \]
??x

--- 
#### Quantum State Tensor Product
Quantum states can be represented as tensor products of individual qubits.

:p Explain how to form a tensor product of two quantum states.
??x
To form the tensor product of two quantum states, we combine their respective vectors. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = |00\rangle + |11\rangle \]
??x

--- 
#### Hamiltonian Eigenvalues
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Determine the eigenvalues for the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
We need to solve the equation \(H |\psi\rangle = E |\psi\rangle\). For example, for the Bell state \(|\beta_{00}\rangle\):
\[ H |\beta_{00}\rangle = \frac{1}{\sqrt{2}} (|00\rangle - |11\rangle) \]
The eigenvalues are:
\[ E_1 = 0 \quad \text{(for } |01\rangle + |10\rangle\text{)} \]
\[ E_2 = -\mu^2 r^{-3} \quad \text{(for } |11\rangle\text{)} \]
Thus, the eigenvalues of the Hamiltonian are \(0\) and \(-\mu^2 r^{-3}\).
??x

--- 
#### Separable States in Quantum Mechanics
Separable states can be represented as tensor products of individual qubits.

:p Explain how to represent a separable state using tensor products.
??x
A separable state is one where the overall state \(|\psi\rangle\) can be factored into separate parts corresponding to individual subsystems. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state as a tensor product is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = |00\rangle + |11\rangle \]
??x

--- 
#### Tensor Product of Pauli Matrices
The tensor product of Pauli matrices combines operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Compute the tensor product of two Pauli Z matrices.
??x
To compute the tensor product of two Pauli \(Z\) matrices, we take:
\[ Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \]
The tensor product \(Z \otimes Z\) is given by:
\[ Z \otimes Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} = \begin{bmatrix} 1 \cdot \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} & 0 \cdot \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \\ 0 \cdot \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} & (-1) \cdot \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \]
Thus, the tensor product \(Z \otimes Z\) is:
\[ Z \otimes Z = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \]
??x

--- 
#### Quantum State Representation
Quantum states can be represented using tensor products of individual qubits.

:p Explain how to represent a quantum state as a tensor product.
??x
A quantum state in multiple qubits can be represented as a tensor product of their individual states. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = |00\rangle + |11\rangle \]
??x

--- 
#### Tensor Product of Pauli Matrices
The tensor product of Pauli matrices combines operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Compute the tensor product of two Pauli X and Y matrices.
??x
To compute the tensor product of two Pauli \(X\) and \(Y\) matrices, we take:
\[ X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \]
The tensor product \(X \otimes Y\) is given by:
\[ X \otimes Y = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} = \begin{bmatrix} 0 \cdot \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} & 1 \cdot \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \\ 1 \cdot \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} & 0 \cdot \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 0 & 0 & -i & 0 \\ 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \end{bmatrix} \]
Thus, the tensor product \(X \otimes Y\) is:
\[ X \otimes Y = \begin{bmatrix} 0 & 0 & -i & 0 \\ 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \end{bmatrix} \]
??x

--- 
#### Quantum State Tensor Product
Quantum states can be represented as tensor products of individual qubits.

:p Explain how to form the tensor product of two quantum states.
??x
To form the tensor product of two quantum states, we combine their respective vectors. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = |00\rangle + |11\rangle \]
??x

--- 
#### Hamiltonian Eigenvalues
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Determine the eigenvalues for the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
We need to solve the equation \(H |\psi\rangle = E |\psi\rangle\). For example, for a specific state like the Bell state \(|\beta_{00}\rangle\):
\[ H |\beta_{00}\rangle = \frac{1}{\sqrt{2}} (|00\rangle - |11\rangle) \]
The eigenvalues are determined by the form of the Hamiltonian and the states it acts on. For this specific Hamiltonian, the eigenstates can be identified as:
\[ E_1 = 0 \quad \text{(for } |01\rangle + |10\rangle\text{)} \]
\[ E_2 = -\mu^2 r^{-3} \quad \text{(for } |11\rangle\text{)} \]
Thus, the eigenvalues of the Hamiltonian are \(0\) and \(-\mu^2 r^{-3}\).
??x

--- 
#### Quantum State Representation
Quantum states can be represented using tensor products of individual qubits.

:p Explain how to represent a separable state as a product of individual states.
??x
A separable state in quantum mechanics is one that can be expressed as a direct product (tensor product) of the individual states of each subsystem. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state as a separable state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = |00\rangle + |11\rangle \]
??x

--- 
#### Tensor Product of Pauli Matrices
The tensor product of Pauli matrices combines operators from separate Hilbert spaces into a single operator acting on the combined space.

:p Compute the tensor product of two Pauli Z and X matrices.
??x
To compute the tensor product of two Pauli \(Z\) and \(X\) matrices, we take:
\[ Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]
The tensor product \(Z \otimes X\) is given by:
\[ Z \otimes X = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} & 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \\ 0 \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} & (-1) \cdot \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & -1 \\ 0 & 0 & -1 & 0 \end{bmatrix} \]
Thus, the tensor product \(Z \otimes X\) is:
\[ Z \otimes X = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & -1 \\ 0 & 0 & -1 & 0 \end{bmatrix} \]
??x

--- 
#### Quantum State Tensor Product
Quantum states can be represented as tensor products of individual qubits.

:p Explain how to form the tensor product of two quantum states.
??x
To form the tensor product of two quantum states, we combine their respective vectors. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = (|0\rangle \oplus |A\rangle) \otimes (|1\rangle \oplus |B\rangle) \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} \]
So the combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]
??x

--- 
#### Hamiltonian Eigenvalues
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Determine the eigenvalues for the simplified dipole interaction Hamiltonian.
??x
To determine the eigenvalues of the simplified dipole interaction Hamiltonian:
\[ H = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \]
we need to analyze the action of this Hamiltonian on specific states.

1. **Identify Eigenstates:**
   The Hamiltonian involves products of Pauli matrices \(X\), \(Y\), and \(Z\) acting on different qubits. We can consider the eigenstates of these Pauli matrices:
   - \(X_A \otimes X_B\) has eigenvalues \(\pm 1 \pm 1 = \pm 2\) and \(\pm 1 - 1 = 0\).
   - \(Y_A \otimes Y_B\) has eigenvalues \(\pm i \pm i = \pm 2i\) and \(\pm i - i = 0\).
   - \(Z_A \otimes Z_B\) has eigenvalues \(\pm 1 \pm 1 = \pm 2\).

2. **Possible States:**
   The states can be written as combinations of the Pauli matrices:
   - For \(X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B\), we need to consider the eigenstates that satisfy the equation.

3. **Specific State Analysis:**
   Consider the Bell state \(|\beta_{00}\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)\):
   - For \(X_A \otimes X_B\), \(|00\rangle\) and \(|11\rangle\) are eigenstates with eigenvalue 0.
   - For \(Y_A \otimes Y_B\), \(|00\rangle\) and \(|11\rangle\) are eigenstates with eigenvalue 0.
   - For \(Z_A \otimes Z_B\), \(|00\rangle\) is an eigenstate with eigenvalue 2, and \(|11\rangle\) is an eigenstate with eigenvalue -2.

4. **Eigenvalues Calculation:**
   The Hamiltonian for the Bell state:
   \[ H |\beta_{00}\rangle = \mu^2 r^{-3} (X_A \otimes X_B + Y_A \otimes Y_B - 2 Z_A \otimes Z_B) \left( \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle) \right) = \mu^2 r^{-3} \left( 0 + 0 - 2 \cdot \frac{1}{\sqrt{2}} (|00\rangle - |11\rangle) \right) \]
   Simplifying, we get:
   \[ H |\beta_{00}\rangle = \mu^2 r^{-3} (-\sqrt{2}) (|00\rangle - |11\rangle) = -\mu^2 r^{-3} \sqrt{2} |\beta_{00}\rangle \]
   Therefore, the eigenvalue for this state is:
   \[ E = -\mu^2 r^{-3} \]

5. **General Eigenvalues:**
   The Hamiltonian also has other states that can be analyzed similarly. For example, \(|\beta_{11}\rangle = \frac{1}{\sqrt{2}} (|01\rangle + |10\rangle)\) is an eigenstate with eigenvalue 0:
   - For \(X_A \otimes X_B\), \(|01\rangle\) and \(|10\rangle\) are eigenstates with eigenvalue 0.
   - For \(Y_A \otimes Y_B\), \(|01\rangle\) and \(|10\rangle\) are eigenstates with eigenvalue 0.
   - For \(Z_A \otimes Z_B\), \(|01\rangle\) and \(|10\rangle\) are eigenstates with eigenvalue 0.

Thus, the eigenvalues of the Hamiltonian are:
\[ E_1 = 0 \quad \text{(for } |01\rangle + |10\rangle\text{)} \]
\[ E_2 = -\mu^2 r^{-3} \quad \text{(for } |00\rangle - |11\rangle\text{)} \]

The final eigenvalues are:
\[ \boxed{0 \text{ and } -\mu^2 r^{-3}} \]
??x

--- 
#### Quantum State Representation
Quantum states can be represented using tensor products of individual qubits.

:p Write the tensor product of two quantum states.
??x
To write the tensor product of two quantum states, we combine their respective vectors. For example, if we have two qubits with states \(|0_A\rangle\) and \(|1_B\rangle\), their combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = (|0\rangle \oplus |A\rangle) \otimes (|1\rangle \oplus |B\rangle) \]
This indicates that the first qubit is in state \(|0\rangle\) and the second qubit is in state \(|1\rangle\). In vector form, this can be written as:
\[ |0_A\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad |1_B\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \]
Thus,
\[ |0_A\rangle \otimes |1_B\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} \]
So the combined state is:
\[ |0_A\rangle \otimes |1_B\rangle = |00\rangle + |11\rangle \]

In general, for two states \(|\psi_1\rangle\) and \(|\psi_2\rangle\), their tensor product is given by:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = (a_0|0\rangle + a_1|1\rangle) \otimes (b_0|0\rangle + b_1|1\rangle) = a_0b_0|00\rangle + a_0b_1|01\rangle + a_1b_0|10\rangle + a_1b_1|11\rangle \]

For example, if \(|\psi_1\rangle = |+\rangle\) and \(|\psi_2\rangle = |-\rangle\), where:
\[ |+\rangle = \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle) \]
\[ |-\rangle = \frac{1}{\sqrt{2}} (|0\rangle - |1\rangle) \]

Then,
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \left( \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle) \right) \otimes \left( \frac{1}{\sqrt{2}} (|0\rangle - |1\rangle) \right) = \frac{1}{2} (|0\rangle + |1\rangle) \otimes (|0\rangle - |1\rangle) \]
\[ = \frac{1}{2} (|00\rangle - |01\rangle + |10\rangle - |11\rangle) \]

The final tensor product state is:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \frac{1}{2} (|00\rangle - |01\rangle + |10\rangle - |11\rangle) \]
??x

--- 
#### Hamiltonian Eigenvalues
The eigenvalues of a Hamiltonian give the possible energy levels of the system.

:p Calculate the eigenvalues for a simple 2-qubit Hamiltonian.
??x
To calculate the eigenvalues for a simple 2-qubit Hamiltonian, we can consider a specific form of the Hamiltonian and find its eigenstates and corresponding eigenvalues. Let's use the following Hamiltonian as an example:
\[ H = X_A \otimes Z_B + Z_A \otimes X_B \]
where \(X\) and \(Z\) are Pauli matrices acting on different qubits.

1. **Pauli Matrices:**
   The Pauli matrices are:
   \[ X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \]

2. **Hamiltonian Matrix:**
   We need to construct the matrix representation of \(H\). The tensor product \(X_A \otimes Z_B\) and \(Z_A \otimes X_B\) can be written as:
   \[ X_A \otimes Z_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} = \begin{bmatrix} 0 \cdot I_2 & 1 \cdot (-I_2) \\ 1 \cdot I_2 & 0 \cdot (-I_2) \end{bmatrix} = \begin{bmatrix} 0 & -Z_2 & Z_2 & 0 \\ Z_2 & 0 & 0 & -Z_2 \end{bmatrix} \]
   where \(I_2\) is the 2x2 identity matrix and \(Z_2 = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\).

3. **Complete Hamiltonian:**
   Similarly, for \(Z_A \otimes X_B\):
   \[ Z_A \otimes X_B = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \otimes \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 \cdot I_2 & 0 \cdot (-I_2) \\ 0 \cdot I_2 & -1 \cdot (-I_2) \end{bmatrix} = \begin{bmatrix} Z_2 & 0 & 0 & -Z_2 \\ 0 & -Z_2 & Z_2 & 0 \end{bmatrix} \]

4. **Combining Hamiltonians:**
   The full Hamiltonian is:
   \[ H = X_A \otimes Z_B + Z_A \otimes X_B = \begin{bmatrix} 0 & -Z_2 & Z_2 & 0 \\ Z_2 & 0 & 0 & -Z_2 \\ Z_2 & 0 & 0 & -Z_2 \\ 0 & -Z_2 & Z_2 & 0 \end{bmatrix} + \begin{bmatrix} Z_2 & 0 & 0 & -Z_2 \\ 0 & -Z_2 & Z_2 & 0 \\ 0 & -Z_2 & Z_2 & 0 \\ -Z_2 & 0 & 0 & Z_2 \end{bmatrix} = \begin{bmatrix} Z_2 & -Z_2 & Z_2 & -Z_2 \\ Z_2 & -Z_2 & Z_2 & -Z_2 \\ Z_2 & -Z_2 & Z_2 & -Z_2 \\ -Z_2 & Z_2 & -Z_2 & Z_2 \end{bmatrix} \]

5. **Eigenvalues Calculation:**
   To find the eigenvalues, we solve the characteristic equation:
   \[ \det(H - E I) = 0 \]
   where \(E\) is an eigenvalue and \(I\) is the identity matrix.

6. **Simplifying the Hamiltonian:**
   Notice that \(H\) can be written in a block diagonal form by considering the Pauli matrices' properties:
   \[ H = \begin{bmatrix} Z_2 & -Z_2 \\ -Z_2 & Z_2 \end{bmatrix} \otimes I_2 + I_2 \otimes \begin{bmatrix} 0 & Z_2 \\ Z_2 & 0 \end{bmatrix} \]

7. **Eigenvalues of Simplified Hamiltonian:**
   The eigenvalues of \(H\) can be found by solving the characteristic polynomial:
   \[ \det(H - E I) = \left( E^2 - (Z_2)^2 \right) \left( E^2 - Z_2^2 \right) = 0 \]
   Thus, the eigenvalues are:
   \[ E = \pm \sqrt{2}, \pm \sqrt{2} \]

8. **Final Eigenvalues:**
   The eigenvalues of the Hamiltonian \(H\) are:
   \[ \boxed{\pm \sqrt{2}, \pm \sqrt{2}} \]
??x

--- 
#### Tensor Product of Quantum States
The tensor product of two quantum states combines their respective vectors.

:p Calculate the tensor product of two specific 2-qubit states.
??x
To calculate the tensor product of two specific 2-qubit states, we will consider the following states:

1. State \( |\psi_1\rangle = \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle) \)
2. State \( |\psi_2\rangle = \frac{1}{\sqrt{2}} (|0\rangle - |1\rangle) \)

We need to find the tensor product \( |\psi_1\rangle \otimes |\psi_2\rangle \).

First, write down the states in vector form:
\[ |\psi_1\rangle = \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \]
\[ |\psi_2\rangle = \frac{1}{\sqrt{2}} (|0\rangle - |1\rangle) = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix} \]

The tensor product \( |\psi_1\rangle \otimes |\psi_2\rangle \) is given by:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \left( \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right) \otimes \left( \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix} \right) \]

Using the definition of the tensor product:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \left( |0\rangle + |1\rangle \right) \otimes (|0\rangle - |1\rangle) \]
\[ = \frac{1}{2} \left( |0\rangle \otimes (|0\rangle - |1\rangle) + |1\rangle \otimes (|0\rangle - |1\rangle) \right) \]

Now, compute each term:
\[ |0\rangle \otimes (|0\rangle - |1\rangle) = |00\rangle - |01\rangle \]
\[ |1\rangle \otimes (|0\rangle - |1\rangle) = |10\rangle - |11\rangle \]

Combining these results:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \frac{1}{2} \left( |00\rangle - |01\rangle + |10\rangle - |11\rangle \right) \]

Thus, the tensor product of the two specific 2-qubit states is:
\[ |\psi_1\rangle \otimes |\psi_2\rangle = \frac{1}{2} (|00\rangle - |01\rangle + |10\rangle - |11\rangle) \]

The final tensor product state is:
\[ \boxed{\frac{1}{2} (|00\rangle - |01\rangle + |10\rangle - |11\rangle)} \]
??x

--- 
#### Tensor Product of Pauli Matrices
The tensor product of Pauli matrices combines their respective actions on different qubits.

:p Calculate the tensor product of two specific Pauli matrices acting on different qubits.
??x
To calculate the tensor product of two specific Pauli matrices acting on different qubits, let's consider the following:

1. The Pauli matrix \(X\) (also known as the Pauli-X matrix):
   \[ X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \]

2. The Pauli matrix \(Z\) (also known as the Pauli-Z matrix):
   \[ Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \]

We need to find the tensor product \( X_A \otimes Z_B \), where \( A \) and \( B \) represent the qubits on which the matrices act.

The tensor product of two matrices is given by:
\[ (A \otimes B)_{ij,kl} = A_{ik} B_{jl} \]

Let's compute the tensor product step-by-step:

1. **Identify the dimensions:**
   - The Pauli matrix \( X \) has dimensions \( 2 \times 2 \).
   - The Pauli matrix \( Z \) also has dimensions \( 2 \times 2 \).

2. **Form the tensor product:**
   \[ X_A \otimes Z_B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \]

3. **Compute each element:**
   \[
   (X_A \otimes Z_B)_{ij,kl} = X_{ik} Z_{jl}
   \]
   where \( i, j, k, l \) are indices that range from 1 to 2.

4. **Calculate the elements:**
   - For \( (i,j,k,l) = (1,1,1,1) \):
     \[
     (X_A \otimes Z_B)_{11,11} = X_{11} Z_{11} = 0 \cdot 1 = 0
     \]
   - For \( (i,j,k,l) = (1,1,1,2) \):
     \[
     (X_A \otimes Z_B)_{11,12} = X_{11} Z_{12} = 0 \cdot 0 = 0
     \]
   - For \( (i,j,k,l) = (1,1,2,1) \):
     \[
     (X_A \otimes Z_B)_{11,21} = X_{12} Z_{11} = 1 \cdot 1 = 1
     \]
   - For \( (i,j,k,l) = (1,1,2,2) \):
     \[
     (X_A \otimes Z_B)_{11,22} = X_{12} Z_{12} = 1 \cdot 0 = 0
     \]
   - For \( (i,j,k,l) = (1,2,1,1) \):
     \[
     (X_A \otimes Z_B)_{12,11} = X_{11} Z_{21} = 0 \cdot 0 = 0
     \]
   - For \( (i,j,k,l) = (1,2,1,2) \):
     \[
     (X_A \otimes Z_B)_{12,12} = X_{11} Z_{22} = 0 \cdot (-1) = 0
     \]
   - For \( (i,j,k,l) = (1,2,2,1) \):
     \[
     (X_A \otimes Z_B)_{12,21} = X_{12} Z_{21} = 1 \cdot 0 = 0
     \]
   - For \( (i,j,k,l) = (1,2,2,2) \):
     \[
     (X_A \otimes Z_B)_{12,22} = X_{12} Z_{22} = 1 \cdot (-1) = -1
     \]

5. **Construct the resulting matrix:**
   \[
   X_A \otimes Z_B = \begin{bmatrix}
   0 & 0 \\
   1 & -1
   \end{bmatrix}
   \]

6. **Extend to a \(4 \times 4\) matrix:**
   The full tensor product is:
   \[
   X_A \otimes Z_B = \begin{bmatrix}
   0 & 0 & 0 & 0 \\
   1 & 0 & 0 & 0 \\
   0 & 0 & 0 & -1 \\
   0 & 1 & -1 & 0
   \end{bmatrix}
   \]

The final tensor product of the Pauli matrices is:
\[
\boxed{\begin{bmatrix}
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & -1 \\
0 & 1 & -1 & 0
\end{bmatrix}}
\]

#### Quantum State Separability and Entanglement

Background context: In quantum computing, understanding whether a state is separable or entangled is crucial. A separable state can be expressed as a tensor product of individual qubit states, while an entangled state cannot.

:p Which eigenstates are separable and which ones are entangled?
??x
The eigenstate |00⟩ (𝜙3) is separable because it can be written as the tensor product of two single-qubit states: |00⟩ = |0⟩ ⊗ |0⟩. The other eigenstates, |01⟩ - |10⟩ (𝜙4), are entangled and cannot be expressed in this form.
??x
The answer with detailed explanations:
- **Separable State**: A state that can be written as a tensor product of individual qubit states. For example, |00⟩ = |0⟩ ⊗ |0⟩.
- **Entangled State**: A state that cannot be written as a tensor product of individual qubit states.

For the given eigenstates:
- |00⟩ (𝜙3) is separable because it can be written as \( \phi_3 = |0⟩ \otimes |0⟩ \).
- The states |01⟩ - |10⟩ (𝜙4) are entangled and cannot be expressed in the form of a tensor product of individual qubits.
??x
---

#### Hamiltonian Matrix Evaluation

Background context: To evaluate the Hamiltonian matrix, we use the eigenstates obtained from the previous section. The Hamiltonian is diagonalized using these eigenstates as basis vectors.

:p Use the eigenvectors to calculate the Hamiltonian matrix \(H\).
??x
The answer with detailed explanations:
To construct the Hamiltonian matrix in terms of the given eigenvectors, we use the formula:

\[ H = \begin{bmatrix}
    \langle \phi_1 | H | \phi_1 \rangle & \langle \phi_1 | H | \phi_2 \rangle & \cdots \\
    \langle \phi_2 | H | \phi_1 \rangle & \langle \phi_2 | H | \phi_2 \rangle & \cdots \\
    \vdots & \vdots & \ddots
\end{bmatrix} \]

Given the eigenvectors:
- \( \phi_1 = |01⟩ - |10⟩ / √2 \)
- \( \phi_2, \phi_3, \phi_4 \) as defined

Using Python and NumPy (as per the provided example), this can be done programmatically. The Hamiltonian matrix obtained should be diagonal with eigenvalues on the diagonal.

Here is a simplified version of how you might set up such a calculation in Python:

```python
import numpy as np

# Define eigenvectors
phi1 = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
phi2 = np.array([0, 0, 0, 1])
phi3 = np.array([1, 0, 0, 0])
phi4 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])

# Define the Hamiltonian (as a placeholder)
H = np.array([
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
])

# Construct the Hamiltonian matrix using the inner products
H_matrix = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        H_matrix[i][j] = (phi1.conj().T @ H @ phi2)[0]

print("Hamiltonian Matrix:")
print(H_matrix)
```

This code constructs the Hamiltonian matrix using inner products of the eigenvectors with the Hamiltonian. The actual values depend on the specific form of \(H\).

Note: Replace the placeholder Hamiltonian with the actual one in your context.
??x
---

#### Controlled NOT (CNOT) Gate

Background context: CNOT is a two-qubit gate that flips the target qubit if the control qubit is 1. It's used to create entanglement and perform various operations on multiple qubits.

:p What is the effect of the CNOT gate on |10⟩, |01⟩, |00⟩, and |11⟩?
??x
The answer with detailed explanations:
- For \(|10\rangle\):
  \[ \text{CNOT}(|10\rangle) = \begin{bmatrix}
      1 & 0 & 0 & 0 \\
      0 & 0 & 0 & 1 \\
      0 & 0 & 1 & 0 \\
      0 & 1 & 0 & 0
    \end{bmatrix} \begin{bmatrix}
      0 \\
      1 \\
      0 \\
      0
    \end{bmatrix} = \begin{bmatrix}
      0 \\
      0 \\
      0 \\
      1
    \end{bmatrix} = |11\rangle \]

- For \(|01\rangle\):
  \[ \text{CNOT}(|01\rangle) = \begin{bmatrix}
      1 & 0 & 0 & 0 \\
      0 & 0 & 0 & 1 \\
      0 & 0 & 1 & 0 \\
      0 & 1 & 0 & 0
    \end{bmatrix} \begin{bmatrix}
      0 \\
      0 \\
      1 \\
      0
    \end{bmatrix} = \begin{bmatrix}
      0 \\
      1 \\
      1 \\
      0
    \end{bmatrix} = |01\rangle \]

- For \(|00\rangle\):
  \[ \text{CNOT}(|00\rangle) = \begin{bmatrix}
      1 & 0 & 0 & 0 \\
      0 & 0 & 0 & 1 \\
      0 & 0 & 1 & 0 \\
      0 & 1 & 0 & 0
    \end{bmatrix} \begin{bmatrix}
      0 \\
      0 \\
      0 \\
      1
    \end{bmatrix} = \begin{bmatrix}
      0 \\
      0 \\
      0 \\
      1
    \end{bmatrix} = |00\rangle \]

- For \(|11\rangle\):
  \[ \text{CNOT}(|11\rangle) = \begin{bmatrix}
      1 & 0 & 0 & 0 \\
      0 & 0 & 0 & 1 \\
      0 & 0 & 1 & 0 \\
      0 & 1 & 0 & 0
    \end{bmatrix} \begin{bmatrix}
      1 \\
      1 \\
      0 \\
      0
    \end{bmatrix} = \begin{bmatrix}
      1 \\
      0 \\
      0 \\
      0
    \end{bmatrix} = |10\rangle \]

The CNOT gate flips the second qubit if the first (control) qubit is 1.
??x
---

#### Controlled Z (CZ) Gate

Background context: The controlled Z gate applies a phase factor to the target qubit when the control qubit is in state |1⟩. This can be used to create entanglement and manipulate states.

:p What is the effect of the CZ gate on \(|00\rangle\), \(|01\rangle\), \(|10\rangle\), and \(|11\rangle\)?
??x
The answer with detailed explanations:
- For \(|00\rangle\):
  \[ \text{CZ}(|00\rangle) = |00⟩ \]

- For \(|01\rangle\):
  \[ \text{CZ}(|01\rangle) = |01⟩ \]

- For \(|10\rangle\):
  \[ \text{CZ}(|10\rangle) = |10⟩ \]

- For \(|11\rangle\):
  \[ \text{CZ}(|11\rangle) = -|11⟩ \]

The Controlled Z (CZ) gate applies a phase factor of -1 to the target qubit when the control qubit is in state |1⟩.
??x
---

#### Quantum Circuit for Creating Entangled States

Background context: Quantum circuits can create entangled states using gates like Hadamard and CNOT. The process involves applying these gates on specific qubits to achieve the desired entanglement.

:p How does the circuit create the Bell state \(|\beta_{00}\rangle\)?
??x
The answer with detailed explanations:
- Start with the initial state \(|00\rangle\).
- Apply Hadamard (H) gate on the first qubit: 
  \[ H|0⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) \]
  So, \(H|0⟩|0⟩ = \frac{1}{\sqrt{2}}(|0⟩|0⟩ + |1⟩|0⟩)\).

- Apply CNOT gate with the first qubit as control and second as target:
  - If the first qubit is 0: no change.
  - If the first qubit is 1: flip the second qubit.

Thus, the circuit transforms \(|00\rangle\) into \( \frac{1}{\sqrt{2}}(|00⟩ + |11⟩) \), which is the Bell state \(|\beta_{00}\rangle\).

The quantum circuit looks like this:

```java
public class Example {
    // Define Hadamard and CNOT gates
    public static void main(String[] args) {
        Complex[] h = {Complex.HALF_SQRT2, 0, 0, -Complex.HALF_SQRT2};
        Complex[] cnotControlledZero = {1, 0, 0, 0, 0, 0, 0, 1};
        Complex[] bellState = applyGate(h, |00⟩).applyGate(cnotControlledZero, |00⟩);

        System.out.println(bellState);
    }
}
```

In this circuit:
- Hadamard gate transforms the first qubit.
- CNOT gate entangles the two qubits.

The resulting state is \( \frac{1}{\sqrt{2}}(|00⟩ + |11⟩) \).
??x
--- 

--- 
(Note: The Java code examples are simplified and for illustration purposes only. They may need to be adjusted based on actual quantum computing libraries or frameworks being used.) 
---

#### Hadamard Gate Basics
Background context: The Hadamard gate is a single-qubit quantum gate that transforms the basis states of qubits. It can be represented as:
\[ H|0⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) \]
and 
\[ H|1⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩). \]

:p What is the Hadamard gate and how does it transform qubits?
??x
The Hadamard gate \(H\) transforms the basis states of a single qubit to superposition states. Specifically, it maps \(|0⟩\) to an equal superposition state:
\[ H|0⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩), \]
and
\[ H|1⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩). \]
This transformation is crucial for creating entangled states and superpositions.

x??

---

#### Two Hadamard Gates
Background context: Applying two Hadamard gates to a qubit should act as the identity operator, meaning that the state of the qubit remains unchanged after applying \(H\) twice.
:p What happens when you apply two Hadamard gates to a single qubit?
??x
Applying two Hadamard gates to a single qubit will effectively return it to its original state. This is because the Hadamard gate applied once transforms \(|0⟩\) to \(\frac{1}{\sqrt{2}}(|0⟩ + |1⟩)\) and \(|1⟩\) to \(\frac{1}{\sqrt{2}}(|0⟩ - |1⟩)\). Applying another Hadamard gate will transform these superposition states back to their original basis states.

For example:
\[ H(H|0⟩) = H\left( \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) \right) = \frac{1}{\sqrt{2}}H|0⟩ + \frac{1}{\sqrt{2}}H|1⟩ \]
\[ = \frac{1}{\sqrt{2}}\left( \frac{1}{\sqrt{2}}(|0⟩ - |1⟩) + \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) \right) = |0⟩. \]

x??

---

#### X and Hadamard Gates
Background context: The combination of the \(X\) gate (NOT gate) and the Hadamard gate is significant in creating eigenstates of Pauli-X operator. Applying \(X\) followed by \(H\) on a qubit initially in state \(|0⟩\) will result in:
\[ H(X|0⟩) = H|1⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩). \]

:p What is the effect of applying an X gate followed by a Hadamard gate to a qubit initially in state \(|0⟩\)?
??x
Applying an \(X\) gate on a qubit in the state \(|0⟩\) first flips it to \(|1⟩\), and then applying a Hadamard gate to this state will create an eigenstate of the Pauli-X operator:
\[ H(X|0⟩) = H|1⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩). \]
This is because \(H\) transforms \(|1⟩\) into a superposition where the probability amplitudes are equal and opposite, creating an eigenstate of Pauli-X.

x??

---

#### 3-Qubit Toffoli Gate
Background context: The Toffoli gate (also known as CCNOT) is a three-qubit quantum logic gate. It takes three bits as input and inverts the third bit if the first two bits are both 1's. This operation is reversible.

:p What does the Toffoli gate do when its control qubits are set to \(|1⟩\)?
??x
The Toffoli gate, or CCNOT (Control-Control-Not) gate, acts on three qubits where the first two qubits are controls and the third qubit is the target. If both control qubits are in the state \(|1⟩\), it flips the target qubit. Otherwise, the target qubit remains unchanged.

For example:
\[ \text{Toffoli}(|1⟩ ⊗ |1⟩ ⊗ |0⟩) = |1⟩ ⊗ |1⟩ ⊗ |1⟩ \]
and
\[ \text{Toffoli}(|0⟩ ⊗ |0⟩ ⊗ |0⟩) = |0⟩ ⊗ |0⟩ ⊗ |0⟩. \]

x??

---

#### 2-Qubit CNOT Gate
Background context: The CNOT (Controlled-NOT) gate is a two-qubit quantum logic gate. It flips the second qubit if the first qubit (control) is in state \(|1⟩\). This can be represented by:
\[ \text{CNOT}(|0⟩ ⊗ |0⟩) = |0⟩ ⊗ |0⟩, \]
\[ \text{CNOT}(|0⟩ ⊗ |1⟩) = |0⟩ ⊗ |1⟩, \]
\[ \text{CNOT}(|1⟩ ⊗ |0⟩) = |1⟩ ⊗ |1⟩, \]
\[ \text{CNOT}(|1⟩ ⊗ |1⟩) = |1⟩ ⊗ 0. \]

:p What is the effect of the CNOT gate on two qubits?
??x
The CNOT (Controlled-NOT) gate applies a NOT operation to the target qubit if the control qubit is in the state \(|1⟩\). This can be represented as:
\[ \text{CNOT}(|0⟩ ⊗ |0⟩) = |0⟩ ⊗ |0⟩, \]
\[ \text{CNOT}(|0⟩ ⊗ |1⟩) = |0⟩ ⊗ |1⟩, \]
\[ \text{CNOT}(|1⟩ ⊗ |0⟩) = |1⟩ ⊗ |1⟩, \]
\[ \text{CNOT}(|1⟩ ⊗ |1⟩) = |1⟩ ⊗ 0. \]

x??

---

#### Cirq Circuit with SWAP Gate
Background context: In the provided code example, a circuit is created using Cirq to demonstrate the use of a SWAP gate between two qubits.

:p What does the CirqSwap.py program do?
??x
The CirqSwap.py program creates and simulates a 2-qubit quantum circuit that applies an X gate on \(q0\), a Z gate on \(q1\), and then swaps the states of \(q0\) and \(q1\) using the SWAP gate. After running the simulation, it prints the output vector.

The logic is as follows:
```python
# CirqSwap.py: Cirq program to create & swap 2 qubits
3import cirq

circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2) # Create two qubits
7circuit.append(cirq.X(q0)) # Append X to q0
circuit.append(cirq.Z(q1)) # Append Z to q1
circuit.append(cirq.SWAP(q0,q1)) # Swap qubits

s = cirq.Simulator() # Initialize simulator
print('Simulate the circuit:')

results = s.simulate(circuit) # Run simulator
print(results)
```

The output vector is:
```plaintext
output vector: |01>
```
This indicates that after applying the X gate on \(q0\) and Z gate on \(q1\), followed by a SWAP operation, the state of \(q0\) becomes \(|0⟩\) and the state of \(q1\) remains \(|1⟩\).

x??

--- 

#### 2-Qubit Cirq Circuit
Background context: The provided example demonstrates how to create and run a simple two-qubit circuit using Cirq.

:p How do you create a 2-qubit circuit in Cirq?
??x
To create a 2-qubit circuit in Cirq, you can use the `cirq.LineQubit.range(2)` method to define two qubits. Then, apply operations such as \(X\), \(Z\), and gates like CNOT.

The code example is:
```python
# CirqCNOT .py: Cirq program with CNOT gate
3import cirq

circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2) # Create two qubits
7circuit.append(cirq.X(q0)) # Append X to q0
circuit.append(cirq.Z(q1)) # Append Z to q1
circuit.append(cirq.CNOT(q0, q1)) # Append CNOT

s = cirq.Simulator() # Initialize simulator
print('Simulate the circuit:')

results = s.simulate(circuit) # Run simulator
print(results)
```

The output vector is:
```plaintext
output vector: |11>
```
This indicates that after applying an X gate on \(q0\), a Z gate on \(q1\), and then a CNOT operation, the state of both qubits becomes \(|1⟩\).

x?? 

---


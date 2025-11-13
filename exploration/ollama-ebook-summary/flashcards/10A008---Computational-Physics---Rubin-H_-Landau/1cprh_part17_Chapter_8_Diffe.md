# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 17)

**Starting Chapter:** Chapter 8 Differential Equations and Nonlinear Oscillations. 8.1 Nonlinear Oscillators

---

#### Nonlinear Oscillators Introduction
This section introduces the study of nonlinear oscillations, focusing on a mass attached to a spring with both a restoring force and an external time-dependent driving force. The motion is constrained to one dimension, and the system's equation of motion can be derived using Newton’s second law.
:p What is the key concept introduced in this section?
??x
The key concept introduced here is the study of nonlinear oscillators, specifically focusing on a mass attached to a spring with both a restoring force and an external driving force. The motion is governed by Newton's second law.
x??

---

#### Linear Spring Model
In the first model, the potential energy $V(x) \approx \frac{1}{2}kx^2 (1 - \frac{2}{3}\alpha x)$, where $ k$and $\alpha$ are constants. The restoring force is given by $F_k(x) = -\frac{dV(x)}{dx} = -kx(1-\alpha x)$. For small displacements, the motion is harmonic, but as $ x \to \frac{1}{\alpha}$, nonlinear effects increase.
:p What equation describes the restoring force in this model?
??x
The equation describing the restoring force in this model is $F_k(x) = -kx(1-\alpha x)$.
x??

---

#### General Nonlinear Oscillator Model
For a more general model, assume the potential energy function is proportional to an even power $p $ of$x $:$ V(x) = \frac{1}{p} kx^p $. The restoring force derived from this potential is$ F_k(x) = -\frac{dV(x)}{dx} = -kx^{p-1}$.
:p What is the general form of the potential energy for a nonlinear oscillator?
??x
The general form of the potential energy for a nonlinear oscillator is $V(x) = \frac{1}{p} kx^p $, where $ p$ is an even number.
x??

---

#### Harmonic Oscillator vs. Nonlinear Oscillator
When $p=2 $, we have the harmonic oscillator, while for higher values of $ p $, such as$ p=6 $, the potential resembles a square well, allowing the mass to move almost freely until it hits the walls at$ x \approx \pm 1 $. The motion is periodic but only harmonically so when$ p=2$.
:p How does the value of $p$ affect the behavior of the oscillator?
??x
The value of $p $ significantly affects the behavior of the oscillator. When$p=2 $, it behaves like a harmonic oscillator, exhibiting harmonic motion. For higher values of$ p $, such as$ p=6 $, the potential resembles a square well, and the motion becomes almost free until hitting the walls at$ x \approx \pm 1$.
x??

---

#### Equation of Motion
The equation of motion for the mass is given by: $m\frac{d^2x}{dt^2} = -kx(1-\alpha x) + F_{ext}(x,t)$. This equation can be simplified to $ m\frac{d^2x}{dt^2} = -kx(1-\alpha x)$, assuming no external force.
:p What is the equation of motion for the mass in this system?
??x
The equation of motion for the mass in this system, neglecting any external forces, is $m\frac{d^2x}{dt^2} = -kx(1-\alpha x)$.
x??

---

#### Nonlinear Effects and Motion Characteristics
For small displacements $x < \frac{1}{\alpha}$, the motion will be periodic but not necessarily harmonic. As the amplitude increases, the symmetry in the motion to the left and right of the equilibrium position becomes broken. For large values of $ x > \frac{1}{\alpha}$, the force becomes repulsive, pushing the mass away from the origin.
:p What happens as the displacement $x $ increases beyond$\frac{1}{\alpha}$?
??x
As the displacement $x $ increases beyond$\frac{1}{\alpha}$, the force acting on the mass becomes repulsive, causing it to be pushed away from the origin. The motion is no longer confined and can become unbound.
x??

---

#### Summary of Nonlinear Models
This section discusses two models: a linear spring model with anharmonic behavior for large displacements and a general potential energy function proportional to $x^p $. Both models exhibit periodic motion but only harmonically so when $ p=2$.
:p What are the two nonlinear models discussed in this section?
??x
The two nonlinear models discussed in this section are:
1. A linear spring model with anharmonic behavior for large displacements, given by $V(x) = \frac{1}{2}kx^2 (1 - \frac{2}{3}\alpha x)$.
2. A general potential energy function proportional to some even power $p $ of$x $, given by$ V(x) = \frac{1}{p} kx^p$.
x??

---

#### First-Order Ordinary Differential Equations (ODEs)
Background context: The general form of a first-order differential equation is given by $\frac{dy}{dt} = f(t, y)$. Here, the "order" refers to the degree of the derivative on the left-hand side (LHS). Even if the function $ f(t, y)$is complex, like in the example where $\frac{dy}{dt} = -3t^2y + t^9 + y^7$, it still qualifies as a first-order ODE.

:p What is the general form of a first-order differential equation?
??x
The general form of a first-order differential equation is $\frac{dy}{dt} = f(t, y)$. Here,$ f(t, y)$can be any function involving both time and position. For example, even if the right-hand side has complex terms like polynomials in $ t$,$ y$, or their combinations, it still counts as a first-order ODE.
x??

#### Second-Order Ordinary Differential Equations (ODEs)
Background context: The general form of a second-order differential equation is given by $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$. This form can include arbitrary functions on the right-hand side (RHS), which may involve any power of the first derivative. An example is $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = -3t^2\left(\frac{dy}{dt}\right)^4 + t^9 y(t)$. This equation is a second-order ODE, as seen in Newton's law.

:p What is an example of a second-order differential equation?
??x
An example of a second-order differential equation is $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = -3t^2\left(\frac{dy}{dt}\right)^4 + t^9 y(t)$. This equation includes the second derivative, first derivative, and the dependent variable on both sides of the equation.
x??

#### Order of a Differential Equation
Background context: The order of a differential equation refers to the highest derivative present in the equation. For instance, $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$ is a second-order ODE because it includes the second derivative of $ y $. In contrast,$\frac{dy}{dt} = f(t, y)$ is a first-order ODE as it only contains the first derivative.

:p What does the order of a differential equation refer to?
??x
The order of a differential equation refers to the highest derivative present in the equation. For example, in $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$, the second-order term indicates it is a second-order ODE, while in $\frac{dy}{dt} = f(t, y)$, the first derivative makes it a first-order ODE.
x??

#### Dependent and Independent Variables
Background context: In differential equations like $\frac{dy}{dt} = f(t, y)$ or $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$, time $ t$is the independent variable and position $ y$ is the dependent variable. This means we can vary $ t $ but not $ y $ directly at a specific $ t $. The symbol used for the dependent variable (e.g.,$ y$) is just a placeholder, which might refer to other variables depending on context.

:p What distinguishes independent and dependent variables in differential equations?
??x
In differential equations like $\frac{dy}{dt} = f(t, y)$, time $ t$is the independent variable because we can vary it freely. Position $ y$ is the dependent variable since its value depends on the value of $ t $. The symbol used for the dependent variable (e.g.,$ y$) is just a placeholder; in different contexts, it might refer to other variables.
x??

#### Ordinary vs Partial Differential Equations
Background context: First-order and second-order ODEs like $\frac{dy}{dt} = f(t, y)$ or $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$ are considered ordinary differential equations (ODEs) because they involve only one independent variable. In contrast, the Schrödinger equation, given by $i\hbar \frac{\partial \psi(x,t)}{\partial t} = -\frac{\hbar^2}{2m}\left( \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} + \frac{\partial^2 \psi}{\partial z^2} \right) + V(x)\psi(x,t)$, contains multiple independent variables ($ x, y, z, t$), making it a partial differential equation (PDE).

:p What distinguishes an ODE from a PDE?
??x
An ordinary differential equation (ODE) involves only one independent variable. For example, $\frac{dy}{dt} = f(t, y)$ or $\frac{d^2y}{dt^2} + \lambda \frac{dy}{dt} = f(t, \frac{dy}{dt}, y)$. In contrast, a partial differential equation (PDE) involves multiple independent variables. The Schrödinger equation is an example of a PDE because it includes spatial coordinates $ x, y, z$and time $ t$.
x??

---

#### Linear vs Nonlinear Equations

Linear equations are those where only the first power of $y $ or its derivative$\frac{dy}{dt}$ appears, while nonlinear equations can contain higher powers. For example:

- Linear:$\frac{dy}{dt} = g_3(t)y(t)$- Nonlinear:$\frac{dy}{dt} = \lambda y(t) - \lambda^2y^2(t)$ The law of linear superposition states that the sum of solutions is also a solution. For example, if $A(t)$ and $B(t)$ are solutions to a linear equation:
$$y(t) = \alpha A(t) + \beta B(t)$$is also a solution for arbitrary values of constants $\alpha $ and$\beta$.

Nonlinear equations do not have this property. Even if we guess that the solution is $y(t) = a(1 + be^{-\lambda t})$, adding two such solutions does not yield another valid solution.

:p What distinguishes linear from nonlinear differential equations?
??x
Linear differential equations are characterized by only containing the first power of the dependent variable and its derivative. Nonlinear equations can contain higher powers or products of the dependent variable and its derivatives.
x??

---

#### Initial and Boundary Conditions

For a first-order ODE, there is one arbitrary constant in the general solution, which is usually determined by an initial condition such as $y(t_0) = y_0 $. For a second-order ODE, two constants are present, typically determined by both position and velocity at some time $ t_0 $:$ y(t_0) = y_0 $and$\frac{dy}{dt}\bigg|_{t=t_0} = v_0$.

Boundary conditions further restrict the solutions to specific values at the boundaries of the solution space.

:p What do initial conditions specify for ODEs?
??x
Initial conditions specify the state of the system at a particular point in time. For example, $y(t_0) = y_0 $ and$\frac{dy}{dt}\bigg|_{t=t_0} = v_0$ determine the initial position and velocity for a second-order ODE.
x??

---

#### Dynamic Form of ODEs

To express an $N $-th order ODE as $ N$simultaneous first-order ODEs, we define new variables. For instance:
$$\frac{dy(0)}{dt} = f_0(t, \{y(i)\})$$
$$\frac{dy(1)}{dt} = f_1(t, \{y(i)\})$$

These can be represented compactly using vectors:
$$\frac{dy}{dt} = f(t, y)$$

Where $y $ and$f $ are $ N$-dimensional column vectors.

For Newton's law of motion, we convert the second-order ODE to a system of first-order ODEs by defining position as the first dependent variable and velocity as the second. This results in:

$$\frac{dy(0)}{dt} = y(1)$$
$$\frac{dy(1)}{dt} = \frac{F(t, x, \frac{dx}{dt})}{m}$$:p How do we convert a second-order ODE to a system of first-order ODEs?
??x
We introduce new variables where position $y(0)$ and velocity $y(1)$ represent the dependent variables. Then, we write the original second-order ODE as two coupled first-order ODEs:
```java
// Pseudocode for converting Newton's law to a system of first-order ODEs
public class Dynamics {
    public double[] dydt(double t, double[] y, double[] params) {
        double x = y[0]; // position
        double v = y[1]; // velocity
        
        double F = calculateForce(t, x, v); // force calculation based on F(x,t)
        
        return new double[]{v, (F / params[0])}; // derivatives dy(0)/dt and dy(1)/dt
    }
    
    private double calculateForce(double t, double x, double v) {
        // implementation of the force function F(t, x, v)
    }
}
```
x??

---


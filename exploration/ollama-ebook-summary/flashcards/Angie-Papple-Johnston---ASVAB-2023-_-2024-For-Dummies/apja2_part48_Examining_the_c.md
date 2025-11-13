# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 48)

**Starting Chapter:** Examining the current of the electrical river. Resistance Putting the brakes on electrical flow. Ohms law Relating resistance to current and voltage

---

---
#### Valence Shell and Electron Flow
Background context: The valence shell of an atom, which is the outermost electron shell, determines whether an element is a conductor, semiconductor, or insulator. The number of electrons in this shell plays a crucial role:
- **Conductors** have more empty spots than they have electrons.
- **Semiconductors** have half-full valence shells (or half-empty depending on your perspective).
- **Insulators** have pretty full valence shells, restricting electron movement.

:p What determines whether an atom is a conductor, semiconductor, or insulator?
??x
The number of electrons in the valence shell of an atom. Conduction depends on the availability and ease of movement of these outermost electrons.
??x
For conductors:
- More empty spots than electrons
For semiconductors:
- Half-full valence shells (neither good conductors nor insulators)
For insulators:
- Pretty full valence shells, restricting electron movement
x??

---
#### Electrical Current and Flow Rate
Background context: Electrical current is the flow of electrons in a conductor. It's measured in amps (A). A coulomb (C) represents one unit of charge and contains approximately $6,241,500,000,000,000,000$ electrons.

:p What is electrical current?
??x
Electrical current is the rate at which electric charges move through a conductor. It's measured in amps (A), where 1 ampere represents one coulomb of charge per second.
??x
How do you measure electrical current in circuits?
??x
Using ammeters, which measure the flow of current through a circuit.
??x
Can you provide an analogy to help understand electrical current flow?
??x
Think of electrons flowing like water from a garden hose. The rate at which they flow is similar to how gallons per minute (GPM) measures water flow in hoses.
x??

---
#### Resistance and Regulation
Background context: Resistance is necessary in circuits to regulate the flow of electricity, preventing overheating and damage to devices. Even wires can act as resistance.

:p What role does resistance play in a circuit?
??x
Resistance regulates the flow of electrical current in a circuit, ensuring that devices function properly and safely without overheating.
??x
How is resistance added or removed from a circuit?
??x
Sometimes circuits must be opened to add or remove resistance. This involves interrupting the flow of electricity to physically change the resistance within the circuit.
x??

---
#### Relationship Between Voltage, Current, and Resistance (Ohm's Law)
Background context: The relationship between voltage (V), current (I), and resistance (R) in a circuit is described by Ohm's law:
$$V = I \times R$$:p What is Ohm’s law?
??x
Ohm’s law states that the current flowing through a conductor between two points is directly proportional to the voltage across the two points, and inversely proportional to the resistance between them. Mathematically:$V = I \times R$.
??x
How does increasing voltage affect the flow of electrons (current)?
??x
Increasing voltage increases the number of amperes (amps) flowing in a wire or conductor because more charge is forced through it per unit time.
x??

---

---
#### Circuit Breaker and Rheostat
Background context: A circuit breaker is a device that interrupts electrical current when necessary, effectively opening the circuit. In contrast, a rheostat can vary resistance without interrupting the circuit. The latter allows devices to continue operating even as their internal resistance changes.
:p What are circuit breakers and rheostats used for?
??x
Circuit breakers are used to safely interrupt an electrical current when it exceeds safe limits, while rheostats adjust resistance in a circuit allowing continuous operation with varying loads or settings. 
???x
---

---
#### Ohm’s Law: Relating Resistance, Current, and Voltage
Background context: Georg Simon Ohm formulated the relationship between voltage (V), current (I), and resistance (R) in electrical circuits. The formula $I = \frac{V}{R}$ represents this relationship.
:p What is Ohm's law?
??x
Ohm's Law states that the current (I) flowing through a conductor between two points is directly proportional to the voltage (V) across the two points, and inversely proportional to the resistance (R) of the conductor. This can be expressed mathematically as $I = \frac{V}{R}$.
???x
---

---
#### Measuring Voltage: Using Voltmeters and Multimeters
Background context: To measure voltage in a circuit, one compares the potential difference between two points using a voltmeter or multimeter. The formula for calculating voltage is $V = I \times R$, where V is the voltage, I is the current, and R is the resistance.
:p How do you measure voltage?
??x
You can measure voltage by comparing the potential difference between two points in a circuit using a voltmeter or multimeter. The formula to calculate voltage is $V = I \times R $, where $ V $ is the voltage, $ I $ is the current, and $ R$ is the resistance.
???x
---

---
#### Understanding Circuits: Conventional vs Electron Flow
Background context: Electricity flows in a circuit like NASCAR cars on a track, constantly moving around in closed loops. The flow of electrical charges follows paths with varying resistances, but conventional current theory considers electricity flowing from positive to negative terminals. Electron flow, however, is the actual movement of electrons from negative to positive.
:p What are conventional and electron flows?
??x
Conventional flow views electricity as moving from the positive terminal to the negative terminal in a circuit, whereas electron flow describes the actual movement of electrons from the negative terminal to the positive terminal.
???x
---

---
#### Components of a Circuit
Background context: A working circuit requires three components: a voltage source (like a battery), a load (such as a light bulb that converts electrical energy into another form), and conductors (materials that carry current between points). These elements work together to ensure continuous flow of electricity.
:p What are the essential components in an electrical circuit?
??x
The essential components in an electrical circuit are:
- A voltage source, such as a battery.
- A load, which converts electrical energy into another form, e.g., a light bulb.
- Conductors that carry current between points.
???x
---

#### Open and Closed Circuits
Background context: An open circuit interrupts the flow of electricity, while a closed circuit allows it to pass uninterrupted. A light switch is used as an example where flipping it "Off" creates an open circuit, preventing current from reaching the bulb.

:p What happens when you flip a light switch to its “Off” position?
??x
When you flip a light switch to its “Off” position, you create an open circuit in the wiring. The electricity can’t flow because there’s a gap in the wire, and since air is typically an insulator (preventing current from passing through), the light turns off.
x??

---

#### Circuit Breakers and Fuses
Background context: Circuit breakers and fuses protect electrical systems by interrupting the circuit when too much energy flows. A detailed explanation of how they work and their differences in functionality and operation is provided.

:p How do circuit breakers and fuses function to protect electrical systems?
??x
Circuit breakers and fuses are safety devices that protect electrical systems from excessive current, which can damage wiring or cause fires.
- **Circuit Breaker:** When the current exceeds a safe level, a circuit breaker trips (physically opens) the circuit. It resets when cooled down, allowing power to be restored.
- **Fuse:** A fuse contains a thin wire that melts and breaks the circuit if the current exceeds its threshold. Once blown, it needs replacement.

```java
public class CircuitBreaker {
    public void checkCurrent(double current) throws Exception {
        if (current > safeLimit) {
            throw new Exception("Circuit Breaker Tripped");
        }
    }

    public void reset() {
        // Reset the breaker to allow power restoration
    }
}

class Fuse {
    private double threshold;
    private boolean isBlown;

    public Fuse(double threshold) {
        this.threshold = threshold;
        this.isBlown = false;
    }

    public void checkCurrent(double current) throws Exception {
        if (current > threshold && !isBlown) {
            blow();
        }
    }

    private void blow() {
        isBlown = true; // Circuit is now broken
        System.out.println("Fuse blown - Circuit interrupted.");
    }
}
```
x??

---

#### Series Circuits
Background context: In a series circuit, there's only one path for current to flow. The same amount of current passes through every component, and if any part breaks, the entire circuit stops working.

:p What is a series circuit?
??x
In a series circuit, all components are connected in a single continuous loop, with no branching paths. Current flows sequentially through each component. If one component fails (breaks), the entire circuit stops functioning because there's only one path for current to flow.
x??

---

#### Parallel Circuits
Background context: In parallel circuits, multiple branches exist, allowing current to split and flow in different directions. Breaks in one branch do not affect others.

:p What is a parallel circuit?
??x
In a parallel circuit, components are connected across the same two points, creating separate paths for the current. This allows current to split among different branches and continue flowing even if one path breaks.
x??

---

#### Series-Parallel Circuits
Background context: Series-parallel circuits combine both series and parallel configurations, offering flexibility but with more complex behavior.

:p What is a series-parallel circuit?
??x
A series-parallel circuit combines elements of both series and parallel connections. Components are arranged such that some parts operate in series while others are in parallel. This arrangement provides flexibility in design but requires careful calculation to determine overall resistance.
x??

---

#### Voltage in Circuits
Background context: Voltage remains consistent throughout a series circuit, while it is the same across each branch of a parallel circuit.

:p How does voltage behave differently in series and parallel circuits?
??x
- **Series Circuit:** Voltage remains constant throughout the circuit. The total voltage is distributed among all components.
- **Parallel Circuit:** Voltage is the same across each parallel branch but varies with different resistances; current can vary between branches, but the total current must equal the sum of currents in each branch.

```java
public class VoltageMeasurement {
    public double calculateVoltage(double R1, double R2) {
        // Assuming a constant voltage source V = 10V for simplicity
        return 10 / (1 / R1 + 1 / R2); // Equivalent resistance formula: R_total = 1/(1/R1+1/R2)
    }
}
```
x??

---

#### Short Circuits
Background context: A short circuit occurs when a wire bypasses the intended path, causing excessive current flow. While not always dangerous, it can be problematic in electrical systems.

:p What is a short circuit?
??x
A short circuit happens when a wire or component bypasses the intended path, allowing current to flow directly between points of different potential without passing through components. This can cause excessive current and potentially damage equipment.
x??

---

#### Resistance in Circuits
Background context: Calculating resistance involves understanding series, parallel, and series-parallel configurations.

:p How do you calculate total resistance in a series circuit?
??x
In a series circuit, the total resistance $R_{total}$ is simply the sum of all individual resistances. For example:
$$R_{total} = R_1 + R_2 + R_3$$

This means if multiple loads are connected in series, their resistances add up.

```java
public class ResistanceCalculation {
    public double calculateSeriesResistance(double r1, double r2, double r3) {
        return r1 + r2 + r3;
    }
}
```
x??

---

#### Parallel Circuits and Total Resistance
Background context: In a parallel circuit, the total resistance $R_{total}$ can be calculated using the formula involving the reciprocal of each individual resistance.

:p How do you calculate total resistance in a parallel circuit?
??x
In a parallel circuit, the total resistance is given by:
$$\frac{1}{R_{total}} = \frac{1}{R_1} + \frac{1}{R_2} + \ldots + \frac{1}{R_n}$$

For example, if you have two resistors $R_1 $ and$R_2$:
$$R_{total} = \left( \frac{1}{R_1} + \frac{1}{R_2} \right)^{-1}$$

```java
public class ParallelResistanceCalculation {
    public double calculateParallelResistance(double r1, double r2) {
        return 1 / (1/r1 + 1/r2);
    }
}
```
x??


# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 49)

**Starting Chapter:** Switching Things Up with Alternating and Direct Current

---

#### Total Resistance Calculation in Series-Parallel Circuits
In a series-parallel circuit, you need to first calculate the resistance of the parallel combination and then combine it with the series resistances. The formula for the total resistance $R_{total}$ in a parallel circuit is:
$$\frac{1}{R_{total}} = \frac{1}{R_1} + \frac{1}{R_2}$$:p How do you calculate the total resistance of a series-parallel circuit?
??x
To find the total resistance $R_{total}$ in a series-parallel circuit, first determine the equivalent resistance ($ R_{eq}$) of the parallel combination using:
$$\frac{1}{R_{eq}} = \frac{1}{R_1} + \frac{1}{R_2}$$

Then, add this to any series resistances. For example:
- Given $R_1 = 10 $ and$R_2 = 50$:
$$\frac{1}{R_{eq}} = \frac{1}{10} + \frac{1}{50} = \frac{5 + 1}{50} = \frac{6}{50} = \frac{3}{25}$$
$$

R_{eq} = \frac{25}{3} \approx 8.33 \, \Omega$$- If this parallel combination is in series with another resistance $ R_3 = 80$:
$$R_{total} = 80 + 8.33 = 88.33 \, \Omega$$

However, the exact example given uses different values to match the answer choices.

x??

---

#### Power Calculation Using Voltage and Current
Power (P) in watts can be calculated using the formula:
$$

P = V \times I$$where $ V $ is the voltage (in volts) and $ I$ is the current (in amperes).

:p How do you calculate power when given voltage and current?
??x
To calculate power, use the formula:
$$P = V \times I$$

For example, if the voltage $V = 120 $ volts and the current$I = 5$ amperes:
$$P = 120 \times 5 = 600 \, \text{watts}$$x??

---

#### Kilowatt-Hour Calculation
Kilowatt-hours (kWh) represent the energy of 1,000 watts working for one hour. To calculate kWh:
$$\text{kWh} = \text{kW} \times \text{hours}$$where kW is kilowatts and hours is the time in hours.

:p How do you convert watt-hours to kilowatt-hours?
??x
To convert watt-hours (Wh) to kilowatt-hours (kWh), divide by 1,000:
$$\text{kWh} = \frac{\text{Wh}}{1000}$$

For example, if you use a 40-watt light bulb for 5 hours:
- Total watt-hours:$40 \times 5 = 200 \, \text{Wh}$- Convert to kWh:$\frac{200}{1000} = 0.2 \, \text{kWh}$ x??

---

#### Chemical Effect of Electricity
The chemical effect occurs when electricity passes through a chemical compound and breaks it down into its components (electrolysis). This is used in processes like electroplating.

:p What is the chemical effect of electricity?
??x
The chemical effect of electricity involves passing an electric current through a solution containing ions, causing those ions to undergo reduction or oxidation at the electrodes. This process is called electrolysis and can be used for:
- Electroplating: Depositing a thin layer of metal on another material.
- Manufacturing: Producing pure metals from their compounds.

Example: In electroplating silver onto an object using a silver nitrate solution, the electrons from the negative electrode reduce silver ions to solid silver depositing on the object.

x??

---

#### Heat Effect of Electricity
Electricity passing through conductors causes them to heat up due to resistance. This is used in heating elements like electric stoves.

:p What is the heat effect of electricity?
??x
The heat effect occurs because current must overcome the resistance of the wire, generating heat:
$$P = I^2 R$$where $ P $is power (watts),$ I $is current (amperes), and$ R $is resistance ($\Omega$).

For example, if 10 amperes of current flow through a 5-ohm resistor:
$$P = 10^2 \times 5 = 500 \, \text{watts}$$x??

---

#### Magnetic Effect and Electromagnetic Induction
Magnetic fields are created by electric currents in wires. When wrapped around an iron core, the wire forms a coil that can generate a strong magnetic field.

:p What is the magnetic effect of electricity?
??x
The magnetic effect involves creating a magnetic field with a current-carrying wire. The strength of the magnetic field depends on several factors:
- Number of turns: Increasing the number of turns increases the field.
- Closeness of the turns: Closer wraps increase the field.
- Amount of current: Higher currents increase the field.

Electromagnetic induction is when a changing magnetic field induces an electromotive force (EMF) in a conductor, as seen in generators:
$$\text{EMF} = -N \frac{\Delta \Phi}{\Delta t}$$where $ N $ is the number of turns and $\Phi$ is the magnetic flux.

x??

---

#### Electromagnetic Induction
When a conductive material (like a loop of wire) moves through a magnetic field, an induced current is created due to the change in magnetic flux. This is used in electric generators.

:p What role does electromagnetic induction play?
??x
Electromagnetic induction is fundamental for generating electricity:
- Passing a stationary conductor through a uniform magnetic field generates no EMF.
- Rotating the conductor through the lines of force creates an induced current due to changing magnetic flux.

For example, in a simple generator setup, a loop of wire rotates inside a magnetic field, causing a varying magnetic flux that induces current in the wire:
```java
// Pseudocode for basic generator
public class Generator {
    private boolean isRotating;

    public void rotate() {
        if (isRotating) {
            // Induce EMF and generate current
        }
    }

    public void stop() {
        isRotating = false;
    }
}
```

x??

---
#### Physiological Effect of Current
Background context: When a current passes through your bicep (or any muscle), it causes the muscle to contract. This effect is used in medicine, such as for physiotherapy or biofeedback devices.

:p What is the physiological effect of current when passing through muscles?
??x
The physiological effect of current is that it causes muscle contraction. This principle is used in medical applications like electrotherapy and physiotherapy.
x??

---
#### Direct Current (DC)
Background context: DC flows in one direction, making it suitable for devices like batteries, phones, and flashlights.

:p What type of source provides direct current?
??x
Direct current often comes from a battery. It is what your phone, kids' toys, and flashlights use when not charging from the wall.
x??

---
#### Alternating Current (AC)
Background context: AC changes direction constantly in a regular pattern. Higher voltages are easier to generate with AC, making it more efficient for long-distance power transmission.

:p What type of current comes directly into your home?
??x
Alternating current comes directly into your home from the power company. It can change direction and voltage when encountering a transformer.
x??

---
#### Frequency of Alternating Current
Background context: The frequency of AC is measured in Hertz (Hz), which equals one complete cycle per second. Household AC typically operates at 60 Hz, but devices often use higher frequencies.

:p What does the term "frequency" refer to in alternating current?
??x
The number of times an alternating current changes direction per second is known as its frequency. It is measured in Hertz (Hz), with one Hz equaling one complete cycle per second.
x??

---
#### Grounding for Electrical Safety
Background context: Grounding provides a safe path for excess electricity, preventing fires or electric shocks. Homes typically have grounding wires parallel to "hot" and "neutral" wires.

:p What is the purpose of grounding in electrical systems?
??x
The purpose of grounding is to provide a safe place for excess electricity to flow without harming anyone. Grounding wires help capture stray current from damaged live wires, directing it safely into the ground.
x??

---
#### AC/DC Conversion for Devices
Background context: Many modern devices use AC/DC conversion where AC enters through an outlet and is transformed by a rectifier to DC.

:p How do most electronic devices receive direct current?
??x
Most electronic devices receive alternating current from an outlet, which is then transformed into direct current using a rectifier.
x??

---
#### Impedance in Circuits
Background context: Impedance combines resistance with properties specific to AC like capacitive and inductive reactance. Capacitors store charge while inductors use magnetic fields to resist current changes.

:p What are the two types of reactance that affect alternating current?
??x
The two types of reactance that affect alternating current are capacitive reactance and inductive reactance.
x??

---
#### Capacitors and Inductors
Background context: Capacitors store charge, while inductors use magnetic fields to resist changes in current. These components provide specific reactances needed for circuit operation.

:p What is the function of a capacitor in an AC circuit?
??x
In an AC circuit, capacitors are used to store or hold a charge and act as opposition to changing AC voltage through capacitive reactance.
x??

---
#### Inductive Reactance Example
Background context: Inductors use magnetic fields to resist changes in current. In an AC circuit, the rate of change of current flow creates resistance called inductive reactance.

:p What happens when a coil (inductor) has full current but then the current is removed?
??x
When a coil (inductor) has full current and then the current is removed, the magnetic field decays gradually. This decay continues to push electrons in the path they were going.
x??

---

---
#### Rectification Process and Components
Background context: The process of converting alternating current (AC) to direct current (DC) is called rectification. This is essential for powering many electronic devices that require DC, such as computers and smartphones. Diodes are a crucial part of these circuits, allowing electricity to flow in only one direction.

Relevant formulas and explanations:
- A diode's behavior can be modeled by the Shockley diode equation:$I_D = I_S \left( e^{\frac{V_D}{nV_T}} - 1 \right)$ where $ V_D $ is the voltage across the diode,$n $ is the ideality factor, and$I_S$ is the reverse saturation current.
- Inductors ($L $) store energy in a magnetic field and capacitors ($ C$) store energy in an electric field.

:p What is rectification?
??x
Rectification is the process of converting alternating current (AC) to direct current (DC). This is typically achieved using diodes, which allow electricity to flow only in one direction. Inductors and capacitors are also often used in these circuits.
x??

---
#### Semiconductor Diodes
Background context: Semiconductors, like diodes made from materials such as silicon or germanium, play a critical role in rectification by allowing current to pass in only one direction.

Relevant formulas and explanations:
- The forward voltage drop across a diode can be approximated as 0.7V for silicon diodes (and 0.3V for germanium).
- Reverse biasing a diode will not allow significant current flow unless the applied voltage is extremely high, which would damage the diode.

:p What are semiconductor diodes?
??x
Semiconductor diodes are components made of materials like silicon or germanium that conduct electricity in only one direction. They play a crucial role in rectifying AC to DC.
x??

---
#### Transistor Function and Properties
Background context: A transistor is a semiconductor device used for amplification, controlling the flow of electric current. It's essential in devices such as transistor radios.

Relevant formulas and explanations:
- The relationship between collector-emitter voltage ($V_{CE}$) and base current ($ I_B $) can be described by the following equation:$ I_C = \beta I_B + I_{CBO}$where $\beta$ is the current gain factor, and $I_{CBO}$ is the collector-base open circuit current.

:p What does a transistor control?
??x
A transistor controls the flow of electric current in a circuit. It can amplify signals, making it useful in devices like transistor radios.
x??

---
#### Block Diagrams for Circuit Systems
Background context: Block diagrams are used to represent complex systems made up of multiple circuits. They help in understanding the overall system and its components.

Relevant formulas and explanations:
- No specific formula is required here; however, block diagrams often use symbols such as boxes to indicate different functional blocks within a system.

:p What are block diagrams?
??x
Block diagrams are used to represent complex systems made up of multiple circuits. They help in understanding the overall system by breaking it down into manageable components and showing how they interact.
x??

---
#### Electrical Circuit Components: Wires
Background context: Wires are crucial for passing current between different parts of an electrical circuit. They connect various components and ensure a path for the flow of electricity.

Relevant formulas and explanations:
- No specific formula is required; however, understanding wire connections (joined vs unjoined) is important for interpreting complex circuit diagrams.

:p What do wires represent in circuits?
??x
Wires represent paths through which current flows in an electrical circuit. Joined wires are indicated by a dark circle, while unconnected or unjoined wires are shown with a hump symbol to indicate they don't cross electrically.
x??

---

---
#### Cell
Background context: A cell supplies electrical current. While some people refer to a single cell as a battery, technically, a battery consists of two or more cells. The large terminal on a cell diagram is positive.

:p What is a cell?
??x
A cell is an individual unit that generates electrical current. In diagrams, the large terminal (on the left side) represents the positive terminal.
x??

---
#### Battery
Background context: A battery is composed of two or more cells connected in series to provide a higher voltage than any single cell can supply. The large terminal on a battery diagram is also positive.

:p What is a battery?
??x
A battery consists of multiple cells connected together, providing increased voltage and current capabilities compared to a single cell. The large terminal indicates the positive side.
x??

---
#### DC Power Supply
Background context: A DC power supply provides direct current (DC), which flows in one direction only. This is different from AC power supplies.

:p What is a DC power supply?
??x
A DC power supply delivers direct current, which always flows in one direction. It converts other forms of energy into electrical energy.
x??

---
#### AC Power Supply
Background context: An AC power supply provides alternating current (AC), which constantly changes direction at a specific frequency.

:p What is an AC power supply?
??x
An AC power supply delivers alternating current, where the direction of flow continuously reverses according to a specified frequency.
x??

---
#### Fuse
Background context: A fuse serves as a safety device that melts when the current flowing through it exceeds its rated value. Fuses help prevent overheating and electrical fires.

:p What is a fuse?
??x
A fuse protects circuits by melting if the current exceeds its rated value, thus breaking the circuit to prevent damage or fire.
x??

---
#### Transformer
Background context: A transformer uses two coils of wire linked by an iron core to step up (increase) or step down (decrease) AC voltages without making a direct electrical connection between the coils.

:p What is a transformer?
??x
A transformer consists of two coils and an iron core, used to adjust voltage levels in AC circuits. The magnetic field transfers energy between the coils.
x??

---
#### Ground
Background context: A ground provides a connection to the Earth, which helps ensure safety by providing a path for excess current.

:p What is a ground?
??x
A ground is a connection to the Earth that helps prevent electrical hazards and ensures proper circuit operation.
x??

---
#### Transducer
Background context: A transducer converts energy from one form to another. Examples include lighting lamps, indicator lamps, motors, heaters, bells, buzzers, microphones, earphones, and speakers.

:p What is a transducer?
??x
A transducer is a device that transforms energy from one form to another for various purposes such as lighting, motion, heat generation, sound production, and more.
x??

---
#### Inductor
Background context: An inductor creates a magnetic field when current passes through it. This property makes it useful in circuit design.

:p What is an inductor?
??x
An inductor is a coil of wire that generates a magnetic field when current flows through it. It's used for filtering, storing energy, and controlling current.
x??

---
#### Switch
Background context: Various types of switches exist to control the flow of electrical current, including push switches, push-to-break switches, on/off switches, two-way switches, dual on/off switches, and relays.

:p What is a switch?
??x
A switch controls the flow of electrical current in circuits. Different types include push switches, push-to-break switches, on/off switches, two-way switches, dual on/off switches, and relays.
x??

---
#### Resistor (Nonvariable)
Background context: Resistors limit current flow and are used to control voltage levels in a circuit.

:p What is a resistor?
??x
A resistor limits the flow of electrical current to control voltage levels in circuits. There are two basic types of resistor symbols.
x??

---

---
#### Resistors and Their Color Coding
Resistors restrict electric current flow, rated in ohms with a color-coded system. The first two bands represent the significant digits, while the third band indicates the multiplier (number of zeros after those digits). A gold or silver band can indicate tolerance levels, and sometimes there is an additional quality band.

The resistor value for red-brown-orange would be calculated as follows:
- First digit: Red = 2
- Second digit: Brown = 1
- Multiplier: Orange = $10^3 $ Thus, the value is$2 \times 10^3$ ohms or 2 kΩ.

:p What does a red-brown-orange resistor indicate in terms of its resistance?
??x
The resistor indicates a resistance value of 2 kΩ. This calculation follows the steps:
- The first band (Red) represents 2.
- The second band (Brown) represents 1.
- The third band (Orange) multiplies these by $10^3$.
Thus, $2 \times 10^3 = 2000$ ohms or 2 kΩ.

```java
public class Resistor {
    public static void main(String[] args) {
        int firstDigit = getDigit("Red"); // Returns 2 for Red
        int secondDigit = getDigit("Brown"); // Returns 1 for Brown
        String multiplierBand = "Orange"; // Multiplier is Orange

        int resistance = firstDigit * Math.pow(10, getMultiplierValue(multiplierBand));
        System.out.println(resistance); // Output should be 2000 or 2 kΩ
    }

    private static int getDigit(String color) {
        switch (color) {
            case "Black":
                return 0;
            case "Brown":
                return 1;
            case "Red":
                return 2;
            case "Orange":
                return 3;
            case "Yellow":
                return 4;
            // Add other cases as needed
            default:
                throw new IllegalArgumentException("Unknown resistor color");
        }
    }

    private static int getMultiplierValue(String band) {
        switch (band) {
            case "Brown":
                return 1;
            case "Red":
                return 2;
            case "Orange":
                return 3;
            // Add other cases as needed
            default:
                throw new IllegalArgumentException("Unknown multiplier color");
        }
    }
}
```
x??
---

#### Rheostats, Potentiometers, and Preset Resistors
Rheostats are variable resistors with two contacts, primarily used for controlling current such as adjusting lamp brightness or motor speed. Potentiometers have three contacts and control voltage. Preset resistors can be adjusted using a screwdriver but are typically set during circuit assembly.

:p What is the primary use of a rheostat?
??x
Rheostats are primarily used to control current in circuits, such as adjusting lamp brightness or motor speed by varying resistance. They operate with two contacts and their resistance can be adjusted.
x??

---
#### Capacitors: Storage and Filtering
Capacitors store electric charge and are crucial for timing circuits where it takes time for a capacitor to fill with charge. They also play a role in filter circuits, passing AC signals while blocking DC.

:p What is the main application of capacitors in filtering?
??x
The primary application of capacitors in filtering is their ability to pass alternating current (AC) signals and block direct current (DC) signals. This property makes them essential components in filter circuits.
x??

---
#### Diodes: Directional Current Flow
Diodes allow electricity to flow only in one direction, acting as an electrical valve. Light-emitting diodes (LEDs) emit light when a current passes through them. Zener diodes can conduct current in the opposite direction after reaching a certain voltage threshold.

:p What does a diode do?
??x
A diode allows electricity to flow in only one direction, functioning similarly to a valve or check valve in electrical circuits. It ensures that current cannot flow backwards.
x??

---
#### Transistors: Current Amplification
Transistors amplify current by controlling the amount of current flowing through them. They are used to amplify small currents from logic chips so they can operate high-current devices like lamps, relays, or other components.

:p What is a common use of transistors?
??x
A common use of transistors is amplifying the small output current from a logic chip so that it can operate high-current devices such as lamps, relays, or other components. This allows for controlling more powerful systems with low-power signals.
x??

---
#### Amplifiers: Circuit Circumvention
Amplifiers are not electronic components but complex circuits used to magnify power, current, or voltage.

:p What is the role of an amplifier in electronics?
??x
The role of an amplifier in electronics is to amplify signals—such as voltage, current, or power. It takes a small input signal and produces a larger output signal, making it useful for boosting weak signals.
x??

---
#### Antennas: Signal Reception and Transmission
Antennas are devices designed to receive and/or transmit radio signals.

:p What do antennas primarily do?
??x
Primarily, antennas are used to receive and transmit radio signals. They facilitate communication by converting electrical currents into electromagnetic waves for transmission and vice versa for reception.
x??

---
#### Circuit Diagrams: Component Connection Visualization
Circuit diagrams show how electronic components are connected together, using symbols to represent each component clearly with all wires drawn neatly as straight lines. Actual physical layouts differ from these diagrams.

:p What is the purpose of circuit diagrams?
??x
The purpose of circuit diagrams is to visually represent how electronic components are interconnected in a circuit. They help in understanding and testing circuits by providing clear connections between components, even though actual physical layouts may vary.
x??

---


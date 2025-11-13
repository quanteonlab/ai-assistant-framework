# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 106)

**Starting Chapter:** Cruising toward a Better Score

---

---
#### Catalytic Converter Function
Background context explaining the catalytic converter's role in exhaust emissions control. It neutralizes toxic fumes through chemical reactions, converting harmful compounds into less-harmful ones like carbon dioxide and water.

:p What does a catalytic converter do?
??x
A catalytic converter is an emission control device that reduces harmful gases from internal combustion engines before they are expelled into the atmosphere. It works by utilizing a catalyst to transform toxic exhaust gases such as carbon monoxide (CO), unburned hydrocarbons (HC), and nitrogen oxides (NOx) into less dangerous compounds like water vapor, carbon dioxide, and nitrogen.

```java
public class CatalyticConverter {
    public void neutralizeExhaust(Gases gases) {
        // Simulate the chemical reactions in a catalytic converter.
        gases.carbonMonoxide = 0; // CO is oxidized to CO2
        gases.unburnedHydrocarbons = 0; // HC are converted to H2O and CO2
        gases.nitrogenOxides = 0; // NOx are reduced to N2 and O2
    }
}
```
x??

---
#### Exhaust Manifold and Tailpipe Function
Background context explaining how the exhaust manifold and tailpipe facilitate the emission process. Gases collect in the manifold before being expelled through the tailpipe, often after undergoing chemical transformations.

:p What are the roles of the exhaust manifold and tailpipe?
??x
The exhaust manifold collects toxic fumes from the engine's cylinders and routes them to the catalytic converter for further processing. After treatment by the converter, the gases pass through the tailpipe, which is the final exit point where they are expelled into the atmosphere.

```java
public class ExhaustSystem {
    public void processExhaust(Gases gases) {
        gases = catalyticConverter.treat(gases); // Treat gases in the catalytic converter.
        exhaustManifold.collectGases(gases);
        tailpipe.expelGases(exhaustManifold.getCollectedGases());
    }
}
```
x??

---
#### Positive-Crankcase Ventilation System
Background context explaining how this system recirculates unburned or partially burned fuel back into the engine to increase efficiency and reduce emissions. It operates by forcing crankcase vapors through a valve to the intake manifold.

:p How does positive-crankcase ventilation work?
??x
Positive-crankcase ventilation (PCV) is designed to improve engine performance and reduce harmful emissions. The system uses a valve that routes unburned or partially burned fuel from the crankcase back into the intake manifold, where it can be reintroduced to the combustion process.

```java
public class PositiveCrankcaseVentilation {
    public void operate(PCVValve valve, Engine engine) {
        Gases gases = valve.extractGases();
        engine.intakeManifold.addFuel(gases);
    }
}
```
x??

---
#### Air-Injection System Function
Background context explaining the air-injection system's role in reusing unburned or partially burned fuel. It injects additional air into the exhaust stream to enhance combustion and reduce pollutants.

:p What is the purpose of an air-injection system?
??x
An air-injection system introduces extra oxygen into the exhaust stream to help burn any remaining, unburned hydrocarbons (HC) and carbon monoxide (CO). This process can significantly decrease harmful emissions while improving overall fuel efficiency.

```java
public class AirInjectionSystem {
    public void injectAir(ExhaustStream stream, int volume) {
        stream.addOxygen(volume);
    }
}
```
x??

---
#### Universal Joint and Drive Shaft Function
Background context explaining how these components transfer power from the engine to the wheels. The universal joint allows the drive shaft to move up and down, ensuring continuous power transmission while the vehicle is in motion.

:p What does a universal joint do?
??x
A universal joint (U-joint) enables the drive shaft to maintain smooth power transmission when it moves up and down with the suspension. This ensures that torque is effectively transferred from the engine to the wheels without disrupting the drivetrain's function.

```java
public class UniversalJoint {
    public void transferTorque(int torque, Axle axle) {
        // Simulate power transfer through the U-joint.
        axle.receiveTorque(torque);
    }
}
```
x??

---
#### Rear-Wheel Drive vs. Front-Wheel Drive vs. All-Wheel Drive
Background context explaining different drive systems and their applications in automobiles. Rear-wheel drive pushes the car using the rear wheels, front-wheel drive pulls it with the front wheels, while all-wheel drive uses both sets of wheels for better traction.

:p What are the differences between rear-wheel drive, front-wheel drive, and all-wheel drive?
??x
Rear-wheel drive (RWD) uses the rear wheels to push the car. The power comes from the engine through a transmission and driveshaft that connect to the rear axle. Front-wheel drive (FWD) pulls the car using the front wheels, where the power travels directly from the transmission to the front axle. All-wheel drive (AWD), commonly found in SUVs, uses both sets of wheels for propulsion and traction, often with locking differentials to manage speed differences between driven wheels.

```java
public class DriveSystem {
    public void applyDrive(DriveType type) {
        switch (type) {
            case REAR_WHEEL_DRIVE:
                // Implement RWD logic.
                break;
            case FRONT_WHEEL_DRIVE:
                // Implement FWD logic.
                break;
            case ALL_WHEEL_DRIVE:
                // Implement AWD logic with traction control.
                break;
        }
    }
}
```
x??

---

---
#### Transmission Types and Functions
Vehicles have two types of transmissions: automatic or manual. Both allow the driver to control the amount of torque used, which is crucial for different driving conditions.

Automatic transmission uses a torque converter to shift gears automatically based on speed and engine load:
```java
public class AutomaticTransmission {
    public void shiftGears(double currentSpeed) {
        if (currentSpeed < LOW_SPEED_THRESHOLD) {
            setGear(1); // First gear for low speeds
        } else if (currentSpeed >= HIGH_SPEED_THRESHOLD) {
            setGear(5); // Fifth gear for high speeds
        }
    }

    private void setGear(int gearNumber) {
        // Logic to change gears using torque converter
    }
}
```
:p What are the two main types of transmissions in vehicles?
??x
The two main types of transmissions in vehicles are automatic and manual. Automatic transmissions use a torque converter to shift gears automatically based on speed and engine load, while manual transmissions require the driver to manually shift gears.
x?

---
#### Torque Control with Gears
Torque is increased or decreased using different gears in both automatic and manual transmissions.

In an automatic transmission, the gear change is managed by the car's computer:
```java
public class AutomaticTransmission {
    public void controlTorque(double speed) {
        if (speed > SLOW_SPEED_THRESHOLD && speed < MEDIUM_SPEED_THRESHOLD) {
            setGear(2); // Second gear for medium speeds
        } else if (speed >= HIGH_SPEED_THRESHOLD) {
            setGear(5); // Fifth gear for high speeds, less torque needed
        }
    }

    private void setGear(int gearNumber) {
        // Logic to change gears using a torque converter
    }
}
```
:p How does an automatic transmission manage torque?
??x
An automatic transmission manages torque through a series of gears and a torque converter. The car's computer shifts the gears based on the current speed, ensuring that the appropriate amount of torque is applied for efficient driving.
x?

---
#### Manual Transmission Operation
In a manual transmission, the driver controls gear changes by compressing the clutch to disconnect the engine from the drive shaft.

The process involves:
```java
public class ManualTransmission {
    public void shiftGear(int currentGear) {
        if (currentGear == 1 && speed > STARTING_SPEED) { // If in first gear and car is moving
            releaseClutch(); // Release clutch to change gears
            setGear(2); // Shift to second gear
            engageClutch(); // Engage clutch back into the drive shaft
        }
    }

    private void releaseClutch() {
        // Logic to disengage clutch
    }

    private void engageClutch() {
        // Logic to re-engage clutch
    }

    private void setGear(int gearNumber) {
        // Logic to change gears manually
    }
}
```
:p How does a driver operate a manual transmission?
??x
A driver operates a manual transmission by compressing the clutch to disconnect the engine from the drive shaft, allowing them to shift gears. The process involves temporarily disengaging the clutch, shifting into the desired gear, and then re-engaging the clutch.
x?

---
#### Suspension System Components
The suspension system includes struts, shock absorbers, and tires that work together to maximize friction between the vehicle and road surface.

Struts support weight with attached springs:
```java
public class Strut {
    private Spring spring;

    public Strut(Spring spring) {
        this.spring = spring;
    }

    public void absorbImpact() {
        if (tireHitBump()) { // Tire hits a bump
            spring.stretch(); // Spring stretches to absorb impact
        }
    }
}

public class Spring {
    public void stretch() {
        // Logic for spring stretching under load
    }
}
```
:p What are the main components of a suspension system?
??x
The main components of a suspension system include struts, shock absorbers, and tires. Struts support the vehicle's weight with attached springs that help adapt to road irregularities.
x?

---
#### Shock Absorber Functionality
Shock absorbers reduce vibrations by converting them into heat energy.

A typical shock absorber works as follows:
```java
public class ShockAbsorber {
    private Piston piston;
    private Fluid fluid;

    public ShockAbsorber(Piston piston, Fluid fluid) {
        this.piston = piston;
        this.fluid = fluid;
    }

    public void absorbVibration(double impactForce) {
        piston.moveUp(impactForce); // Piston moves up due to impact
        fluid.dissipateHeat(piston.velocity()); // Convert vibration energy into heat
    }
}

public class Fluid {
    private double temperature;

    public void dissipateHeat(double velocity) {
        this.temperature += 0.1 * velocity; // Simple heat dissipation model
    }
}
```
:p What is the primary function of a shock absorber?
??x
The primary function of a shock absorber is to reduce vibrations by converting them into heat energy, providing a smoother ride and better vehicle control.
x?

---
#### Tire Functionality
Tires are essential for contact with the road surface and provide stability and traction.

The role of tires in handling is explained as follows:
```java
public class Tires {
    public void handleRoadSurface() {
        // Logic to create friction and stability
    }
}

public class RubberCompound {
    private double gripLevel;

    public void increaseGrip(double speed) {
        if (speed < LOW_SPEED_THRESHOLD) {
            this.gripLevel += 0.2; // Increase grip for low speeds
        } else {
            this.gripLevel -= 0.1; // Decrease grip at high speeds to prevent overheating
        }
    }
}
```
:p What role do tires play in vehicle handling?
??x
Tires are crucial for contact with the road surface, providing stability and traction. They create friction caused by their rubber compound, which adapts to different driving conditions to maintain optimal performance.
x?

---

---
#### Springs Function
Background context explaining how springs function in a vehicle's suspension system. They work alongside shocks to allow smooth wheel movement by absorbing impacts and vibrations.

:p What is the role of springs in a vehicle’s suspension?
??x
Springs play a crucial role in supporting the chassis, working with the shocks to absorb impacts and ensure that the wheels move up and down smoothly.
x??

---
#### Steering Knuckle Description
Background context on the steering knuckle's function as the connection point between tie rods and wheels, controlling wheel direction.

:p What is a steering knuckle?
??x
A steering knuckle is the pivotal joint connecting the tie rod to the wheel. It controls the direction in which the wheel turns.
x??

---
#### Control Arms Explanation
Background context on control arms (also known as A-arms) and their role in maintaining vertical alignment of wheels during movement.

:p What are control arms?
??x
Control arms, also called A-arms, are long metal pieces that connect to the steering knuckle via ball joints. They help maintain the wheel's vertical position as it moves up and down.
x??

---
#### Tie Rods Function
Background context on tie rods transferring force from the steering system to wheels, enabling them to turn.

:p How do tie rods function in a vehicle?
??x
Tie rods transfer force from the steering linkage or rack to the steering knuckle. This action causes the wheel to turn, allowing for directional control of the vehicle.
x??

---
#### Steering Systems Types
Background context on the two main types of steering systems: rack-and-pinion and Pitman arm, with mention of power-assist features.

:p What are the primary types of steering systems?
??x
The primary types of steering systems include rack-and-pinion and Pitman arm. Both can be power-assisted to ease turning the steering wheel and getting a response from the vehicle's wheels.
x??

---
#### Brake System Components
Background context on the brake system components, including how they work together to stop the car.

:p What are the key components of a brake system?
??x
Key brake system components include the brake pedal for driver input, the master cylinder that pushes fluid through lines, and brake assemblies at each wheel. The system can use either drum or disc brakes.
x??

---
#### Drum Brakes Operation
Background context on how drum brakes work using hydraulic pressure to stop a car.

:p How do drum brakes function?
??x
In drum brakes, lines connect to a hydraulic cylinder that forces pistons outward, causing brake shoes to press against the rotating metal drum inside. This creates friction and stops the wheel.
x??

---
#### Disc Brakes Operation
Background context on disc brakes using hydraulic pressure to stop a car.

:p How do disc brakes function?
??x
Disc brakes use a master cylinder pushing fluid through a caliper with pistons and brake pads that squeeze against a rotating rotor, stopping the vehicle via fluid and pressure.
x??

---
#### Brake System Types in Modern Cars
Background context on modern cars using both drum and disc brakes.

:p What types of brakes are commonly used in modern cars?
??x
Most modern vehicles use both drum and disc brakes. Drum brakes are typically found on rear wheels, while disc brakes are usually on the front.
x??

---
#### Stopping Power Comparison
Background context comparing the stopping power of drum and disc brake systems.

:p How do drum and disc brakes compare in terms of stopping power?
??x
Drum brakes have a single-sided contact with the rotating drum, offering less stopping power. Disc brakes provide double-sided pressure on a rotor, effectively doubling their stopping power.
x??

---

---
#### Engine Cooling System Components
The water-cooled engine dissipates heat through various components, each playing a crucial role. The radiator is specifically designed to transfer excess heat from the coolant to the surrounding air.

:p What component of a water-cooled engine dissipates heat?
??x
The component that dissipates heat in a water-cooled engine is the **radiator**.
x??

---
#### Carburetor Function
A carburetor's primary function is to mix fuel and air in proper proportions for combustion. This mixture is then delivered to the engine cylinders through intake valves.

:p A carburetor has the same function as what other component?
??x
A carburetor has the same function as a **fuel-injection system**.
x??

---
#### Engine Energy Storage Mechanism
An engine's rotational energy, which is converted from fuel combustion, can be stored using a mechanical device. The flywheel stores this rotational energy by spinning and releasing it slowly.

:p Which mechanical device stores an engine’s rotational energy?
??x
The mechanical device that stores an engine’s rotational energy is the **flywheel**.
x??

---
#### Catalytic Converter Role
A catalytic converter plays a critical role in reducing harmful emissions. It operates by facilitating chemical reactions that convert toxic substances into less harmful ones.

:p What does a catalytic converter do?
??x
A catalytic converter converts toxic substances into less harmful ones, thereby **reducing toxic emissions**.
x??

---
#### Routine Automotive Tuneup
A routine automotive tuneup involves several maintenance tasks to ensure the car runs efficiently. However, some items are not typically part of this process.

:p Which of the following is NOT normally part of a routine automotive tuneup?
??x
The item that is **NOT** normally part of a routine automotive tuneup is replacing the **CV axles**.
x??

---
#### Antifreeze Function
Antifreeze serves multiple purposes in an engine's cooling system, including preventing the coolant from boiling and freezing.

:p What does antifreeze do?
??x
Antifreeze is used to prevent the engine from overheating and to prevent water in the cooling system from **freezing**.
x??

---
#### Assembly Identification
The assembly pictured could be one of several components. Understanding each part's function helps diagnose and repair issues efficiently.

:p What assembly is pictured?
??x
The assembly shown is most likely a **radiator**.
x??

---
#### Jump Start Safety

Jump-starting another vehicle can be tricky, especially with modern electronic systems that might not respond as expected when receiving power. This can lead to potential damage or failure of the jump-started vehicle's electronics.

:p Why might you hesitate to offer a jump start to another vehicle?
??x
You might hesitate to offer a jump start if one of the vehicles has a **digital ignition system**, as it could be damaged by incorrect battery voltage.
x??

---
#### Spark Plug Gap Issues

A spark plug with an improper gap can lead to various issues, including misfires, which reduce engine efficiency and performance. Ensuring the correct gap is essential for proper operation.

:p What could happen if a spark plug’s gap is too wide?
??x
If a spark plug's gap is **too wide**, it could cause the spark plug to **misfire**.
x??

---
#### Intake Manifold Function

The intake manifold is crucial in ensuring that the right amount of air and fuel reaches each cylinder at the correct time. Its role is essential for maintaining engine performance.

:p What is the primary purpose of an intake manifold?
??x
The primary purpose of an intake manifold is to **distribute the air/fuel mixture** to the cylinders.
x??

---
#### Vehicle Part Identification

Identifying different components in a vehicle's cooling and fuel systems can help in diagnosing issues. Understanding each part’s role aids in effective troubleshooting.

:p Identify the vehicle part shown here.
??x
The part shown is most likely a **water pump**.
x??

---
#### Engine Power Source

When the engine is running, power needs to be supplied to the vehicle's electronics. The alternator fulfills this role by generating electrical energy and charging the battery.

:p What component keeps running power to a vehicle’s electronics when the engine is running?
??x
The component that keeps running power to a vehicle’s electronics when the engine is **running** is the **alternator**.
x??

---
#### Power Distribution System

A vehicle's transmission system manages how much of the engine's power reaches the wheels. Understanding this system helps in diagnosing drivetrain issues and improving performance.

:p The system that controls how much power goes to a vehicle’s wheels is its
??x
The system that controls how much power goes to a vehicle’s wheels is its **transmission**.
x??

---
#### Catalytic Converter Function

A catalytic converter reduces emissions by facilitating chemical reactions within the exhaust gases. Its role in converting harmful substances into less toxic ones is crucial for meeting emission standards.

:p What does a catalytic converter do?
??x
A catalytic converter **reduces toxic emissions**.
x??

---
#### Vehicle Frame Support

The frame of a vehicle supports its weight and helps keep it stable during operation. Components like the A-arms, shocks, and struts are integral to this structure.

:p What part of the vehicle supports its weight?
??x
The part that supports the vehicle's **weight** is the **frame**.
x??

---

---
#### Coolant Function and Operation
Coolant, typically a mixture of antifreeze and water, circulates through the radiator to regulate engine temperature. The coolant exposes itself to outside air in the radiator to allow heat energy to escape.

:p What is the primary function of coolant in an automotive system?
??x
The primary function of coolant is to maintain optimal operating temperatures by circulating through the radiator and transferring excess heat from the engine to the atmosphere, ensuring the engine operates within a safe temperature range.
x??

---
#### Purpose of Carburetor vs. Fuel-Injection System
A carburetor combines fuel and air in appropriate proportions for combustion, while modern vehicles use fuel-injection systems that achieve similar outcomes without the need for a physical carburetor.

:p What distinguishes a carburetor from a fuel-injection system?
??x
A carburetor physically mixes fuel with air to create an air-fuel mixture, which is then sent to the engine. In contrast, fuel-injection systems use computer-controlled injectors to precisely meter and atomize the fuel into the air stream directly at the intake valves.
x??

---
#### Role of Flywheel in Engine Performance
The flywheel stores rotational energy by accelerating a rotor to high speed and uses this stored energy to maintain constant engine speed. It works closely with other components like the crankshaft, connecting rod, and piston.

:p What is the primary role of a flywheel?
??x
The primary role of a flywheel is to store kinetic energy during periods when the engine's speed increases and then release that energy slowly to maintain stable engine speed. This works in conjunction with the crankshaft, connecting rods, and pistons.
x??

---
#### Function of Catalytic Converter
A catalytic converter converts harmful gases from the exhaust into less toxic substances through chemical reactions.

:p What is the main purpose of a catalytic converter?
??x
The main purpose of a catalytic converter is to reduce the emissions of harmful pollutants by facilitating chemical reactions that convert them into less toxic compounds before they are expelled through the vehicle's exhaust.
x??

---
#### Components of General Tuneup
A general automotive tuneup includes checking or replacing air and fuel filters, belts, spark plugs, battery, fluids, ignition timing, and tire pressure. For older vehicles, components like the positive crankcase ventilation valve and points may also need attention.

:p What does a general tuneup typically include?
??x
A general tuneup typically involves checking or replacing various components to ensure optimal vehicle performance. This includes air and fuel filters, belts, spark plugs, battery, fluids, ignition timing, tire pressure, and additional parts like the positive crankcase ventilation valve and points for older vehicles.
x??

---
#### Role of Antifreeze
Antifreeze raises the boiling point of water and lowers its freezing point to prevent the coolant from overheating or freezing. This is crucial for maintaining engine temperature in a wide range of weather conditions.

:p What does antifreeze do?
??x
Antifreeze increases the boiling point of water, preventing the coolant from boiling at high temperatures, and decreases the freezing point, ensuring it doesn't freeze at low temperatures. Together, these properties help maintain optimal engine temperature.
x??

---
#### Carburetor Definition and Usage
A carburetor is an assembly in older cars and machines that mixes fuel and air to the proper ratio for combustion.

:p What is a carburetor?
??x
A carburetor is an assembly used in older automotive engines to mix air and fuel into the correct proportions for combustion. It is less common in modern vehicles, which use fuel-injection systems.
x??

---
#### Capacitive Discharge Ignition System
Capacitive discharge ignitions store energy in a capacitor within the vehicle, releasing it on demand to the spark plug. This system ensures controlled ignition without relying on external battery power.

:p What does a capacitive discharge ignition do?
??x
A capacitive discharge ignition stores charged energy in a capacitor and releases it on-demand to the spark plug. This method of ignition doesn't require constant battery input, ensuring efficient use of stored electrical energy.
x??

---
#### Impact of Spark Plug Gap
A spark plug with an improperly set gap may fail to fire properly or misfire at high speeds.

:p What can happen if a spark plug's gap is too wide?
??x
If a spark plug has too wide a gap, it may not fire at all or it may misfire at high speeds. Proper setting of the gap ensures consistent and effective ignition for smooth engine operation.
x??

---
#### Intake Manifold Function
The intake manifold evenly distributes air/fuel mixture to each cylinder in an internal combustion engine.

:p What is the role of the intake manifold?
??x
The intake manifold's primary function is to distribute the air-fuel mixture uniformly among all cylinders, ensuring consistent fuel delivery and efficient engine performance.
x??

---
#### Role of Oil Pump
An oil pump circulates oil from the oil pan to the moving parts of the engine.

:p What does an oil pump do?
??x
The oil pump's role is to circulate lubricating oil from the oil pan through the engine to reduce friction and provide necessary cooling, ensuring smooth operation of all moving parts.
x??

---
#### Function of Alternator
The alternator converts mechanical energy into electrical energy, powering vehicle accessories and maintaining battery charge.

:p What does an alternator do?
??x
An alternator converts mechanical energy from the engine's rotation into electrical energy to power various vehicle accessories and keep the battery charged while the engine is running.
x??

---
#### Purpose of Transmission
The transmission ensures the right amount of power reaches the wheels at a specific speed by managing gear ratios.

:p What does the transmission do?
??x
The transmission controls the distribution of engine power to the wheels, allowing for different speeds and torque levels. Lower gears provide more power but slower speeds, while higher gears increase speed with less power.
x??

---
#### Role of Catalytic Converter in Exhaust System
A catalytic converter neutralizes toxic exhaust gases through chemical reactions.

:p What is the role of a catalytic converter?
??x
The role of a catalytic converter is to reduce harmful emissions by facilitating chemical reactions that convert toxic pollutants into less hazardous compounds.
x??

---
#### Function of Struts in Vehicle Suspension
Struts support the vehicle's weight and incorporate shock absorbers and coil springs, shifting the car’s weight onto its tires.

:p What do struts do?
??x
Struts support the vehicle's weight by integrating shock absorbers and coil springs. They help distribute the weight to the tires, ensuring a smooth ride and proper handling.
x??

---


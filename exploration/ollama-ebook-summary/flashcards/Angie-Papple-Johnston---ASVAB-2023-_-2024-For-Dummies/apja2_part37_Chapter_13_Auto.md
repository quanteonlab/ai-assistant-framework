# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 37)

**Starting Chapter:** Chapter 13 Auto Information. Chassis and frame Holding it all together

---

#### Understanding Auto Information on ASVAB
Background context: The Auto Information section of the ASVAB tests basic knowledge of automotive systems and malfunctions. It is a crucial part for those interested in automotive-related jobs but does not affect the AFQT score directly.

:p What are the key areas tested in the Auto Information subtest?
??x
The key areas tested include understanding basic automotive systems such as engine, transmission, and electrical systems, as well as recognizing common malfunctions in these systems. This section also covers the importance of maintaining proper tools and equipment in a shop setting.
x??

---

#### Time Management for Auto Information on ASVAB (CAT-ASVAB)
Background context: The CAT-ASVAB presents Auto Information questions within a limited time frame, requiring efficient use of time to answer all questions accurately.

:p How much time do you have to answer 10 Auto Information questions in the CAT-ASVAB version?
??x
In the CAT-ASVAB, you have 7 minutes to answer 10 Auto Information questions. This equates to about 35 seconds per question.
x??

---

#### Time Management for Auto & Shop Information (P&P Version)
Background context: The paper-and-pencil version of the ASVAB has a shorter time limit and more questions in the Auto & Shop Information subtest, testing both automotive systems and shop principles.

:p How many minutes do you have to answer 25 Auto & Shop Information questions in the P&P version?
??x
In the paper-and-pencil version, you have 11 minutes to answer 25 Auto & Shop Information questions. This gives about 44 seconds per question.
x??

---

#### Importance of AS Score for Certain Jobs
Background context: The AS score is used by branches of the military to determine qualifications for certain jobs but does not factor into the AFQT score.

:p Which part of the ASVAB measures your qualifications for certain automotive-related jobs?
??x
The Auto Information and Shop Information sections measure your qualifications for certain automotive-related jobs. These scores are only used in determining job eligibility within the military.
x??

---

#### Differentiation Between AS Score and AFQT
Background context: The AS score is specific to auto and shop information, while the AFQT score includes a broader range of test areas.

:p How does the AS score differ from the AFQT score?
??x
The AS score specifically measures knowledge in automotive systems and shop principles. It is used by military branches for job qualification purposes but is not part of the overall AFQT score which combines multiple subtests to determine general eligibility.
x??

---

#### Practical Knowledge of Automotive Systems
Background context: Understanding basic automotive systems like engines, transmissions, and electrical components can significantly improve performance on the ASVAB.

:p What kind of knowledge should you focus on for Auto Information questions?
??x
You should focus on understanding how basic automotive systems such as engines (fuel, ignition), transmissions (manual, automatic), and electrical systems (batteries, wiring) function and recognizing common malfunctions in these systems.
x??

---

#### Role of Tools and Shop Principles
Background context: Familiarity with shop tools and principles is also tested to ensure candidates understand the practical aspects of working in a repair environment.

:p What other areas are included in Auto Information besides automotive systems?
??x
Besides automotive systems, Auto Information includes knowledge about shop tools and basic shop principles. This ensures that test-takers understand the practical aspects required for working with vehicles.
x??

---

---
#### Chassis and Frame Overview
Background context: The chassis is a critical component of a vehicle, analogous to the skeleton providing structural support. It houses the engine, gearbox, axles, and other systems, ensuring everything stays together during operation.

:p What does the chassis provide in a vehicle?
??x
The chassis provides the framework that holds all components of the car together, ensuring they function effectively and safely.
x??

---
#### Unibody vs. Body-on-Frame Construction
Background context: Most modern vehicles use unibody construction where the frame and body are integrated into one piece. Older cars often used a body-on-frame approach with separate body and frame.

:p How does unibody construction differ from body-on-frame?
??x
In unibody construction, the chassis (including the frame) and body are integrated as a single unit. In contrast, in body-on-frame construction, the body is mounted on a separate frame.
x??

---
#### Engine Internal Combustion Process
Background context: An internal combustion engine converts chemical energy from fuel into mechanical energy to move the vehicle. This process involves several cycles with specific strokes.

:p What are the four strokes of an internal combustion engine?
??x
The four strokes of an internal combustion engine are:
1. Intake stroke
2. Compression stroke
3. Combustion stroke
4. Exhaust stroke
x??

---
#### Intake Stroke Details
Background context: The intake stroke is the first phase in a cycle where fuel and air mix to form an explosive mixture, which will be compressed and ignited later.

:p Describe what happens during the intake stroke.
??x
During the intake stroke, the intake valve opens as the connecting rod pulls the piston down from its TDC position. This creates a vacuum that draws in the air-fuel mixture into the cylinder. The piston reaches BDC, then the intake valve closes to seal the mixture inside the combustion chamber.

```java
// Pseudocode for Intake Stroke Logic
public void intakeStroke() {
    // Open intake valve
    if (pistonPosition == TDC) {
        intakeValve.open();
    } else {
        // Pull piston down, creating vacuum
        piston.down();
        intakeValve.close(); // Seal mixture in combustion chamber
    }
}
```
x??

---
#### Compression Stroke Explanation
Background context: The compression stroke involves compressing the air-fuel mixture to build up energy for the next phase.

:p What happens during the compression stroke?
??x
During the compression stroke, the connecting rod pushes the piston up, compressing the air-fuel mixture inside the combustion chamber. This compression builds up energy by exciting molecules and generating heat. The flywheel helps further compress the charge (the volume of compressed air-fuel mixture).

```java
// Pseudocode for Compression Stroke Logic
public void compressionStroke() {
    // Push piston up to compress air-fuel mix
    piston.up();
    flywheel.compress(chargeVolume);
}
```
x??

---
#### Combustion Stroke Process
Background context: The combustion stroke involves igniting the compressed mixture and converting its energy into mechanical work.

:p What happens during the combustion stroke?
??x
During the combustion stroke, the explosion forces the piston down. Since both intake and exhaust valves are closed, this explosion pushes the connecting rod to turn the crankshaft. The crankshaft turns the flywheel, which keeps the engine going.

```java
// Pseudocode for Combustion Stroke Logic
public void combustionStroke() {
    // Ignite air-fuel mix (piston near TDC)
    sparkPlug.releaseSpark();
    explosion.force(piston.down());
    crankshaft.turn(flywheel);
}
```
x??

---
#### Exhaust Stroke Description
Background context: The exhaust stroke expels the leftover gases from the previous combustion, preparing for the next cycle.

:p What happens during the exhaust stroke?
??x
During the exhaust stroke, the exhaust valve opens as the connecting rod moves the piston back up, pushing out the leftover gases. The valves are timed using push rods attached to the camshaft, ensuring complete expulsion of exhaust before the next intake stroke begins.

```java
// Pseudocode for Exhaust Stroke Logic
public void exhaustStroke() {
    // Open exhaust valve and expel gases
    if (pistonPosition == BDC) {
        exhaustValve.open();
        piston.up(); // Push out leftover gases
    }
}
```
x??

---

---
#### Fuel Injectors and Electronic Fuel Injection (EFI)
Fuel injectors replaced carburetors on newer cars to manage the air-fuel mixture. The first mass-produced fuel injectors appeared since the late 1950s, but they became widely used starting from the late 1980s and early 1990s.

The fuel injector acts as a nozzle that injects liquid fuel directly into the engine’s air stream, which is usually controlled by an external pump. Modern cars use electronic fuel injection (EFI) systems, where commands are sent to the powertrain control module (PCM). The PCM then determines the amount of fuel needed based on information from various sensors.

A throttle is either mechanically or electronically connected to the EFI computer and carburetor. Advancing (opening) the throttle causes more fuel transfer. This action is controlled by pushing harder on the gas pedal, which sends a signal through electrical connections to advance the throttle.

:p What are fuel injectors and how do they differ from carburetors?
??x
Fuel injectors replace carburetors in modern cars for managing air-fuel mixtures efficiently. They inject liquid fuel directly into the engine's air stream and require an external pump, whereas carburetors were simpler mechanisms that mixed fuel with air.
x??

---
#### Mass-Produced Cars and EFI Systems
All mass-produced cars today use electronic fuel injection (EFI) systems controlled by a powertrain control module (PCM). The PCM receives information from various sensors to determine the optimal amount of fuel needed for engine operation. This system replaces traditional carburetors, which were used in older vehicles.

:p What is the primary difference between EFI and carburetor systems?
??x
The primary difference lies in how they manage the air-fuel mixture. EFI uses electronic signals and sensors to control precise fuel delivery, while carburetors rely on mechanical adjustments to mix fuel with air.
x??

---
#### Internal Combustion Engine Components
Most internal combustion engines share common components within the engine block: pistons, piston rings, cylinders, cylinder heads, combustion chambers, intake valves, exhaust valves, crankshaft, camshafts, wrist pins, and connecting rods.

:p What are the primary components of an internal combustion engine?
??x
The primary components include pistons, which move up and down; piston rings that seal to prevent leaks; cylinders that house these parts; cylinder heads with intake and exhaust valves and ports; combustion chambers for mixing air and fuel; intake and exhaust valves themselves; a crankshaft that converts linear motion into rotational; camshafts that control valve operation; wrist pins connecting pistons to rods; and connecting rods that link the piston assembly to the crankshaft.
x??

---
#### Pistons and Their Role
Pistons are cylindrical objects with solid crowns. One moves up and down in each cylinder, converting energy from the air-fuel mixture's combustion into mechanical motion.

:p What is the role of pistons in an engine?
??x
Pistons play a critical role by moving up and down in the cylinders to convert the chemical energy from burning fuel-air mixtures into mechanical energy. This motion drives the crankshaft, which then powers the vehicle.
x??

---
#### Piston Rings and Their Function
Piston rings seal the piston to the cylinder wall. They prevent gases from leaking out and stop engine oil from entering the combustion chamber.

:p What is the purpose of piston rings?
??x
The primary purpose of piston rings is to ensure a tight seal between the piston and the cylinder wall, preventing gases from escaping and stopping engine oil from entering the combustion chamber.
x??

---
#### Cylinders and Combustion Process
Cylinders house pistons along with other components. They are crucial as they contain the air-fuel mixture that combusts to generate power.

:p What is the role of cylinders in an internal combustion engine?
??x
Cylinders are essential for containing the air-fuel mixture, where it burns and generates power. This process drives the pistons up and down, converting chemical energy into mechanical motion.
x??

---
#### Cylinder Heads and Valves
The cylinder head sits above the piston, housing parts like intake valves that let fuel-air mix enter and exhaust valves to expel waste gases.

:p What are the key components of a cylinder head?
??x
A cylinder head contains intake and exhaust valves, as well as the combustion chamber. It facilitates the entry of air-fuel mixture and the exit of waste gases.
x??

---
#### Combustion Chambers and Their Function
Combustion chambers are located inside the cylinder heads above pistons. They are where fuel-air mixtures ignite to produce energy.

:p What is the role of a combustion chamber in an engine?
??x
The combustion chamber is critical as it houses the location where fuel-air mixtures ignite, producing the necessary energy for the engine's operation.
x??

---
#### Intake and Exhaust Valves Functionality
Intake valves allow air-fuel mixture entry into the combustion chamber. After sealing when the mixture is inside, exhaust valves enable waste gases to exit after combustion.

:p What are intake and exhaust valves responsible for in an engine?
??x
Intake valves open to let the air-fuel mix enter the combustion chamber and close once the mixture is inside. Exhaust valves then open to allow the release of spent gases post-combustion.
x??

---
#### Crankshaft Operation
The crankshaft converts the piston's up-and-down motion into rotary motion, facilitating the engine's power output.

:p What does a crankshaft do in an internal combustion engine?
??x
A crankshaft transforms the linear motion of pistons moving up and down into rotational movement. This process is vital for generating the mechanical energy that powers the vehicle.
x??

---
#### Camshafts Operation and Timing
Camshafts operate intake and exhaust valves, turning at half the speed of the crankshaft to ensure proper timing for gas exchange.

:p What role do camshafts play in an engine?
??x
Camshafts control the opening and closing of intake and exhaust valves. They rotate at half the crankshaft's speed to manage when fuel enters and exhaust gases exit, ensuring efficient operation.
x??

---
#### Wrist Pins Functionality
Wrist pins connect pistons to connecting rods, allowing for the transfer of force between these components.

:p What is the function of wrist pins?
??x
Wrist pins link pistons to connecting rods, facilitating the transfer of force. They ensure that the linear motion of pistons is accurately converted into rotational energy via the crankshaft.
x??

---


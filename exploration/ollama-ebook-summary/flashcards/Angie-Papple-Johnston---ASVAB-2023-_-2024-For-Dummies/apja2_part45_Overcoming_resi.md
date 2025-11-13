# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 45)

**Starting Chapter:** Overcoming resistance. Relying on Machines to Help You Work

---

---
#### Potential Energy and Its Transitions
Potential energy is the stored energy an object possesses due to its position or state. When this energy is released, it transforms into kinetic energy, which is the energy of motion. For instance, a book held above a table has potential energy that converts to kinetic energy when dropped.
:p What happens to a book’s energy when it is dropped from a height?
??x
When a book is dropped, its gravitational potential energy (PE = mgh, where $m $ is mass,$g $ is acceleration due to gravity, and$h $ is height) is converted into kinetic energy (KE =$\frac{1}{2}mv^2 $, where $ v$ is velocity). The book’s potential energy decreases as it falls and its kinetic energy increases.
x??

---
#### Newton's First Law of Motion
Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force. This principle explains inertia, which is the tendency of objects to resist changes in their state of motion.
:p Explain Newton’s First Law of Motion.
??x
Newton's First Law states that a body remains at rest or continues moving at a constant velocity unless it is acted on by an external force. In simpler terms, objects tend to keep doing what they are doing until something else causes them to change their state.
x??

---
#### Mass and Inertia
Mass is the measure of inertia, which is an object’s resistance to changes in its motion. A more massive object has a greater tendency to resist changes in its state of motion compared to a less massive one.
:p How does mass relate to an object's tendency to resist changes?
??x
Mass directly correlates with an object's inertia; the higher the mass, the greater the inertia. This means that heavier objects are harder to accelerate or decelerate than lighter ones because they have more resistance to changes in their state of motion.
x??

---
#### Work and Force
Work is defined as a force acting on an object over a distance (displacement). It involves both force and displacement, making it easier to understand through practical scenarios. For example, lifting a barbell involves applying force over a certain height.
:p What are the two main ingredients in work according to physics?
??x
The two main ingredients in work are force and displacement. These must be present for work to occur; when you lift a barbell, you apply force over a distance (height), thus doing work on it.
x??

---
#### Frictional Resistance vs. Weight
Frictional resistance is the opposing force that occurs when surfaces rub against each other. It can differ from an object's weight, especially in scenarios where there are forces like friction acting against motion.
:p How does frictional resistance compare to an object’s full weight?
??x
Frictional resistance and an object's full weight are not always equivalent. For instance, moving a heavy piano across the floor involves overcoming friction rather than its entire weight. The actual force required can be less because it's the resistance of the surfaces (friction) that must be overcome.
x??

---

---
#### Power Definition and Calculation
Power is the rate at which work is done. Mathematically, it can be expressed as:
$$\text{Power} = \frac{\text{Work}}{\text{Time}}$$- Work is usually measured in joules (J).
- Time is measured in seconds (s).
- Power is then measured in watts (W).

Alternatively, power can also be measured in horsepower. For this unit, the formula changes to:
$$\text{Horsepower} = \frac{\text{Work}}{\text{Time}}$$

Where:
- Work is measured in foot-pounds.
- Time is measured in minutes.
- Power is in foot-pounds per minute.

1 horsepower is equivalent to 33,000 foot-pounds of work per minute. Also, 1 horsepower equals 550 foot-pounds of work per second or approximately 745.7 watts.

:p What is the formula for calculating power and how are different units used?
??x
The formula for calculating power in terms of joules and seconds is:
$$\text{Power (W)} = \frac{\text{Work (J)}}{\text{Time (s)}}$$

For horsepower, it’s:
$$\text{Horsepower} = \frac{\text{Work (ft-lbs)}}{\text{Time (min)}}$$

In both cases, the formula measures the rate at which work is done.

```java
public class PowerCalculation {
    public static double calculatePower(double workJoules, double timeSeconds) {
        return workJoules / timeSeconds;
    }

    public static double calculateHorsepower(double workFootPounds, double timeMinutes) {
        return (workFootPounds * 550) / (timeMinutes * 60);
    }
}
```

x??

---
#### Machines and Human Efficiency
Machines have been used by humans to increase efficiency since ancient times. The earliest known use of a machine was the wedge, which dates back about 2 million years ago.

:p What is the significance of early hominids in relation to machines?
??x
Early hominids like Homo habilis started using simple tools such as wedges, which helped them perform tasks more efficiently. This marks one of the earliest uses of machines by humans.

```java
public class EarlyHumanMachines {
    public static void main(String[] args) {
        // Simulate the use of a wedge to split wood
        int wedgeUsage = 10; // Example usage count
        System.out.println("Wedge was used " + wedgeUsage + " times to improve efficiency.");
    }
}
```

x??

---
#### Levers and Mechanical Advantage
Levers are simple machines that work by using a fulcrum to reduce resistance and multiply the effect of effort. The mechanical advantage (MA) of a lever is determined by:
$$\text{Mechanical Advantage} = \frac{\text{Length of Effort Arm}}{\text{Length of Resistance Arm}}$$- Effort arm: Distance from the fulcrum where effort is applied.
- Resistance arm: Distance from the fulcrum where resistance is applied.

The longer the effort arm compared to the resistance arm, the greater the mechanical advantage.

:p What is a lever and how does it work?
??x
A lever works by using a fulcrum (a pivot point) to reduce the force needed to move an object. It has three main parts:
- Effort: The input force applied.
- Fulcrum: The support point around which the lever rotates.
- Load/Resistance: The weight that needs to be moved.

The mechanical advantage can be calculated using:
$$\text{Mechanical Advantage} = \frac{\text{Length of Effort Arm}}{\text{Length of Resistance Arm}}$$

For example, if an effort arm is 6 inches and a resistance arm is 3 inches, the MA would be 2.

```java
public class LeverMechanics {
    public static void main(String[] args) {
        double effortArm = 6; // in inches
        double resistanceArm = 3; // in inches
        double mechanicalAdvantage = effortArm / resistanceArm;
        System.out.println("Mechanical Advantage: " + mechanicalAdvantage);
    }
}
```

x??

---
#### Types of Levers
Levers are classified into three types based on the position of the fulcrum:
1. **Class 1**: Fulcrum between effort and load (e.g., scissors, pliers).
2. **Class 2**: Load between effort and fulcrum (e.g., wheelbarrows, crowbars).
3. **Class 3**: Effort opposite to the load and beside the fulcrum (e.g., tweezers, staplers).

:p What are the three classes of levers?
??x
The three classes of levers are:
1. **Class 1** - Fulcrum between effort and load.
2. **Class 2** - Load between effort and fulcrum.
3. **Class 3** - Effort opposite to the load and beside the fulcrum.

Each type has unique characteristics based on where the fulcrum is located:
- Class 1 levers use the fulcrum to increase force but decrease distance (e.g., scissors).
- Class 2 levers increase the distance the object moves but require more effort (e.g., wheelbarrows).
- Class 3 levers provide a trade-off by increasing speed and decreasing force (e.g., tweezers).

```java
public class LeverTypes {
    public static void main(String[] args) {
        System.out.println("Class 1 lever example: Scissors");
        System.out.println("Class 2 lever example: Wheelbarrow");
        System.out.println("Class 3 lever example: Tweezers");
    }
}
```

x??

---
#### Inclined Plane
An inclined plane, also known as a ramp, is another simple machine that makes moving an object easier. It spreads the work over a longer distance, reducing the force needed to do the work.$$\text{Mechanical Advantage} = \frac{\text{Length of Ramp}}{\text{Height of Ramp}}$$:p What is an inclined plane and how does it work?
??x
An inclined plane or ramp makes moving heavy objects easier by spreading out the effort over a longer distance. For example, to lift a 50-pound barrel 3 feet using a 6-foot long ramp, you would only need half as much force (25 pounds) because the mechanical advantage is 2.

```java
public class InclinedPlaneExample {
    public static void main(String[] args) {
        double lengthOfRamp = 6; // in feet
        double heightOfRamp = 3; // in feet
        double mechanicalAdvantage = lengthOfRamp / heightOfRamp;
        System.out.println("Mechanical Advantage: " + mechanicalAdvantage);
    }
}
```

x??

#### Inclined Planes (Ramps and Wedges)
Background context: Inclined planes, such as ramps or wedges, can reduce the force needed to move an object. The formula for the mechanical advantage of a ramp is given by:
$$\text{Mechanical Advantage} = \frac{\text{Length of Ramp}}{\text{Height of Ramp}}$$

Wedges are a type of inclined plane that can be used to split or lift heavy objects.

:p What is the mechanical advantage of an object being moved up a 10-foot ramp with a height of 2 feet?
??x
The mechanical advantage would be:
$$\frac{\text{Length of Ramp}}{\text{Height of Ramp}} = \frac{10\, \text{feet}}{2\, \text{feet}} = 5$$
This means that the force required to move the object up the ramp is reduced by a factor of 5.

x??

---

#### Screw Jacks
Background context: A screw jack combines the properties of an inclined plane (the thread of the screw) and a lever. It can be used to lift heavy objects, such as houses for foundation work.

:p How does a screw jack function?
??x
A screw jack works by using the spiral thread of the screw as an inclined plane to distribute the force applied to it over a greater distance, allowing it to lift heavy weights. The lever aspect comes into play when you turn the screw, which applies torque that is magnified and converted into vertical lifting force.

x??

---

#### Pulleys
Background context: Pulleys are simple machines used to change the direction of a force or to multiply the applied force. They can be fixed (stationary) or movable.

:p What is the difference between a fixed pulley and a movable pulley?
??x
A fixed pulley changes only the direction of the force, not its magnitude. It has a mechanical advantage of 1 because the effort required is equal to the load being lifted. A movable pulley, on the other hand, can provide a mechanical advantage greater than 1 by distributing the load over multiple sections of rope.

x??

---

#### Block and Tackle Systems
Background context: Block and tackle systems use multiple pulleys to increase the mechanical advantage, making it easier to lift heavy objects. The system reduces the effort needed but increases the distance you must pull the rope.

:p How does a block and tackle system provide a mechanical advantage?
??x
A block and tackle system provides a mechanical advantage by using multiple pulleys in series. Each additional pulley section halves the force required, but doubles the distance over which you must apply that force. For example, with two sections of rope, pulling 2 feet of rope raises the load 1 foot.

```java
// Example Pseudocode for Block and Tackle System
public class BlockAndTackle {
    public void liftBox(int weight) {
        int mechanicalAdvantage = 2; // For a 2:1 system
        int effortRequired = weight / mechanicalAdvantage;
        int distanceToPull = 2 * (weight - effortRequired); // Since each section of rope must be pulled twice as much
    }
}
```

x??

---

#### Ratchets in Block and Tackle Systems
Background context: Ratchets or trapped roller bearings are used in block and tackle systems to allow controlled movement only in one direction, preventing backsliding.

:p What is the role of a ratchet mechanism in a block and tackle system?
??x
A ratchet mechanism ensures that you can apply continuous effort over short distances without having to reposition the load each time. It locks the rope or pulley after a certain amount of movement, so it does not slip back when you release the tension.

x??

---

#### Mechanical Advantage in Block and Tackle Systems
Background context: The mechanical advantage (MA) of a block and tackle system is determined by the number of rope sections supporting the load. Each additional section doubles the MA but also doubles the distance over which force must be applied.

:p How does adding more pulleys to a block and tackle arrangement increase its mechanical advantage?
??x
Adding more pulleys to a block and tackle arrangement increases the mechanical advantage because each additional pulley divides the weight of the load among more sections of rope. For example, with one section of rope, the MA is 1 (no change). With two sections, the MA becomes 2 (half the effort required). With four sections, the MA is 4, and so on.

x??

---

These flashcards cover various concepts related to mechanical skills using simple machines like ramps, wedges, screw jacks, pulleys, block and tackle systems, and ratchets.


# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 112)

**Starting Chapter:** Overcoming resistance. Relying on Machines to Help You Work

---

---
#### Potential and Kinetic Energy
Potential energy is stored energy due to an object's position or state, which can be converted into kinetic energy (energy of motion). When a book is lifted and then dropped, its potential energy turns into kinetic energy as it falls. The conversion between these forms of energy follows the law of conservation of energy.
:p Explain the transformation from potential energy to kinetic energy when an object is dropped.
??x
When an object such as a book is lifted, gravitational potential energy (PE) is stored due to its position relative to the ground. Once released and allowed to fall, this PE converts into kinetic energy (KE), which increases with velocity until impact. The relationship between these forms of energy can be understood through the equation:
$$\text{PE} = mgh$$where $ m $is mass,$ g $is gravitational acceleration, and$ h $is height. At ground level ($ h=0$), all PE becomes KE as described by:
$$\text{KE} = \frac{1}{2}mv^2$$

This conversion can be demonstrated with a simple example: if you lift a book to a certain height and then drop it, the potential energy at the top equals the kinetic energy just before impact.
x??
---

---
#### Newton's First Law of Motion
Newton’s first law states that an object in motion stays in motion and an object at rest stays at rest unless acted upon by an unbalanced force. This is often referred to as the principle of inertia, which describes how objects resist changes to their state of motion.
:p Describe the principle of inertia according to Newton's First Law.
??x
Newton’s first law asserts that a body remains in its state of uniform motion or rest unless it is acted upon by an external force. The key here is “unbalanced force,” meaning any net force acting on the object will change its velocity or direction. Inertia is the property of matter to resist changes to its state of motion, and mass is directly related to inertia.
For instance, a car moving at a constant speed resists slowing down unless there is an unbalanced force (like friction or braking). Similarly, a stationary box remains still until pushed by some external force. 
x??
---

---
#### Frictional Resistance vs. Weight
Friction is the resistance that occurs when two surfaces rub against each other. While weight is the gravitational pull on an object, the force required to move an object can be less than its full weight due to friction.
:p Explain how friction affects the amount of work needed to move an object compared to its weight.
??x
When moving objects, such as pushing a piano across a floor or carrying it up stairs, the resistance felt is mainly from friction rather than the full gravitational pull (weight). Frictional forces can be less than the weight of the object. For example:
- Pushing a piano across a smooth surface requires overcoming the friction between the surface and the bottom of the piano.
- Carrying the same piano up stairs involves lifting it against gravity, which is heavier due to its full weight.

Mathematically, frictional force ($F_f$) can be represented by:
$$F_f = \mu N$$where $ N $ is the normal force (which equals the object’s weight in many cases) and $\mu$ is the coefficient of friction. Thus, in some scenarios like pushing a book across carpet, lifting it up might require less energy due to reduced friction.
x??
---

---
#### Work in Physics
In physics, work involves applying a force over a distance, often resulting from displacement. It’s measured in joules and is the product of force and displacement.
:p Define what constitutes "work" in the context of physics.
??x
Work ($W $) is defined as the amount of energy transferred when a force ($ F $) acts on an object over some distance ($ d$). The formula for work is:
$$W = F \cdot d \cdot \cos(\theta)$$where $ F $is the magnitude of the applied force,$ d $is the displacement in the direction of the force, and$\theta$ is the angle between them.

For example, carrying groceries from a car to a kitchen involves doing work because you apply force over a certain distance. Dragging a duffel bag across the floor also counts as work due to the force applied over the distance.
x??
---

---
#### Power Definition and Units
Power is the rate at which work is done. Mathematically, it can be expressed as:
$$\text{Power} = \frac{\text{Work}}{\text{Time}}$$

In many machines, power might also be measured in horsepower (hp), where 1 hp equals 33,000 foot-pounds of work per minute or approximately 745.7 watts.

:p What is the definition and unit of power?
??x
Power is defined as the rate at which work is done. In practical applications, it can be measured in watts (W), where $\text{Power} = \frac{\text{Work}}{\text{Time}}$. Additionally, horsepower (hp) is commonly used, with 1 hp equivalent to 33,000 foot-pounds of work per minute or about 745.7 watts.

```java
public class PowerExample {
    public static void main(String[] args) {
        double work = 66000; // in foot-pounds
        double time = 2; // in minutes
        double powerHP = work / time / 33000; // converting to horsepower
        System.out.println("Power (in HP): " + powerHP);
    }
}
```
x??
---

#### Machines and Work Efficiency
Machines have been used by humans since early hominids like Homo habilis, approximately 2 million years ago. They help increase efficiency and perform tasks that couldn’t be done otherwise.

:p What are machines used for according to the text?
??x
Machines are used to make work easier by increasing efficiency and performing tasks that could not be accomplished manually. They provide a trade-off between force applied and distance over which it is applied, enabling you to use less effort for more effective results.

```java
public class MachineExample {
    public static void main(String[] args) {
        double inputForce = 10; // in pounds
        double outputForce = 50; // in pounds (assuming a simple machine multiplies the force)
        System.out.println("Mechanical Advantage: " + (outputForce / inputForce));
    }
}
```
x??
---

#### Levers and Mechanical Advantage
Levers are among the simplest machines used to help increase force. They work by using a fulcrum (point of support) to reduce resistance and multiply the effect of effort.

The formula for determining mechanical advantage with levers is:
$$\text{Mechanical Advantage} = \frac{\text{Length of Effort Arm}}{\text{Length of Resistance Arm}}$$

Levers fall into three classes: Class 1, Class 2, and Class 3. Each class has a different placement of the fulcrum.

:p What is a lever and how does it work?
??x
A lever is a simple machine that uses a fulcrum (a point of support) to multiply force. The effort is applied at one end (effort arm), while resistance is exerted at the other end (resistance arm). The mechanical advantage can be calculated by dividing the length of the effort arm by the length of the resistance arm.

```java
public class LeverExample {
    public static void main(String[] args) {
        double effortArmLength = 6; // in feet
        double resistanceArmLength = 3; // in feet
        double mechanicalAdvantage = effortArmLength / resistanceArmLength;
        System.out.println("Mechanical Advantage: " + mechanicalAdvantage);
    }
}
```
x??
---

#### Classes of Levers

- **Class 1 Lever**: Fulcrum is between the effort and load. Examples include scissors, pliers, and oars.
- **Class 2 Lever**: Load is between the effort and fulcrum. Examples include wheelbarrows, crowbars, and nutcrackers.
- **Class 3 Lever**: Fulcrum is beside the effort and opposite the load. Examples include tweezers, staplers, and brooms.

:p What are the different classes of levers?
??x
Levers can be classified into three types based on their arrangement:
1. **Class 1 Lever** - The fulcrum is between the effort and load (e.g., scissors, pliers, oars).
2. **Class 2 Lever** - The load is between the effort and fulcrum (e.g., wheelbarrows, crowbars, nutcrackers).
3. **Class 3 Lever** - The fulcrum is beside the effort and opposite the load (e.g., tweezers, staplers, brooms).

```java
public class LeverClassExample {
    public static void main(String[] args) {
        System.out.println("Class 1: Scissors, Pliers, Oars");
        System.out.println("Class 2: Wheelbarrows, Crowbars, Nutcrackers");
        System.out.println("Class 3: Tweezers, Staplers, Brooms");
    }
}
```
x??
---

#### Inclined Plane as a Simple Machine

The inclined plane, also known as a ramp, is another simple machine that makes moving an object from one point to another easier. It spreads the work over a longer distance, thus requiring less force.

For example, lifting a 50-pound barrel to a truck bed 3 feet off the ground requires 50 pounds of force for 3 feet of travel. Using a 6-foot ramp instead would only require half as much force (25 pounds) because the mechanical advantage is 2.

:p What is an inclined plane, and how does it function?
??x
An inclined plane, also called a ramp, is a simple machine that makes moving objects easier by spreading out the work over a longer distance. This reduces the required force. For instance, lifting a 50-pound barrel to a truck bed 3 feet off the ground typically requires 50 pounds of force for 3 feet. However, using a 6-foot ramp would only require half as much force (25 pounds) because the mechanical advantage is 2.

```java
public class InclinedPlaneExample {
    public static void main(String[] args) {
        double weight = 50; // in pounds
        double height = 3; // in feet
        double rampLength = 6; // in feet
        double forceRequired = (weight * height) / rampLength;
        System.out.println("Force Required: " + forceRequired);
    }
}
```
x??
---

#### Inclined Planes: Ramps, Wedges, and Screws
Background context explaining inclined planes. Include any relevant formulas or data here.
Inclined planes are simple machines that allow objects to be moved with less force than would be required if the object were lifted vertically. The advantage of using a ramp can be expressed as:
$$\text{Advantage} = \frac{\text{Length of Ramp}}{\text{Height of Ramp}}$$

Wedges are a form of inclined plane and can multiply your effort in much the same way that a ramp can. Screws are also inclined planes, only in spiral form.

Screw jacks, which you can use to lift your house up to build a new foundation, are a combination of a lever and an inclined plane.
:p What is the advantage formula for using an inclined plane?
??x
The advantage formula for using an inclined plane is given by:
$$\text{Advantage} = \frac{\text{Length of Ramp}}{\text{Height of Ramp}}$$

This means that if you have a ramp with a length of 10 feet and a height of 2 feet, the advantage would be $\frac{10}{2} = 5$. This indicates that using this inclined plane can reduce the force required to move an object by a factor of 5.
x??

---

#### Pulleys and Gears: Changing Force Magnitude and Direction
Background context explaining pulleys and gears. Include any relevant formulas or data here.
Pulleys and gears are simple machines used to change the magnitude (size) and direction of force. They are commonly found in elevators, escalators, cars, and watches.

:p How do pulleys and gears help in changing the force?
??x
Pulleys and gears help in changing the force by altering its magnitude and direction. For instance, when you use a single fixed pulley to lift an object, although it does not multiply the force, it changes the direction of the applied force, making lifting easier.

In the case of gear systems, they can be used to increase or decrease the rotational speed of a shaft while maintaining the same power input. The relationship between the number of teeth on two gears and their angular velocities is given by:
$$\frac{n_1}{n_2} = \frac{\text{Number of Teeth on Gear 2}}{\text{Number of Teeth on Gear 1}}$$where $ n_1 $ and $ n_2$ are the rotational speeds of the first and second gears, respectively.

:p What is the formula for the relationship between two gears?
??x
The formula for the relationship between two gears is:
$$\frac{n_1}{n_2} = \frac{\text{Number of Teeth on Gear 2}}{\text{Number of Teeth on Gear 1}}$$

This equation shows that if one gear has more teeth than the other, it will rotate at a slower speed while transmitting the same amount of power. Conversely, a smaller gear with fewer teeth will rotate faster.

:p Example: How do gears affect the rotational speed?
??x
Consider two gears where Gear 1 has 20 teeth and Gear 2 has 40 teeth. If Gear 1 is rotating at 60 rpm (revolutions per minute), then using the formula:
$$\frac{n_1}{n_2} = \frac{40}{20}$$we find that $ n_2 = \frac{60 \times 20}{40} = 30$ rpm. This means Gear 2 will rotate at half the speed of Gear 1 while maintaining the same power input.

:x??

---

#### Block and Tackle Systems: Reducing Effort
Background context explaining block and tackle systems. Include any relevant formulas or data here.
Block and tackle systems use multiple pulleys to reduce the effort required to lift heavy objects by distributing the force over a longer distance. They can also be used to change the direction of your pull.

:p How does a simple pulley affect lifting an object?
??x
A simple pulley doesn’t multiply your force but makes the process of lifting easier by allowing you to redirect the force. If you tie a 200-pound crate to one end of a rope, run the rope through a single fixed pulley, and grab the other end, you can pull down on the rope to lift the crate up. Without a pulley, you could pull down on the crate all day, and it wouldn’t go up.

In this case, using a simple pulley, the force of your pull must equal the weight of the object being lifted:
$$\text{Force} = \text{Weight of Object}$$:p Example: How does adding a block and tackle system reduce effort?
??x
Consider lifting a 200-pound crate with a block and tackle system. If you use two pulleys, each section of rope is supporting only half the weight (100 pounds). To lift the crate, you would need to pull on the rope with just 100 pounds of force.

Using a block and tackle system provides a mechanical advantage:
$$\text{Mechanical Advantage} = \frac{\text{Total Number of Rope Segments}}{1}$$

In this case, if there are two segments of rope (one through each pulley), the mechanical advantage is 2. To lift the crate 1 foot, you would need to pull 2 feet of rope.

:p What is a block and tackle system used for?
??x
A block and tackle system is used to reduce effort by lifting heavy objects more easily. It works by distributing the force over multiple pulleys, thereby requiring less effort to lift an object. The mechanical advantage can be increased by adding more pulleys, making it easier to perform the work.

:p How does a block and tackle system provide mechanical advantage?
??x
A block and tackle system provides mechanical advantage by reducing the effort required to lift heavy objects through multiple pulleys. Each additional pulley in the system increases the mechanical advantage. For example, if you have a 2:1 pulley system (two segments of rope), the mechanical advantage is 2, meaning that you need only half the force to lift the object.

:p What are some practical applications of block and tackle systems?
??x
Practical applications of block and tackle systems include lifting heavy objects like house foundations using screw jacks. By threading a rope through multiple pulleys, you can reduce the effort required to lift the object. The system works by distributing the force over a longer distance, making it easier to handle heavy loads.

:x??

---


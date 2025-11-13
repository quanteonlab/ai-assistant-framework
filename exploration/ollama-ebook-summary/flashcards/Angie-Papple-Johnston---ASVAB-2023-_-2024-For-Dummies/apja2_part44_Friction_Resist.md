# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 44)

**Starting Chapter:** Friction Resisting the urge to move

---

---
#### Gravity: What Goes Up Must Come Down
Background context explaining gravity. Isaac Newton's law of universal gravitation states that every object in the universe attracts every other object with a force proportional to the product of their masses and inversely proportional to the square of the distance between them.

$$F = G \frac{m_1 m_2}{r^2}$$where $ F $is the gravitational force,$ G $is the gravitational constant,$ m_1 $and$ m_2 $are the two masses, and$ r$ is the distance between their centers. The force of gravity acting on an object is equal to its weight.

:p What is Newton's law of universal gravitation?
??x
Newton's law states that every object in the universe attracts every other object with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. This is represented by the formula $F = G \frac{m_1 m_2}{r^2}$.
x??

---
#### Applied Force
Background context explaining applied force, which is any type of force that a person or another object exerts on something.

:p What is an example of applied force?
??x
An example of applied force is when you push or pull an object. For instance, applying force to the mop when mopping a parking lot or using a hand to throw a grenade.
x??

---
#### Tension Force
Background context explaining tension force, which is the force transmitted through a rope, string, or wire when force is applied to both ends.

:p What is tension force?
??x
Tension force is the force that acts along a rope, string, or wire and pulls equally on objects at both ends. This force is usually measured in either pounds-force (lbf) or newtons (N). For example, applying force to a rope that is pulling two boxes will result in tension forces acting on each box.
x??

---
#### Centrifugal Force: False Gravity
Background context explaining centrifugal force as the apparent outward force experienced by an object moving along a curved path. It's actually a fictitious force due to inertia.

:p What is centrifugal force?
??x
Centrifugal force is a concept that describes the apparent outward force experienced by objects in circular motion, even though it is a fictitious force arising from the inertia of the object. The amount of centrifugal force depends on the mass of the object, its speed, and its distance from the center.

$$F_{\text{centrifugal}} = m \omega^2 r$$where $ m $is the mass of the object,$\omega $ is the angular velocity, and$r$ is the radius. This force is perpendicular to the centripetal force directed towards the center of rotation.
x??

---

---
#### Friction: Resisting the Urge to Move
Background context explaining that friction is a force that resists the movement of one surface against another. It can be thought of as a resisting force when moving objects, such as pushing a box across different surfaces.

:p What is friction and how does it affect the motion of an object?
??x
Friction is a force that resists the relative motion of solid surfaces, fluid layers, or material elements sliding against each other. When you push a box on a smooth floor versus a carpeted floor, more friction will be present in the latter scenario, requiring more force to move the same distance.

In physics terms:
- **Static Friction** is the force that prevents an object from starting to slide across a surface.
- **Kinetic Friction** is the force that resists the motion of an object sliding across a surface.

The amount of friction depends on the materials in contact and the normal force pressing them together. The formula for kinetic friction (assuming a constant coefficient of friction) is:
$$f_k = \mu_k N$$where $ f_k $is the force of kinetic friction,$\mu_k $ is the coefficient of kinetic friction, and$N$ is the normal force.

C/Java code for calculating simple friction:
```java
public class FrictionCalculator {
    public double calculateFriction(double normalForce, double muK) {
        return muK * normalForce;
    }
}
```
x??

---
#### Rolling vs Sliding Friction
Background context explaining that rolling friction is always less than sliding friction. It discusses the ease of movement when using wheels (like on a dolly), comparing it to moving objects without wheels.

:p Why are wheels easier to use for moving objects compared to dragging them?
??x
Wheels reduce the amount of force required to move an object because rolling friction is significantly lower than sliding friction. When you push a box across a smooth floor, the rolling friction of the wheels is much less than if you were trying to drag the same box on its flat bottom.

The formula for the coefficient of rolling friction (CRR) can be expressed as:
$$\text{CRR} = \frac{F_{roll}}{N}$$where $ F_{roll}$is the force required to roll the object and $ N$ is the normal force.

Using a wheeled dolly, the forces involved are reduced because of the lower rolling friction compared to sliding. The exact reduction depends on the specific materials in contact.

Example:
```java
public class RollingFriction {
    public double calculateRollingForce(double CRR, double normalForce) {
        return CRR * normalForce;
    }
}
```
x??

---
#### Elastic Recoil: The Trampoline of Physics
Background context explaining that elastic recoil is the force a solid object exerts to return to its original shape when deformed. It relates this concept to objects like springs and trampolines.

:p What is elastic recoil, and how does it work?
??x
Elastic recoil (or spring force) describes the restoring force that tries to return an object to its original shape after being deformed. This principle can be seen in materials that resist changes in their form due to applied forces.

Mathematically, the relationship between the force exerted by a spring and the distance it is displaced from its equilibrium position is described by Hooke's Law:
$$F = -kx$$where $ F $is the restoring force,$ k $ is the spring constant (a measure of how stiff or elastic the material is), and $ x$ is the displacement.

Example: A cat on a board.
- When the cat steps off a bent board, the board will try to return to its original shape due to elastic recoil. The force required for this deformation and restoration depends on the spring constant of the board and how much it was displaced ($x$).

C/Java code example:
```java
public class ElasticRecoil {
    public double calculateRestoringForce(double k, double displacement) {
        return -k * displacement;
    }
}
```
x??

---
#### Magnetism: The Law of Attraction
Background context explaining that magnetism is the force generated by magnets to attract or repel other objects. It focuses on ferromagnetism as a strong type of magnetic property involving iron.

:p What is magnetism, and what are some common types?
??x
Magnetism is the physical phenomenon arising from the forces between electrically charged particles that result in attractive (attraction) or repulsive (repulsion) interactions. Ferromagnetism, which involves materials like iron, is a strong type of magnetic property where domains within the material align to create a net magnetization.

Key points about ferromagnetism:
- **Ferromagnetic Materials**: Strongly attracted by magnets.
- **Diamagnetic Materials**: Weakly repelled by magnets.
- **Paramagnetic Materials**: Weakly attracted by magnets.

Example: A simple model of magnetic attraction and repulsion can be visualized with a diagram where different materials react to a magnet based on their type (ferromagnetic, diamagnetic, paramagnetic).

C/Java code for basic magnet interaction:
```java
public class Magnetism {
    public boolean isFerromagnet(String material) {
        return "Iron".equals(material) || "Nickel".equals(material) || "Cobalt".equals(material);
    }
}
```
x??

---

---
#### Magnets and Magnetic Fields
Background context: All magnets have north and south poles. Opposite poles are attracted to each other, while like poles repel. When you rub a piece of iron across a magnet, the north-seeking poles align, creating a magnetic field that can attract or repel other objects.

:p Explain how a magnetic field is created by rubbing a magnet on iron.
??x
A magnetic field is generated when you rub a magnet on iron because the rubbing process realigns the electrons in the iron. Specifically, all the north-seeking poles (atoms with unpaired electrons) align in the same direction. This alignment creates a net magnetic field that can attract or repel other objects.

For example:
```java
public class Magnet {
    private boolean[] electronSpins; // Array representing electron spins

    public void rubWithMagnet(int length, int direction) {
        for (int i = 0; i < length; i++) {
            electronSpins[i] = direction == NORTH_POLE; // Align electrons with north pole
        }
    }

    private static final int NORTH_POLE = true;
}
```
x?
---

#### Atoms and Electric Charges
Background context: Substances are made of atoms, which have electrons carrying electric charges. These electrons spin and circle the nucleus, generating an electric current that makes each electron act like a tiny magnet.

:p Describe how each atom can behave as a tiny magnet due to its electrons.
??x
Each atom behaves as a tiny magnet because its electrons spin and orbit the nucleus in a way that creates an electric current. This current generates a magnetic field around the atom. However, most substances cancel out their net magnetism because half of the electrons spin in one direction while the other half spin in the opposite direction.

For example:
```java
public class Atom {
    private boolean electronSpin; // True for north-seeking pole

    public void spinElectrons(boolean direction) {
        this.electronSpin = direction;
    }

    public boolean getMagnetism() {
        return this.electronSpin;
    }
}
```
x?
---

#### Static Electricity and Opposite Charges
Background context: Static electricity, or electrical force, is the buildup of charges on a substance's surface. Opposite charges attract each other, allowing for phenomena such as a balloon sticking to a wall after being rubbed.

:p Explain how opposite charges attract in static electricity.
??x
In static electricity, opposite charges attract each other because they have different signs (positive and negative). When you rub a balloon on your hair, it gains electrons, making its surface negatively charged. The wall usually has a positive charge or is neutral. Since the balloon and the wall have opposite charges, they are attracted to each other.

For example:
```java
public class Balloon {
    private int charge;

    public void rubWithHair(int charge) {
        this.charge = -charge; // Negative charge indicates it gained electrons
    }

    public boolean isAttractedToWall() {
        return Wall.getCharge() * this.charge < 0; // Opposite charges attract
    }
}

public class Wall {
    private int charge;

    public void setCharge(int charge) {
        this.charge = charge;
    }

    public static int getCharge() {
        return -1; // Example negative charge, opposite of balloon's positive charge
    }
}
```
x?
---

#### Drag Forces and Streamlining
Background context: Drag forces act to slow down objects moving through a fluid (like air or water). The amount of drag depends on the object’s shape and the substance it moves through. Streamlined shapes reduce drag, making vehicles more efficient.

:p Describe how streamlining reduces drag for vehicles.
??x
Streamlining reduces drag by minimizing resistance when an object moves through a fluid like air or water. Streamlined shapes are designed to have smooth curves and surfaces that allow fluids to flow around them with minimal turbulence. This design minimizes the pressure difference between the front and back of the object, reducing the force acting against its motion.

For example:
```java
public class Vehicle {
    private double dragCoefficient;
    private boolean streamlinedShape;

    public void setStreamlinedShape(boolean shape) {
        this.streamlinedShape = shape;
    }

    public double getDragForce(double speed, double density) {
        return 0.5 * this.dragCoefficient * density * Math.pow(speed, 2);
    }
}
```
x?
---

#### Work and Energy
Background context: Work is the result of a force (usually in pounds) moving over a distance (usually in feet). In the US, work is often measured as foot-pounds. Potential energy is stored energy due to an object's position or state, while kinetic energy is the energy of motion.

:p Define what constitutes work in mechanical terms.
??x
Work is defined as the result of a force overcoming resistance over a distance. It can be mathematically represented by the formula:

$$\text{Work} = \text{Force} \times \text{Distance}$$

In the US, this is often measured in foot-pounds (1 foot-pound occurs when 1 pound moves 1 foot). The unit of work here is the foot-pound.

For example:
```java
public class Work {
    public static final double GAIN_IN_FOOT_POUNDS = 1; // 1 foot-pound of work

    public static void calculateWork(double force, double distance) {
        System.out.println("Work done: " + (force * distance) + " foot-pounds");
    }
}
```
x?
---


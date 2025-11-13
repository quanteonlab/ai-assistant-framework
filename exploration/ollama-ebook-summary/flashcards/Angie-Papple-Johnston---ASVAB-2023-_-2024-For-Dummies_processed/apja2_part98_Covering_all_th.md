# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 98)

**Starting Chapter:** Covering all the bases and the acids too

---

#### Chemical Formulas and Functional Groups

Background context: Understanding chemical formulas is essential for identifying components of substances. Functional groups are specific clusters of atoms within molecules that determine a molecule’s properties. The number of these functional groups is denoted by subscripts.

:p What do the symbols in a chemical formula represent?
??x
In a chemical formula, elements are represented by their symbols (e.g., C for carbon, H for hydrogen), and numbers indicate the quantity of each atom present in one molecule. Functional groups like NO3 and CHO2 are specific arrangements of atoms that confer particular characteristics to molecules.
x??

---
#### Sugar's Molecular Formula

Background context: The molecular formula of sugar is CHO12 22 11, indicating it consists of 12 carbon atoms, 22 hydrogen atoms, and 11 oxygen atoms. Functional groups like these determine the substance’s properties.

:p What does the molecular formula CHO12 22 11 tell us about sugar?
??x
The molecular formula CHO12 22 11 tells us that one molecule of sugar contains 12 carbon, 22 hydrogen, and 11 oxygen atoms. The specific arrangement (functional groups) of these atoms gives sugar its unique properties.
x??

---
#### Mixtures in Chemistry

Background context: A mixture is a combination of pure substances with varying compositions depending on the source. Each component retains its chemical characteristics within the mixture.

:p How does bread illustrate the concept of mixtures?
??x
Bread illustrates the concept of mixtures because even though all ingredients (flour, oil, water) are the same, different bakers can create variations in taste due to slight differences in their mixtures. Each ingredient retains its chemical properties throughout the process.
x??

---
#### Chemical Reactions

Background context: A chemical reaction involves a rearrangement of atoms within substances, leading to new products with altered molecular structures.

:p What is the key difference between boiling and freezing versus chemical reactions?
??x
Boiling and freezing are physical changes that alter a substance's state but do not change its molecular structure. In contrast, chemical reactions involve rearranging atomic bonds, resulting in entirely different molecules.
x??

---
#### Types of Chemical Reactions

Background context: There are several types of chemical reactions, including combination, decomposition, and combustion.

:p Explain the concept of a combination reaction using an example.
??x
A combination reaction occurs when two or more reactants merge to form one product. For instance, sodium (Na) and chlorine (Cl) combine to create sodium chloride (table salt), represented as 2 Na + Cl2 → 2 NaCl.
x??

---
#### Decomposition Reactions

Background context: In decomposition reactions, a compound breaks down into simpler substances.

:p Provide an example of a decomposition reaction.
??x
Water decomposes into hydrogen and oxygen gases. The chemical equation is 2 H2O → 2 H2 + O2.
x??

---
#### Combustion Reactions

Background context: Combustion reactions typically involve the burning of carbon-containing materials with oxygen.

:p Describe what happens during a combustion reaction in your car.
??x
During a combustion reaction in a car, fuel (a hydrocarbon) and air are ignited by a spark, combining with oxygen to produce heat, light, and exhaust gases like CO2 and H2O. The overall chemical equation can be simplified as: CxHy + O2 → CO2 + H2O + energy.
x??

---

#### pH Scale and Acid-Base Properties
Background context explaining how acids, bases, and neutral substances interact with water. The pH scale ranges from 0 to 14, where 0 is highly acidic (like battery acid), 7 is neutral (pure water), and 14 is highly basic (like liquid drain cleaner). Each whole number unit in the acidic range represents a tenfold increase in acidity compared to the next higher value.

:p What are acids and bases in terms of ion release when dissolved in water?
??x
Acids give up positively charged hydrogen ions ($H^+$) when dissolved in water, while bases give up negatively charged hydroxyl ions ($ OH^-$). 
```java
// Example of pH calculation (pseudocode)
public class pHCalculation {
    public double calculatePH(double hydrogenIonConcentration) {
        return -Math.log10(hydrogenIonConcentration);
    }
}
```
x??

---

#### SI Units and Measurement in Physics
Background context explaining the importance of standard units like those defined by the International System of Units (SI). These units ensure consistency across scientific communities worldwide. The table provided lists basic quantities, their names, and abbreviations used by scientists.

:p List the base quantities, names, and abbreviations for SI units.
??x
The base quantities are length ($m $), mass ($ kg $), time ($ s $), electric current ($ A $), thermodynamic temperature ($ K $), amount of substance ($ mol $), and luminous intensity ($ cd$).

For example:
```java
public class SIUnits {
    public static final String LENGTH = "m";
    public static final String MASS = "kg";
    public static final String TIME = "s";
    public static final String CURRENT = "A";
    public static final String TEMPERATURE = "K";
    public static final String SUBSTANCE = "mol";
    public static final String INTENSITY = "cd";
}
```
x??

---

#### Weight vs. Mass
Background context explaining the difference between weight and mass, emphasizing that while they are related, they are distinct concepts. The formula $W = m \times g $(where $ g$ is the acceleration due to gravity) illustrates this relationship.

:p Explain why your weight on Earth differs from your weight on the moon.
??x
Your weight on Earth is influenced by the Earth's gravitational pull ($9.8 \, m/s^2 $), while your weight on the moon is much less because of its weaker gravitational field ($1.62 \, m/s^2$). Therefore, despite no change in mass, you would weigh less on the moon.

For example:
```java
public class WeightMass {
    public static final double EARTH_GRAVITY = 9.8;
    public static final double MOON_GRAVITY = 1.62;

    public void compareWeights(double mass) {
        System.out.println("Weight on Earth: " + (mass * EARTH_GRAVITY));
        System.out.println("Weight on Moon: " + (mass * MOON_GRAVITY));
    }
}
```
x??

---

#### SI Derived Units
Background context explaining that derived units are measurements obtained from base quantities through equations. The table lists common derived units such as area, volume, speed, magnetic field strength, luminance, force, energy, power, electric resistance, and more.

:p What is the SI unit for area?
??x
The SI unit for area is square meters ($m^2$).

```java
public class AreaUnit {
    public static final String AREA_UNIT = "square meters (m²)";
}
```
x??

---

#### Concept of Force in Physics
Background context explaining that force is a vector quantity and can be calculated using the equation $F = ma $ where$F $ is force,$m $ is mass, and$a $ is acceleration. The SI unit for force is Newton ($ N$).

:p What is the formula to calculate force?
??x
The formula to calculate force is $F = m \times a$, where:
- $F$ is force in Newtons (N),
- $m$ is mass in kilograms (kg), and
- $a $ is acceleration in meters per second squared ($ m/s^2$).

```java
public class ForceCalculation {
    public double calculateForce(double mass, double acceleration) {
        return mass * acceleration;
    }
}
```
x??

---

---
#### Newton's First Law of Motion
Newton’s first law of motion states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction, unless acted upon by unbalanced forces. This is also known as the law of inertia.
:p What does Newton's first law of motion state?
??x
Newton's first law of motion states that objects tend to stay in their current state (rest or uniform motion) unless a net external force acts on them. Inertia, the resistance to changes in motion, is a key concept here. The law implies that without any unbalanced forces, an object will remain at rest if it is initially at rest, and it will continue moving with constant velocity if it was already in motion.
x??
---

---
#### Unbalanced Forces and Equilibrium
According to Newton’s first law, objects are in equilibrium when the net force acting on them is zero. This means that they do not accelerate or change direction unless an unbalanced force acts upon them.
:p What does "equilibrium" mean in the context of Newton's first law?
??x
Equilibrium refers to a state where all forces acting on an object are balanced, resulting in no net acceleration or change in velocity. In other words, if an object is at rest and remains at rest, or if it is moving with constant velocity and continues to do so, the system is in equilibrium.
x??
---

---
#### Newton's Second Law of Motion
Newton’s second law of motion states that the acceleration (a) of a body is directly proportional to the net force (F) acting on it and inversely proportional to its mass (m). Mathematically, this can be expressed as $F = m \cdot a$.
:p What does Newton's second law of motion state?
??x
Newton’s second law states that the acceleration of an object is directly proportional to the net force applied to it and inversely proportional to its mass. In simpler terms, if you apply more force to an object, it will accelerate faster; if you increase the mass of an object, it will require a greater force to achieve the same acceleration.
x??
---

---
#### Application of Newton's Second Law in Everyday Scenarios
When throwing a baseball or firing a bullet, various forces come into play. For example, when you throw a ball from left field to first base, gravity causes the ball to follow an upward path and then drop. Similarly, when shooting a bullet at a target, the angle of the shot is affected by gravitational force.
:p How do external forces affect the trajectory of objects like baseballs or bullets?
??x
External forces such as gravity significantly influence the trajectories of objects. When throwing a ball, it follows an upward path due to the initial throw but eventually drops because of the downward pull of gravity. Similarly, when firing a bullet, gravity causes it to drop as well, requiring adjustments in aim based on distance and angle.
x??
---

---
#### Forces Acting on Moving Objects
When forces are not balanced (unbalanced), they cause acceleration or deceleration of an object. The more unbalanced the force is, the greater the acceleration will be according to Newton's second law.
:p How does unbalanced force affect moving objects?
??x
Unbalanced forces accelerate or decelerate objects by changing their velocity. According to Newton’s second law ($F = m \cdot a$), if an unbalanced force acts on an object, it results in acceleration proportional to the net force and inversely proportional to the mass of the object.
x??
---

---
#### The Concept of Inertia
Inertia is the tendency of objects to resist changes in their state of motion. Newton’s first law defines inertia as the property that causes objects to remain at rest or continue moving with constant velocity unless acted upon by an unbalanced force.
:p What is inertia and how does it relate to Newton's laws?
??x
Inertia is the resistance of any physical object to changes in its state of motion. According to Newton’s first law, it explains why objects tend to stay at rest or continue moving with constant velocity unless they are acted upon by an unbalanced force.
x??
---


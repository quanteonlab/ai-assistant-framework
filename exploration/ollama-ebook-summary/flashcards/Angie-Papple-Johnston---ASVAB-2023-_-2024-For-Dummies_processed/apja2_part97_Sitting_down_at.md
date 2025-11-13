# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 97)

**Starting Chapter:** Sitting down at the periodic table. Changing states Physical and chemical moves for molecules

---

---
#### SI System and Units in Chemistry
Chemists use the SI system for measurements to communicate scientifically with others worldwide. The base units are as follows:
- Length: meter (m)
- Mass: kilogram (kg)
- Volume: cubic meter (m³) and liter (L)
- Temperature: kelvin (K)
- Pressure: pascal (Pa) and newton (N)
- Energy: joule (J) and calorie

:p What are the base units of measurement used by chemists in the SI system?
??x
The base units in the SI system for chemistry include:
- Length: meter (m)
- Mass: kilogram (kg)
- Volume: cubic meter (m³) and liter (L)
- Temperature: kelvin (K)
- Pressure: pascal (Pa) and newton (N)
- Energy: joule (J) and calorie

x??
---
#### Atoms and Elements
An atom is the smallest unit of an element that still retains its properties. Each atom consists of:
- Protons: positively charged particles.
- Neutrons: uncharged particles.
- Electrons: negatively charged particles.

The atomic number of an element indicates the number of protons in its nucleus. For example, hydrogen has 1 proton and a corresponding atomic number of 1, while magnesium has 12 protons with an atomic number of 12.

:p What is an atom, and how are atoms identified within elements?
??x
An atom is the smallest unit of an element that still retains its properties. Atoms consist of three main particles:
- Protons: positively charged particles.
- Neutrons: uncharged particles.
- Electrons: negatively charged particles.

The atomic number of an element specifies the number of protons in its nucleus, uniquely identifying it. For instance, hydrogen (1 proton) and magnesium (12 protons) have distinct atomic numbers.

x??
---
#### Molecules and Compounds
Atoms can combine to form molecules. When atoms from two or more different elements bond together, a compound is formed. Compounds exhibit properties that differ significantly from those of their constituent elements. For example, sodium and chlorine are lethal when isolated but combine to form harmless table salt (NaCl).

:p What distinguishes a molecule from a compound?
??x
A molecule consists of atoms from the same element bonded together. A compound results from atoms of two or more different elements bonding, often exhibiting properties very different from their individual components.

For instance:
- Water (H₂O) is a molecule.
- Table salt (NaCl) is a compound formed by sodium and chlorine.

x??
---
#### Periodic Table
The periodic table categorizes all elements based on atomic numbers. Each element has its own unique atomic number, symbol, and atomic weight (average mass).

:p What information does the periodic table provide about elements?
??x
The periodic table provides:
- Atomic Number: The number of protons in an atom's nucleus.
- Symbol: A one or two-letter abbreviation for each element.
- Atomic Weight: The average mass of one atom, often listed as a decimal.

For example, copper (Cu) has an atomic number of 29 and an atomic weight of 63.546, while helium (He) has an atomic number of 2 and an atomic weight of 4.0026.

x??
---
#### States of Matter
Particles in matter are always moving, with the kinetic energy determining the state (solid, liquid, gas). Heat or cold changes this energy, affecting the state of matter:
- Solids: Particles close together, minimal movement, definite shape.
- Liquids: Particles more spread out, higher movement, definite volume but not shape.
- Gases: Particles highly spread out and fast-moving, no definite shape or volume.

:p What factors determine the state of matter?
??x
The state of matter is determined by:
- Kinetic Energy (motion energy) of particles.
- Heat application increases kinetic energy, changing states; cold decreases it.

Solids have closely packed particles with minimal movement and a fixed shape. Liquids have more spread-out particles moving around freely but retain a definite volume. Gases have highly dispersed particles moving very fast, exhibiting no fixed shape or volume.

x??
---

---
#### Boiling and Freezing of Water
Background context: When cooking spaghetti, you boil water to change its state from a solid to a liquid. The process of changing states involves molecular behavior as heat is added or removed.

:p What happens when you add heat to water?
??x
When heat is added to water, the molecules begin to move faster, increasing their kinetic energy. As they absorb more heat, the temperature of the water rises until it reaches its boiling point at 100 degrees Celsius (or 212 degrees Fahrenheit at sea level). At this point, the temperature remains constant as the water changes from a liquid to a gas (steam), with no further increase in temperature.

```java
public class WaterBoiling {
    public static void main(String[] args) {
        double initialTemperature = 25; // Initial temperature of water in Celsius
        double boilingPoint = 100; // Boiling point at sea level

        if (initialTemperature < boilingPoint) {
            System.out.println("Water is heating up.");
        } else {
            System.out.println("Water has reached its boiling point and will start to turn into steam.");
        }
    }
}
```
x??

---
#### Condensation
Background context: Condensation is the process by which a substance changes from a gas to a liquid. This is the opposite of evaporation, where water turns into vapor.

:p What causes condensation?
??x
Condensation occurs when an environment is colder than the air it's in. The temperature difference causes water molecules in the air to cool and eventually come together to form droplets on surfaces that are cooler than the ambient temperature. Examples include fogging glasses, dew on grass, or a cold cup sweating.

```java
public class CondensationExample {
    public static void main(String[] args) {
        boolean isAmbientCold = false; // Ambient temperature is not cold enough for condensation to occur

        if (!isAmbientCold) {
            System.out.println("No condensation will form.");
        } else {
            System.out.println("Condensation will occur as water vapor in the air cools and forms droplets on surfaces.");
        }
    }
}
```
x??

---
#### Freezing
Background context: Freezing is another state change where a substance transitions from a liquid to a solid. The freezing point of most substances is equal to its melting point, but the process happens in reverse.

:p What causes water to freeze?
??x
Water freezes when it cools down to its freezing point (0 degrees Celsius or 32 degrees Fahrenheit at sea level). As the temperature decreases, the molecules slow down and arrange themselves into a rigid structure, turning from liquid to solid.

```java
public class FreezingExample {
    public static void main(String[] args) {
        double initialTemperature = -5; // Initial temperature of water in Celsius

        if (initialTemperature < 0) {
            System.out.println("Water will freeze as it reaches its freezing point.");
        } else {
            System.out.println("Water remains a liquid at this temperature.");
        }
    }
}
```
x??

---
#### Sublimation
Background context: Sublimation is the process where a substance changes directly from a solid to a gas without passing through the liquid phase. An example of sublimation is dry ice, which is solid carbon dioxide.

:p What happens during sublimation?
??x
During sublimation, a substance transitions directly from a solid state to a gaseous state. For instance, dry ice (solid CO2) turns into a colorless gas without first becoming liquid. This process can create visible effects like clouds due to the rapid cooling and condensation of water vapor in the air.

```java
public class SublimationExample {
    public static void main(String[] args) {
        boolean isDryIce = true; // Representing dry ice as a solid CO2

        if (isDryIce) {
            System.out.println("Dry ice undergoes sublimation, turning directly from solid to gas.");
        } else {
            System.out.println("No sublimation occurs in this scenario.");
        }
    }
}
```
x??

---

#### Compounds: Elements Joining Together
Compounds are formed when elements join by chemical bonds in specific ratios. For example, water (H₂O) is a compound of hydrogen and oxygen atoms combined in a 2:1 ratio.
:p What is a compound?
??x
A compound is a substance formed when two or more different elements chemically bond together in fixed proportions.
x??

---
#### Water as a Compound
Water (H₂O) consists of hydrogen and oxygen atoms bonded in a specific way. The subscript '2' after H indicates there are two hydrogen atoms, while the absence of a number next to O means one oxygen atom is present.
:p How do you represent water chemically?
??x
Water is represented as H₂O, where the subscript '2' signifies that there are two hydrogen atoms and the absence of a number next to oxygen indicates only one oxygen atom in this compound.
x??

---
#### Sodium Chloride (Table Salt)
Sodium chloride (NaCl) is an ionic compound composed of sodium and chlorine. The formula NaCl represents one sodium atom for each chlorine atom, forming discrete molecules.
:p What is the chemical formula for table salt?
??x
The chemical formula for table salt is NaCl, representing one sodium (Na) ion paired with one chloride (Cl) ion.
x??

---
#### Baking Soda
Baking soda has the chemical formula NaHCO₃. It consists of sodium (Na), hydrogen (H), carbon (C), and oxygen (O).
:p What are the elements that make up baking soda?
??x
Baking soda is composed of sodium (Na), hydrogen (H), carbon (C), and oxygen (O). The formula NaHCO₃ indicates one sodium atom, one hydrogen atom, one carbon atom, and three oxygen atoms.
x??

---
#### Nitroglycerin
Nitroglycerin has the chemical formula C₃H₅N₃O₉. It contains carbon, hydrogen, nitrogen, and oxygen in a 3:5:3:9 ratio respectively.
:p What is the chemical composition of nitroglycerin?
??x
Nitroglycerin consists of three carbon (C), five hydrogen (H), three nitrogen (N), and nine oxygen (O) atoms. Its formula C₃H₅N₃O₉ represents this specific arrangement of elements.
x??

---
#### Silver Nitrate
Silver nitrate (AgNO₃) is a compound containing silver, nitrogen, and oxygen in the ratio 1:1:3.
:p What are the components of silver nitrate?
??x
Silver nitrate consists of one silver (Ag), one nitrogen (N), and three oxygen (O) atoms. Its formula AgNO₃ indicates this composition.
x??

---
#### Nicotine
Nicotine has a complex chemical structure with multiple carbon, hydrogen, and nitrogen atoms in specific ratios. It is represented by the formula C₁₀H₁４N₂.
:p What is the chemical formula for nicotine?
??x
Nicotine's chemical formula is C₁₀H₁４N₂, indicating ten carbon (C), fourteen hydrogen (H), and two nitrogen (N) atoms.
x??

---


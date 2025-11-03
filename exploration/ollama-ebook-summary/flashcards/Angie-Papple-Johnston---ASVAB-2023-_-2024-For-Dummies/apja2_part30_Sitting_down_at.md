# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 30)

**Starting Chapter:** Sitting down at the periodic table. Changing states Physical and chemical moves for molecules

---

#### Base Units in SI System
Background context: Chemists use the SI system to measure physical quantities for global communication. The base units for various measurements are defined as follows:
- Length: meter (m)
- Mass: kilogram (kg)
- Volume: cubic meter (m³) and liter (L)
- Temperature: kelvin (K)
- Pressure: pascal (Pa) and newton (N)
- Energy: joule (J) and calorie

:p What are the base units for measuring length, mass, volume, temperature, pressure, and energy in the SI system?
??x
The meter (m) is the base unit of length. The kilogram (kg) serves as the base unit of mass. For volume, both cubic meters (m³) and liters (L) are used. Temperature is measured using kelvins (K). Pressure can be measured in pascals (Pa) or newtons (N), while energy is quantified in joules (J) or calories.

```java
// Example of using the SI units in a simple Java program
public class UnitConversion {
    public static void main(String[] args) {
        double length = 10.5; // Length in meters
        double mass = 2.3;    // Mass in kilograms
        double volumeLiters = 5.0; // Volume in liters
        double tempKelvin = 298; // Temperature in kelvins
        double pressurePa = 101325; // Pressure in pascals
        double energyJoules = 4186; // Energy in joules (approx. 1 calorie)
        
        System.out.println("Length: " + length + " meters");
        System.out.println("Mass: " + mass + " kilograms");
        System.out.println("Volume: " + volumeLiters + " liters");
        System.out.println("Temperature: " + tempKelvin + " kelvins");
        System.out.println("Pressure: " + pressurePa + " pascals");
        System.out.println("Energy: " + energyJoules + " joules (equivalent to 1 calorie)");
    }
}
```
x??

---

#### Atoms and Atomic Structure
Background context: An atom is the smallest unit of an element that retains its elemental characteristics. Each atom consists of particles, including electrons, neutrons, and protons.
- Protons are positively charged and located in the nucleus.
- Neutrons are neutral and also found in the nucleus.
- Electrons carry a negative charge and orbit around the nucleus.

The atomic number is determined by the number of protons. For example:
- Hydrogen (H) has an atomic number of 1, meaning it has one proton.
- Magnesium (Mg) has an atomic number of 12, indicating it contains 12 protons.

:p What are the basic components of an atom and how do they relate to each other?
??x
An atom consists of three main particles:
- Protons: Positively charged particles found in the nucleus.
- Neutrons: Neutral particles also located in the nucleus.
- Electrons: Negatively charged particles that orbit around the nucleus.

The atomic number (Z) is equal to the number of protons, which defines an element. For example, hydrogen has 1 proton and thus an atomic number of 1; magnesium has 12 protons and an atomic number of 12.

```java
// Example of representing an atom in a simple Java class
public class Atom {
    private int atomicNumber;
    
    public Atom(int atomicNumber) {
        this.atomicNumber = atomicNumber;
    }
    
    public void displayInfo() {
        System.out.println("Atomic Number: " + atomicNumber);
        // Additional methods to represent other particles could be added here.
    }
}

public class Main {
    public static void main(String[] args) {
        Atom hydrogen = new Atom(1);  // Hydrogen atom
        Atom magnesium = new Atom(12);  // Magnesium atom
        
        hydrogen.displayInfo();
        magnesium.displayInfo();
    }
}
```
x??

---

#### The Periodic Table
Background context: The periodic table is a tabular arrangement of all known elements. Elements are listed by their atomic numbers and grouped into families based on similar properties.
- Atomic number (Z): Number of protons in the nucleus.
- Abbreviation: Symbol used to represent an element.
- Atomic weight: Average mass of one atom of the element.

:p What is the periodic table, and what information does it provide about elements?
??x
The periodic table classifies all known elements. It lists each element by its atomic number (number of protons) and provides information such as:
- Abbreviation or symbol for the element.
- Atomic weight: The average mass of one atom.

For example, copper (Cu) has an atomic number 29 and an atomic weight of 63.546, while helium (He) with an atomic number 2 has an atomic weight of 4.0026.

```java
// Example of representing the periodic table in a simple Java class
public class Element {
    private String symbol;
    private int atomicNumber;
    private double atomicWeight;
    
    public Element(String symbol, int atomicNumber, double atomicWeight) {
        this.symbol = symbol;
        this.atomicNumber = atomicNumber;
        this.atomicWeight = atomicWeight;
    }
    
    public void displayInfo() {
        System.out.println("Symbol: " + symbol);
        System.out.println("Atomic Number: " + atomicNumber);
        System.out.println("Atomic Weight: " + atomicWeight);
    }
}

public class Main {
    public static void main(String[] args) {
        Element copper = new Element("Cu", 29, 63.546); // Copper
        Element helium = new Element("He", 2, 4.0026); // Helium
        
        copper.displayInfo();
        helium.displayInfo();
    }
}
```
x??

---

#### States of Matter: Physical and Chemical Changes
Background context: The state of matter is determined by the kinetic energy (motion energy) of particles in a substance. Heat or cold can cause physical changes, but not chemical ones.
- Solids have closely packed particles that move little.
- Liquids have more spread out particles that move faster.
- Gases have highly spread-out particles moving very quickly.

:p How do temperature and kinetic energy affect the state of matter?
??x
Temperature and kinetic energy determine the state of matter. When heat is applied:
- Gas particles move very quickly and are widely separated.
- Liquid particles move more slowly than gas particles but still have some freedom to move around each other.
- Solid particles move much slower, maintaining a rigid structure due to strong intermolecular forces.

Cold can reverse these effects by slowing down particle movement. The state changes (melting, freezing, vaporization) are physical changes, meaning the molecular composition remains unchanged.

```java
// Example of changing states in a simple Java program
public class StateChange {
    public static void main(String[] args) {
        double temperature = 100; // Temperature in Celsius
        
        if (temperature > 100) {
            System.out.println("Water is in the gas state: Steam");
        } else if (temperature < 0) {
            System.out.println("Water is in the solid state: Ice");
        } else {
            System.out.println("Water is in the liquid state: Liquid Water");
        }
    }
}
```
x??

---

---
#### Boiling Point and State Changes of Water
When you're making spaghetti, boiling water is used to change the pasta from a solid state into a more softened form. The temperature of water increases as it absorbs heat until it reaches its boiling point at 100 degrees Celsius (212 degrees Fahrenheit) under normal atmospheric pressure.

The transition of water from liquid to gas involves the process of evaporation, where molecules gain enough energy to break free and enter the gaseous state. This temperature remains constant during the phase change because all the heat energy is used for changing states rather than raising the temperature further.

:p What is the boiling point of water at sea level?
??x
The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.
x??

---
#### Condensation and Phase Change
Condensation is a phase change where a gas transforms into a liquid. This happens when cooler surfaces attract water vapor in the air, causing it to condense into droplets. Examples of this include fogging glasses, dew on grass, or a cold drink sweating.

:p What is an example of condensation?
??x
An example of condensation is seeing dew on grass in the morning.
x??

---
#### Freezing and Deposition
Freezing occurs when heat is removed from a substance, slowing down molecular motion until it solidifies. The freezing point for water is -0°C or 32°F at sea level.

Deposition is the reverse of sublimation; it’s when a gas turns directly into a solid without going through a liquid phase. For instance, dry ice (solid CO2) sublimes to form carbon dioxide gas but can create a visible white cloud due to condensation of water vapor in the air as it cools.

:p What is deposition?
??x
Deposition is the process where a substance changes directly from a gaseous state to a solid state without passing through the liquid phase.
x??

---
#### Sublimation and Dry Ice
Sublimation happens when a substance transitions directly from a solid to a gas, bypassing the liquid stage. An example of this is dry ice (solid CO2) turning into carbon dioxide gas.

The process of sublimation can create dramatic effects like smoke or fog in magic shows or nightclubs due to the rapid cooling and evaporation producing visible clouds.

:p What substance is commonly used for creating smoke or fog effects through sublimation?
??x
Dry ice, which is solid carbon dioxide, is commonly used for creating smoke or fog effects.
x??

---
#### States of Matter Overview
The text describes how different states of matter (solid, liquid, gas) can change due to temperature and pressure changes. Molecules in each state move differently; solids have the least movement, liquids more, and gases the most.

:p How do molecules behave differently in solid, liquid, and gaseous states?
??x
In a solid state, molecules vibrate but stay in fixed positions relative to each other. In a liquid state, molecules can flow past one another with some freedom of motion. In a gas state, molecules move very rapidly and spread out as far apart as they can.
x??

---

---
#### Compounds: Elements Joining by Chemical Bonds
Compounds are formed when elements combine through chemical bonds at specific ratios. For instance, water (H2O) is a compound where hydrogen and oxygen combine in a ratio of 2 atoms of hydrogen to every 1 atom of oxygen.

:p How do elements form compounds?
??x
Elements form compounds by combining with each other in specific ratios through chemical bonds. This means that for water (H2O), two hydrogen atoms bond with one oxygen atom, adhering to the strict stoichiometric ratio.
x??

---
#### Reading Chemical Formulas: Basic Structure
Chemical formulas are written as shorthand expressions of molecules and compounds. A formula includes the elemental symbols and subscripts indicating the number of atoms in a molecule. For example, H2O indicates two hydrogen atoms bonding with one oxygen atom.

:p What do subscripts in chemical formulas represent?
??x
Subscripts in chemical formulas indicate the number of atoms of each element present in a single molecule. In H2O, the subscript 2 means there are two hydrogen atoms, and the absence of a subscript after O indicates there is only one oxygen atom.
x??

---
#### Reading Chemical Formulas: Number of Molecules
Chemical formulas also include numerical coefficients that indicate how many molecules or units of a compound you have. For example, "5He" denotes five individual helium atoms.

:p How do you represent multiple molecules in chemical formulas?
??x
To represent multiple molecules in a formula, place the number before the molecule's elemental symbol. So, "5He" signifies five helium atoms.
x??

---
#### Reading Complex Formulas with Brackets
Some complex formulas use brackets to group elements together before adding subscripts. For instance, AgNO3 includes brackets around NO3, indicating that this is a single unit.

:p What do brackets in chemical formulas indicate?
??x
Brackets in chemical formulas represent groups of atoms that are considered as one unit. In AgNO3, the bracketed NO3 group means that the nitrogen and three oxygen atoms form a single entity.
x??

---
#### Example Compounds: Sodium Bicarbonate
Sodium bicarbonate (NaHCO3) is an example where sodium (Na), hydrogen (H), carbon (C), and oxygen (O) are combined in specific ratios to form this compound.

:p What elements make up sodium bicarbonate?
??x
Sodium bicarbonate is composed of sodium (Na), hydrogen (H), carbon (C), and oxygen (O). The formula NaHCO3 shows the combination of these elements in a specific ratio.
x??

---
#### Example Compounds: Table Salt
Table salt (NaCl) consists of sodium (Na) and chlorine (Cl) combined in a 1-to-1 ratio, forming this compound.

:p What are the components of table salt?
??x
Table salt is composed of sodium (Na) and chlorine (Cl), which combine in equal proportions to form NaCl.
x??

---
#### Example Compounds: Nitroglycerin
Nitroglycerin (C3H5N3O9) involves a more complex combination where carbon, hydrogen, nitrogen, and oxygen atoms are arranged in specific ratios.

:p What elements make up nitroglycerin?
??x
Nitroglycerin is made up of carbon (C), hydrogen (H), nitrogen (N), and oxygen (O). The formula C3H5N3O9 indicates the arrangement of these elements.
x??

---
#### Example Compounds: Silver Nitrate
Silver nitrate (AgNO3) consists of silver (Ag), nitrogen (N), and oxygen (O) in a specific ratio.

:p What are the components of silver nitrate?
??x
Silver nitrate is composed of silver (Ag), nitrogen (N), and oxygen (O). The formula AgNO3 shows how these elements combine.
x??

---


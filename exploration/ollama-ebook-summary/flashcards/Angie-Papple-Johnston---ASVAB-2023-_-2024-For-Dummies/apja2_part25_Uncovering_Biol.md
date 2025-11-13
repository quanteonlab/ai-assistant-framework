# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 25)

**Starting Chapter:** Uncovering Biology from Big to Small

---

#### Scientific Method
Background context explaining the steps of the scientific method and its importance in science. The process helps ensure that observations and experiments are objective, systematic, and repeatable.

:p What is the first step of the scientific method?
??x
The first step of the scientific method is to observe some aspect of the universe. This involves making careful observations and noting patterns or phenomena.
x??

---
#### Scientific Laws vs Theories
Background context explaining the differences between laws and theories in science, emphasizing that they are distinct concepts with specific meanings.

:p How do scientists view a law differently from a theory?
??x
Scientists view a law as a statement about something that happens, whereas a theory is a detailed explanation of why it happens. Laws describe phenomena (e.g., gravity), while theories provide explanations for these phenomena.
x??

---
#### Metric System
Background context explaining the importance of measurements in science and the use of the metric system for precise measurement.

:p What are some common units of measurement in the metric system?
??x
Common units of measurement in the metric system include:
- Meter (m) for length
- Liter (L) for volume
- Gram (g) for mass (similar to weight)
x??

---
#### Scientific Laws and Theories You Already Know
Background context explaining common scientific laws and theories that people experience daily, such as gravity and evolution.

:p Name a scientific law mentioned in the text.
??x
A scientific law mentioned in the text is Newton's third law of motion: For every action, there is an equal and opposite reaction. This means when you hold this book, your hands exert force on it, and the book exerts an equal and opposite force back on your hands.
x??

---
#### Metric Prefixes
Background context explaining the metric system prefixes and their meanings.

:p What does the prefix 'milli-' mean in the metric system?
??x
The prefix 'milli-' in the metric system means one-thousandth (0.001). For example, a millimeter is one-thousandth of a meter.
x??

---
#### Common Metric Measurement Abbreviations
Background context explaining common abbreviations for metric measurements.

:p List two common units and their abbreviations from Table 10-2.
??x
Two common units and their abbreviations from Table 10-2 are:
- Meter (m)
- Liter (L)
x??

---
#### Common Metric Conversion Formulas
Background context explaining the most common conversion formulas in the metric system.

:p What is the formula for converting meters to centimeters?
??x
The formula for converting meters to centimeters is: 1 meter = 100 centimeters.
x??

---
#### Scientific Laws and Theories Example (Continued)
Background context reiterating the example of scientific laws and theories in everyday life.

:p What theory explains why you aren't identical to your parents?
??x
The theory that explains why you aren't identical to your parents is the theory of evolution. This theory states that living things on Earth have origins in other, preexisting types, and the differences from ancestors are due to modifications in DNA over generations.
x??

---
#### Scientific Method Example (Continued)
Background context reiterating an example of applying the scientific method.

:p How does Newton's third law apply to holding a book?
??x
Newton's third law applies when you hold a book. The book exerts force on your hands, and in response, your hands exert an equal and opposite force back on the book.
x??

---

#### Ecology Overview
Ecology is the study of the environment, specifically focusing on the relationships between organisms and their surroundings. Ecosystems consist of producers that make their own food (like plants through photosynthesis), consumers that eat other things, and decomposers like bacteria that break down dead matter.
:p What are the main components of an ecosystem?
??x
The main components of an ecosystem include producers, consumers, and decomposers. Producers create their own food using sunlight or chemical energy, while consumers depend on other organisms for sustenance. Decomposers break down dead plants, animals, and waste materials.
x??

---

#### Common Metric Units and Their Abbreviations
This table outlines common metric units used in scientific measurements:
- Length: millimeter (mm), centimeter (cm), meter (m), kilometer (km)
- Liquid Volume: milliliter (mL), centiliter (cL), liter (L), kiloliter (kL)
- Mass: milligram (mg), centigram (cg), gram (g), kilogram (kg)

:p What are the metric units for measuring length, liquid volume, and mass?
??x
For length, the metric units include millimeters (mm), centimeters (cm), meters (m), and kilometers (km). For liquid volume, the common units are milliliters (mL), centiliters (cL), liters (L), and kiloliters (kL). Mass is measured using milligrams (mg), centigrams (cg), grams (g), and kilograms (kg).
x??

---

#### Imperial to Metric Conversions
This table shows conversions between imperial units and their metric equivalents:
- Inches: 1 inch = 2.54 cm
- Feet: 1 foot = 0.3 m
- Yards: 1 yard = 0.9 m
- Miles: 1 mile = 1.6 km
- Square Inches: 1 square inch = 6.45 square centimeters
- Square Feet: 1 square foot = 0.09 square meters
- Quarts: 1 quart = 0.94 liters
- Gallons: 1 gallon = 3.78 liters
- Ounces: 1 ounce = 28.3 grams
- Pounds: 1 pound = 0.45 kilograms

:p How do you convert inches to centimeters?
??x
To convert inches to centimeters, use the formula:
$$\text{centimeters} = \text{inches} \times 2.54$$

For example, if you have 5 inches, it would be converted to:
$$5 \text{ inches} \times 2.54 = 12.7 \text{ cm}$$

```java
public class Conversion {
    public static double inchesToCentimeters(double inches) {
        return inches * 2.54;
    }
}
```
x??

---

#### Categories of Consumers
Background context: Animals cannot produce their own food, making them consumers. These consumers can be classified into three categories based on their diet: carnivores, herbivores, and omnivores.

:p What are the three main categories of animal consumers?
??x
The three main categories of animal consumers are:
- Carnivores: eat only meat.
- Herbivores: eat only plants.
- Omnivores: eat both plants and other animals.

For example:
```java
public class ConsumerType {
    public static void classify(String diet) {
        if (diet.equals("meat")) {
            System.out.println("Carnivore");
        } else if (diet.equals("plants")) {
            System.out.println("Herbivore");
        } else if (diet.equals("both")) {
            System.out.println("Omnivore");
        } else {
            System.out.println("Unknown diet");
        }
    }

    public static void main(String[] args) {
        classify("meat"); // Prints "Carnivore"
        classify("plants"); // Prints "Herbivore"
        classify("both"); // Prints "Omnivore"
    }
}
```
x??

---

#### Ecosystem Conditions for Plants
Background context: For plants to grow, several conditions are necessary: adequate sunlight, good soil, moderate temperatures, and water. These conditions support the growth of producers, which in turn sustain other consumers.

:p What conditions must be present for plants to grow?
??x
The conditions that must be present for plants to grow include:
- Adequate sunlight.
- Good soil.
- Moderate temperatures.
- Water.

For example:
```java
public class PlantGrowthConditions {
    public static boolean checkConditions(String sunlight, String soilQuality, int temperature, int waterAmount) {
        return (sunlight.equals("adequate") && 
                soilQuality.equals("good") &&
                temperature >= 50 && temperature <= 100 && // Moderate temperatures
                waterAmount > 0);
    }

    public static void main(String[] args) {
        System.out.println(checkConditions("adequate", "good", 75, 500)); // Returns true
    }
}
```
x??

---

#### Biodiversity and Species Importance
Background context: Biodiversity refers to the variety of life in the world or specific habitats. Every species plays an important role in natural sustainability. Loss of biodiversity can disrupt entire food chains.

:p Why is biodiversity crucial for the continuation of life on Earth?
??x
Biodiversity is crucial for the continuation of life on Earth because:
- Each species has a unique role in maintaining natural balance.
- If one species that serves as food for another dies out, it creates a domino effect disrupting the entire food chain (up to humans).
- Biodiversity ensures that species can adapt to changing conditions over time due to their genetic diversity.

For example:
```java
public class Biodiversity {
    public static boolean checkBiodiversity(String[] species) {
        // Assume each unique species contributes positively to biodiversity
        return new HashSet<>(Arrays.asList(species)).size() > 10;
    }

    public static void main(String[] args) {
        String[] species = {"tiger", "elephant", "giraffe", "zebra", "lion", "snake"};
        System.out.println(checkBiodiversity(species)); // Returns true if at least 10 unique species are present
    }
}
```
x??

---

#### Scientific Classification of Organisms
Background context: To effectively study and discuss plants, animals, and other living creatures, scientists developed a system of scientific classification. The most common system was created by Carl Linnaeus in the mid-18th century.

:p How did Carl Linnaeus contribute to the classification of organisms?
??x
Carl Linnaeus contributed to the classification of organisms by developing the widely used binomial nomenclature system and publishing ten editions of his works from 1753 to 1758. His system helped scientists use consistent names for plants, animals, and other living creatures.

For example:
```java
public class LinnaeusClassification {
    public static void classify(String name) {
        System.out.println(name + " is classified under the binomial nomenclature system by Carl Linnaeus.");
    }

    public static void main(String[] args) {
        classify("Canis lupus familiaris"); // Prints "Canis lupus familiaris is classified under the binomial nomenclature system by Carl Linnaeus."
    }
}
```
x??

---

#### Domain
Domains are the broadest taxonomic groupings of organisms, classified based on characteristics such as cell structure and chemistry. There are three domains: Bacteria, Archaea, and Eukarya.

:p What are the three main domains of living organisms?
??x
The three main domains of living organisms are Bacteria, Archaea, and Eukarya.
x??

---

#### Kingdom
Kingdoms group organisms based on developmental characteristics and whether they can make their own food. There is debate about the exact number, but there are generally five or six kingdoms: Animalia, Plantae, Fungi, Protista, Monera, and sometimes Chromista.

:p How many kingdoms are generally recognized by scientists?
??x
Scientists generally recognize five or six kingdoms: Animalia, Plantae, Fungi, Protista, Monera, and occasionally Chromista.
x??

---

#### Phylum
Phyla group organisms within a kingdom based on general characteristics. There are 36 phyla in the Animal Kingdom alone.

:p How many phyla exist in the Animal Kingdom?
??x
There are 36 phyla in the Animal Kingdom.
x??

---

#### Class
Classes further divide organisms of the same phylum by similar characteristics. For example, humans and other primates belong to the class Mammalia, while birds belong to Aves.

:p What is an example of a class within the Animal Kingdom?
??x
An example of a class within the Animal Kingdom is Mammalia, which includes humans.
x??

---

#### Order
Orders separate organisms based on the characteristics of major groups within their class. For instance, primates (humans, apes, and monkeys) are part of the order Primata.

:p What is an example of an order in the Animal Kingdom?
??x
An example of an order in the Animal Kingdom is Primata, which includes humans, apes, and monkeys.
x??

---

#### Family
Families further divide organisms within the same order by similar characteristics. Humans are part of the family Hominidae.

:p What is an example of a family within the Primate order?
??x
An example of a family within the Primate order is Hominidae, which includes humans.
x??

---

#### Genus
Genera group two or more closely related species with unique body structures into a single genus. Humans and gorillas belong to different genera: Homo for humans and Gorilla for gorillas.

:p What is an example of a genus in the Animal Kingdom?
??x
An example of a genus in the Animal Kingdom is Homo, which includes modern humans.
x??

---

#### Species
Species are the most specific taxonomic level. Organisms within the same species share very similar characteristics. For example, all chimpanzees and bonobos belong to the same species Pan troglodytes.

:p What is an example of a species in the Animal Kingdom?
??x
An example of a species in the Animal Kingdom is Pan troglodytes, which includes both chimpanzees and bonobos.
x??

---

#### Linnaean Taxonomic Classification System
The Linnaean system categorizes living organisms into a hierarchical structure starting from broadest to most specific. The mnemonic "Dear King Phillip, come over for good spaghetti" helps remember the order: Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species.
:p What is the purpose of the Linnaean taxonomic classification system?
??x
The primary purpose of the Linnaean system is to organize and categorize living organisms in a systematic manner. This helps scientists understand relationships between different species and facilitates research and communication within the scientific community.

For example:
```java
public class Taxonomy {
    private String domain;
    private String kingdom;
    private String phylum;
    private String classs;
    private String order;
    private String family;
    private String genus;
    private String species;

    // Constructor and methods to set and get taxonomy levels
}
```
x??

---

#### Kingdoms in Taxonomy
There are generally five or six kingdoms in the taxonomic system: Animals, Plants, Fungi, Protists, Eubacteria, and Archaebacteria.
:p List the major kingdoms of life according to the Linnaean system.
??x
The major kingdoms of life according to the Linnaean system are:
- **Animals**: Multicellular organisms that do not have cell walls or chlorophyll. They can move.
- **Plants**: Multicellular organisms with cell walls made of cellulose but no nervous systems. They perform photosynthesis.
- **Fungi**: Organisms like mushrooms and yeast, which do not perform photosynthesis but have chitin in their cell walls.
- **Protists**: One-celled organisms that can move (e.g., protozoans).
- **Eubacteria**: Single-celled organisms without distinct nuclei or organelles found everywhere including inside the human body.
- **Archaebacteria**: Single-celled organisms with different genetic structures and metabolic processes from bacteria.

For example:
```java
public class Kingdom {
    private String name;
    public Kingdom(String name) { this.name = name; }
    
    @Override
    public String toString() {
        return "Kingdom: " + name;
    }
}
```
x??

---

#### Human Species Classification
The human species is classified as Homo sapiens. This means that humans belong to the genus Homo and the species sapiens.
:p What is the scientific classification of humans?
??x
Humans are scientifically classified as:
- **Genus**: Homo
- **Species**: Sapiens

For example, in a simple class:
```java
public class Human {
    private String genus;
    private String species;

    public Human(String genus, String species) {
        this.genus = genus;
        this.species = species;
    }

    @Override
    public String toString() {
        return "Genus: " + genus + ", Species: " + species;
    }
}
```
x??

---

#### Classification of Strawberry and House Cat
- **Strawberry**: Domain Eukarya, Kingdom Plantae, Phylum Spermatophyta, Class Dicotyledonae.
- **House Cat**: Domain Eukarya, Kingdom Animalia, Phylum Chordata, Class Mammalia.
:p What is the scientific classification of a strawberry and a house cat?
??x
For a strawberry:
- **Domain**: Eukarya (living organisms with complex cells)
- **Kingdom**: Plantae (plants)
- **Phylum**: Spermatophyta (seed plants)
- **Class**: Dicotyledonae (flowering plants with two seed leaves)

For a house cat:
- **Domain**: Eukarya
- **Kingdom**: Animalia (multicellular organisms that can move and do not perform photosynthesis)
- **Phylum**: Chordata (animals with a notochord, spinal cord, and other chordate traits)
- **Class**: Mammalia (mammals)

For example:
```java
public class Strawberry {
    private String genus;
    private String species;

    public Strawberry(String genus, String species) {
        this.genus = genus;
        this.species = species;
    }

    @Override
    public String toString() {
        return "Strawberry: Genus " + genus + ", Species " + species;
    }
}
public class HouseCat {
    private String genus;
    private String species;

    public HouseCat(String genus, String species) {
        this.genus = genus;
        this.species = species;
    }

    @Override
    public String toString() {
        return "House Cat: Genus " + genus + ", Species " + species;
    }
}
```
x??

---

#### Staphylococcus aureus Classification
Staphylococcus aureus is a bacterium classified as:
- **Domain**: Bacteria (single-celled organisms without distinct nuclei)
- **Kingdom**: Eubacteria (true bacteria)
- **Phylum**: Firmicutes
- **Class**: Bacilli
- **Order**: Staphylococcaceae
- **Family**: Staphylococcaceae
- **Genus**: Staphylococcus
- **Species**: aureus
:p What is the scientific classification of Staphylococcus aureus?
??x
Staphylococcus aureus is classified as:
- **Domain**: Bacteria
- **Kingdom**: Eubacteria
- **Phylum**: Firmicutes
- **Class**: Bacilli
- **Order**: Staphylococcaceae
- **Family**: Staphylococcaceae
- **Genus**: Staphylococcus
- **Species**: aureus

For example:
```java
public class Bacterium {
    private String genus;
    private String species;

    public Bacterium(String genus, String species) {
        this.genus = genus;
        this.species = species;
    }

    @Override
    public String toString() {
        return "Bacterium: Genus " + genus + ", Species " + species;
    }
}
```
x??

---


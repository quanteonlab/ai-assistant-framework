# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 129)

**Starting Chapter:** Subtest 3 Word Knowledge Answers

---

#### Division of Fractions
Background context explaining how to divide fractions, including the concept that dividing by a fraction is equivalent to multiplying by its reciprocal. This involves converting mixed numbers to improper fractions for easier manipulation.

:p How do you divide 10½ by ¾?
??x
To divide 10½ by ¾, first convert the mixed number 10½ into an improper fraction:

$$10\frac{1}{2} = \frac{(10 \times 2) + 1}{2} = \frac{21}{2}$$

Next, multiply by the reciprocal of $\frac{3}{4}$:

$$\frac{21}{2} \div \frac{3}{4} = \frac{21}{2} \times \frac{4}{3}$$

Now, perform the multiplication:
$$\frac{21 \times 4}{2 \times 3} = \frac{84}{6} = 14$$

So,$10\frac{1}{2} \div \frac{3}{4} = 14$.

This process can be represented in code as follows:

```java
public class FractionDivision {
    public static double divideFractions(double numerator1, double denominator1, double numerator2, double denominator2) {
        return (numerator1 / denominator1) * (denominator2 / numerator2);
    }
    
    public static void main(String[] args) {
        // Example: 10½ ÷ ¾
        System.out.println(divideFractions(21.0 / 2, 3.0 / 4)); // Output should be 14.0
    }
}
```

x??

---

#### Stones Needed for Patio Calculation

:p How many paving stones are needed to cover a patio that measures 12 feet by 14 feet if each stone is 8 inches square?

??x
First, convert the dimensions of the patio from feet to inches:
- For 12 feet: $12 \times 12 = 144$ inches.
- For 14 feet: $14 \times 12 = 168$ inches.

Next, calculate how many stones fit along each dimension:
- Along the 12-foot side (144 inches): $144 / 8 = 18$ stones.
- Along the 14-foot side (168 inches): $168 / 8 = 21$ stones.

Finally, multiply these two numbers to get the total number of stones required:
$$18 \times 21 = 378$$

So, you need a total of 378 paving stones.

This can be computed in Java as follows:

```java
public class PatioStones {
    public static int calculatePavingStones(int lengthFeet, int widthFeet, int stoneLengthInches) {
        // Convert feet to inches
        int lengthInches = lengthFeet * 12;
        int widthInches = widthFeet * 12;
        
        // Calculate the number of stones per dimension and multiply
        return (lengthInches / stoneLengthInches) * (widthInches / stoneLengthInches);
    }
    
    public static void main(String[] args) {
        System.out.println(calculatePavingStones(12, 14, 8)); // Output should be 378
    }
}
```

x??

---

#### Salary and Deduction Calculation

:p How much is the net pay of a programmer whose gross salary is$25,000 if 28% of it is deducted for taxes?

??x
First, calculate the amount of the deduction:
$$ \text{Deduction} = \$25,000 \times 0.28 = \$7,000 $$Then subtract this from the gross salary to find the net pay:
$$ \text{Net Pay} = \$25,000 - \$7,000 = \$18,000 $$So, the net pay is$18,000.

This calculation can be done in Java as follows:

```java
public class SalaryDeduction {
    public static double calculateNetPay(double grossSalary, double deductionPercentage) {
        return grossSalary - (grossSalary * deductionPercentage);
    }
    
    public static void main(String[] args) {
        System.out.println(calculateNetPay(25000.0, 0.28)); // Output should be 18000.0
    }
}
```

x??

---

#### Supplementary Angles

:p What is the supplement of an angle measuring 55 degrees?

??x
The supplement of an angle is found by subtracting the given angle from 180 degrees:
$$\text{Supplement} = 180^\circ - 55^\circ = 125^\circ$$

So, the supplement of a 55-degree angle is 125 degrees.

x??

---

#### Lumber Stack Calculation

:p How many pieces of lumber are in a stack that measures 6 feet high if each piece is 4 inches thick?

??x
First, convert the height from feet to inches:
$$6 \times 12 = 72 \text{ inches}$$

Then divide by the thickness of each piece of lumber (4 inches) to find the number of pieces:
$$\frac{72}{4} = 18$$

So, there are 18 pieces of lumber in the stack.

This can be calculated using Java as follows:

```java
public class LumberStack {
    public static int calculatePiecesOfLumber(double heightFeet, double thicknessInches) {
        // Convert feet to inches and divide by thickness
        return (int) ((heightFeet * 12) / thicknessInches);
    }
    
    public static void main(String[] args) {
        System.out.println(calculatePiecesOfLumber(6.0, 4)); // Output should be 18
    }
}
```

x??

---

#### Word Knowledge Concepts

:p What does the word "abeyance" mean?

??x
Abeyance is a state of temporary disuse or suspension.

x??

---

#### Definition of Null Adjective
Null is an adjective that means having or associated with the value zero; it also refers to something that has no legal or binding force or to something that is invalid. This term often appears in contexts where a lack of value or significance is relevant.

:p What does null mean as an adjective?
??x
Null as an adjective means having or associated with the value zero, and can refer to something that lacks legal or binding force, or is invalid.
x??

---

#### Definition of Indigent Adjective and Noun
Indigent is an adjective that means poor or needy. It’s also a noun that refers to a needy person. This term is often used in social contexts where someone's financial status is critical.

:p What does indigent mean as both an adjective and a noun?
??x
As an adjective, indigent means poor or needy; as a noun, it refers to a needy person.
x??

---

#### Definition of Impertinent Adjective
Impertinent is an adjective that means rude or not showing proper respect. This term is often used in social contexts where behavior that lacks courtesy or respect is highlighted.

:p What does impertinent mean?
??x
Impertinent means rude or lacking proper respect.
x??

---

#### Definition of Lustrous Adjective
Lustrous is an adjective that means shining or glossy. This term is commonly associated with objects that have a reflective surface, such as metals or gemstones.

:p What does lustrous mean?
??x
Lustrous means shining or glossy, typically describing objects with a reflective appearance.
x??

---

#### Definition of Pardon Verb and Noun
Pardon is a verb that means to forgive or excuse someone or something. It’s also a noun that refers to the action of forgiving (or being forgiven) for something. This term often appears in contexts involving forgiveness, legal proceedings, or apologies.

:p What does pardon mean as both a verb and a noun?
??x
As a verb, pardon means to forgive or excuse someone or something; as a noun, it refers to the act of forgiving or being forgiven.
x??

---

#### Definition of Veracious Adjective
Veracious is an adjective that means speaking or representing the truth. This term is often used in contexts where honesty and authenticity are emphasized.

:p What does veracious mean?
??x
Veracious means speaking or representing the truth, emphasizing honesty and accuracy.
x??

---

#### Paragraph Comprehension Subtest for AFQT

The Paragraph Comprehension subtest of the ASVAB contributes to your overall AFQT score. To improve performance on this subtest, focus on analytical reading skills by identifying main points, remembering key details, or summarizing passages.

:p What is the purpose of the paragraph comprehension subtest in the context of the AFQT?
??x
The Paragraph Comprehension subtest helps assess reading and understanding abilities, contributing to your overall AFQT score. Improving this involves focusing on analytical reading skills such as identifying main points, remembering key details, or summarizing passages.
x??

---

#### Question 1: Describing a Quaint Country Setting

The author is describing a quaint country setting.

:p What does the passage describe in terms of location?
??x
The passage describes a quaint country setting. There are no specific locations mentioned, but it is set in a rural or countryside area.
x??

---

#### Question 2: Direction of Brooks

A few miles north, the brooks run in an opposite direction (north).

:p What do the brooks run in relation to their direction in the village?
??x
The passage states that the brooks in the village run south. A few miles north of the village, these brooks run in the opposite direction, which is north.
x??

---

#### Question 3: Revolt Against Colombia

Panama revolted against Colombia.

:p What event or conflict did Panama have with Colombia?
??x
The passage states that Panama revolted against Colombia. It does not mention a fight over the canal or any specific conflict other than a general revolt, so this is incorrect.
x??

---

#### Question 4: Torn U.S. Flag

A torn U.S. flag can be professionally mended, but a severely torn flag should be destroyed. The preferred method of destruction is by burning.

:p What is the preferred method for destroying a severely torn U.S. flag?
??x
The preferred method for destroying a severely torn U.S. flag is by burning.
x??

---

#### Question 5: Purpose and Similarity of Guilds

Guilds had economic and social purposes, were similar to labor unions, protected merchants and craftspeople; they held considerable economic power.

:p What purpose did guilds serve in their communities?
??x
Guilds served dual purposes—they had both economic (by protecting members' interests) and social functions. They were also similar to modern labor unions and held significant economic power.
x??

---

#### Question 6: Wright Brothers and Aircraft

It took more than 4 years for the government to believe that anyone had flown a heavier-than-air craft. The historic flight was in December 1903, and the first aircraft was delivered to the government in August 1908.

:p How long did it take for the government to accept the Wright brothers' claim?
??x
It took more than 4 years for the government to believe that anyone had flown a heavier-than-air craft. The flight occurred in December 1903, and the first aircraft was delivered to the government in August 1908.
x??

---

#### Question 7: Freud's Comments on Memory

Freud comments on the characteristics of memory throughout the entire passage.

:p What does Freud comment on in the passage?
??x
Freud comments on the characteristics of memory throughout the entire passage. This indicates that the passage likely discusses various aspects or properties of human memory.
x??

---

#### Difference Between Troy and Common Weights
Background context: The passage explains that troy weights are used for precious metals like gold and silver, while common or avoirdupois weights are more general. It also provides specific details about the grain weight of an ounce in each system.

:p What is a common misconception regarding the weight of a common ounce?
??x
A common misconception might be that a common ounce weighs exactly 438 grains, whereas it actually weighs just shy of this number.
x??

---

#### Leadership and Worker Performance
Background context: The passage discusses leadership principles, specifically mentioning that showing interest in workers’ problems is a key principle. It also explains the outcome of effective leadership involvement.

:p Which action by leaders directly improves worker performance according to the passage?
??x
Showing interest in workers' problems.
x??

---

#### Leukemia and Blood Cells
Background context: The passage describes leukemia as a disease that interferes with blood production, specifically mentioning red blood cells. It also states where white blood cells are found.

:p How does leukemia affect the body according to the passage?
??x
Leukemia interferes with the body's ability to produce red blood cells.
x??

---

#### Types of Military Operations
Background context: The passage mentions that high-intensity conflict is listed as a type of military operation, but it also lists four other operational concepts.

:p Which type of military operation is NOT mentioned in the list provided by the passage?
??x
High-intensity conflict is already mentioned, so you should focus on the types not included.
x??

---

#### The JV 2020 Vision
Background context: The passage states that the Joint Vision 2020 (JV 2020) guides all military services with its vision of future war fighting. It also mentions other aspects but does not specifically reference training.

:p What does the JV 2020 primarily guide according to the passage?
??x
The JV 2020 primarily guides all the military services with its vision of future war fighting.
x??

---

#### Cytogenetics and Genetics Subfields
Background context: The passage differentiates cytogenetics from other genetic subfields, explaining that it focuses on the cellular basis of inheritance. It also mentions human genetics and microbial genetics.

:p What is a key difference between cytogenetics and other genetic subfields mentioned in the passage?
??x
Cytogenetics specifically deals with the study of the cellular basis of inheritance, distinguishing it from fields like human genetics or microbial genetics.
x??

---

#### Genetic Counselors' Role
Background context: The passage explains that genetic counselors advise couples and families on the chances of their offspring having specific genetic defects. It also clarifies what they do not do.

:p What is a common misunderstanding about genetic counseling according to the passage?
??x
A common misunderstanding might be that genetic counselors use genetics to prevent offspring from inheriting defects, whereas in reality, they advise on the chances rather than preventing such occurrences.
x??

---

#### Mathematics Knowledge
Background context: The text emphasizes the importance of mathematics for the AFQT score and suggests reviewing practice questions.

:p What is a key reason given for studying mathematics according to the passage?
??x
A key reason given is that math skills make up half of your AFQT score.
x??

---

#### Area Calculation
Background context: The passage explains how to calculate area by multiplying length times width, which is relevant when only one dimension (width) is provided.

:p How do you find the area of a rectangle if you know its width but not its length?
??x
To find the area, you need to determine the length and then multiply it by the width.
x??

---

#### Exponentiation Rules
Background context: The passage explains that when two exponents with the same base are multiplied, their exponents can be added.

:p How do you simplify an expression like $x^{24} \times x^{24}$?
??x
You simplify it by keeping the base and adding the exponents together, resulting in $x^{48}$.
x??

---

#### Area Calculation with Known Width
Background context: The passage provides a practical example of area calculation.

:p Given that you know the width but not the length of a rectangle, how can you calculate its area?
??x
You need to find the length first and then multiply it by the given width.
x??
---

#### Perimeter and Area of a Rectangle
Background context: The perimeter (P) of a rectangle is given by $P = 2(l + w)$, where $ l$is the length and $ w$ is the width. To find the area (A) of a rectangle, use the formula $ A = l \times w$.

:p Given that the perimeter of a rectangle is 36 inches and its width is 14 inches, what is the length?
??x
To solve for the length ($l$), we can plug in the known values into the perimeter formula:

$$P = 2(l + w)$$

Substituting $P = 36 $ and$w = 14$:

$$36 = 2(l + 14)$$

First, divide both sides by 2 to simplify:
$$18 = l + 14$$

Next, subtract 14 from both sides to isolate $l$:

$$l = 18 - 14$$
$$l = 4$$

Therefore, the length is 4 inches.

x??

---

#### Cube Root
Background context: The cube root of a number $n $ is a value that, when multiplied by itself three times, equals$n $. It can be represented as$\sqrt[3]{n}$.

:p What is the cube root of 64?
??x
The cube root of 64 is 4 because:

$$4^3 = 4 \times 4 \times 4 = 64$$

So, the answer is 4.

x??

---

#### Scientific Notation
Background context: Scientific notation is a way of writing numbers that are too large or too small to be conveniently written in decimal form. A number in scientific notation is written as $a \times 10^b $, where $1 \leq a < 10 $ and$b$ is an integer.

:p Convert the number 314,000 into scientific notation.
??x
To convert 314,000 into scientific notation:

- Move the decimal point to the left until it’s immediately after the first non-zero digit (3.14).
- Count the number of places you moved the decimal: 5 places.
- Therefore, $314,000 = 3.14 \times 10^5$.

The answer is $3.14 \times 10^5$.

x??

---

#### Reciprocal
Background context: The reciprocal of a number $n $ is the number which when multiplied by$n $, gives the product as 1. It can be represented as$\frac{1}{n}$.

:p What is the reciprocal of $\frac{1}{6}$?
??x
The reciprocal of $\frac{1}{6}$ is 6 because:

$$\left( \frac{1}{6} \right) \times 6 = 1$$

Therefore, the answer is 6.

x??

---

#### Solving Equations by Multiplication
Background context: To solve equations involving fractions or decimals, you can multiply both sides of the equation by a common factor to eliminate the fraction or decimal and isolate the variable.

:p Solve for $x $ in the equation$0.05x = 1$.
??x
To solve $0.05x = 1$:

- Multiply both sides by $\frac{1}{0.05}$, which is 20:
$$x = 1 \times 20$$
$$x = 20$$

Therefore, the answer is 20.

x??

---

#### Quadratic Equations
Background context: A quadratic equation in one variable can be written as $ax^2 + bx + c = 0$. It can be solved by factoring or using the quadratic formula. Factoring involves expressing the left side of the equation as a product of two binomials.

:p Solve for $x $ in the equation$x^2 - 6x + 5 = 0$.
??x
To solve $x^2 - 6x + 5 = 0$:

- Factor the quadratic expression:
$$(x - 1)(x - 5) = 0$$

Setting each factor equal to zero gives:
$$x - 1 = 0 \quad \text{or} \quad x - 5 = 0$$
$$x = 1 \quad \text{or} \quad x = 5$$

Therefore, the solutions are $x = 1 $ and$x = 5$.

x??

---

#### Area of a Circle
Background context: The area (A) of a circle is given by the formula $A = \pi r^2 $, where $ r $ is the radius. Here, we use $\pi \approx 3.14$.

:p Find the area of a circle with a radius of 5 inches.
??x
To find the area:

- Use the formula $A = \pi r^2$:
$$A = 3.14 \times (5)^2$$
$$

A = 3.14 \times 25$$
$$

A = 78.5$$

Therefore, the area is approximately 78.5 square inches.

x??

---

#### Inequalities
Background context: When solving inequalities involving division by a negative number, remember to switch the inequality sign direction.

:p Solve for $x $ in the inequality$-2x + 3 > 45$.
??x
To solve the inequality:

- First, subtract 3 from both sides:
$$-2x + 3 - 3 > 45 - 3$$
$$-2x > 42$$- Next, divide both sides by -2. Remember to switch the inequality sign because you are dividing by a negative number:
$$x < \frac{42}{-2}$$
$$x < -21$$

Therefore, the solution is $x < -21$.

x??

---

#### Volume of a Cylinder
Background context: The volume (V) of a cylinder is given by $V = \pi r^2 h $, where $ r $ is the radius and $ h $ is the height. Here, we use $\pi \approx 3.14$.

:p Find the volume of a cylinder with a radius of 5 inches and a height of 9 inches.
??x
To find the volume:

- Use the formula $V = \pi r^2 h$:
$$V = 3.14 \times (5)^2 \times 9$$
$$

V = 3.14 \times 25 \times 9$$
$$

V = 706.5$$

Therefore, the volume is approximately 706.5 cubic inches.

x??

---

#### Right Triangle
Background context: A right triangle has one 90-degree angle.

:p What makes a triangle a right triangle?
??x
A triangle is a right triangle if it contains one 90-degree angle.

x??

---

#### Parallelograms
Background context: In a parallelogram, opposite sides are equal in length.

:p Identify the property of a parallelogram.
??x
The property of a parallelogram is that its opposite sides are of equal length.

x??

---

#### Obtuse Angles
Background context: An obtuse angle measures more than 90 degrees but less than 180 degrees.

:p Define an obtuse angle.
??x
An obtuse angle is defined as an angle that measures more than 90 degrees but less than 180 degrees.

x??

---

#### Factoring Quadratic Equations
Background context: A quadratic equation can be factored by first factoring out the greatest common factor (GCF) and then solving for $x$.

:p Solve for $x $ in the equation$x^2 - 65x + 195 = 0$.
??x
To solve $x^2 - 65x + 195 = 0$:

- First, factor out the GCF (if any), but there isn't one here. Factor the quadratic expression:
$$x^2 - 65x + 195 = (x - 3)(x - 65) = 0$$

Setting each factor equal to zero gives:
$$x - 3 = 0 \quad \text{or} \quad x - 65 = 0$$
$$x = 3 \quad \text{or} \quad x = 65$$

Therefore, the solutions are $x = 3 $ and$x = 65$.

x??

---

#### Volume of a Cube
Background context: The volume (V) of a cube is given by $V = s^3 $, where $ s $ is the length of an edge. The surface area (SA) is given by $ SA = 6s^2$.

:p Find the volume and surface area of a cube with each edge measuring 3 inches.
??x
To find the volume:

- Use the formula $V = s^3$:
$$V = 3^3$$
$$

V = 27$$

Therefore, the volume is 27 cubic inches.

To find the surface area:

- Use the formula $SA = 6s^2$:
$$SA = 6(3)^2$$
$$

SA = 6 \times 9$$
$$

SA = 54$$

Therefore, the surface area is 54 square inches.

x??

---

#### Exponent Rules
Background context: When multiplying powers with the same base, add the exponents. For example,$a^m \times a^n = a^{m+n}$.

:p Simplify $x^3 \times x^3$.
??x
To simplify $x^3 \times x^3$:

- Use the exponent rule: $a^m \times a^n = a^{m+n}$:
$$x^3 \times x^3 = x^{3+3} = x^6$$

Therefore, the simplified form is $x^6$.

x??

---

#### Factorial
Background context: The factorial of a number $n $(denoted $ n!$) is the product of all positive integers less than or equal to $ n$. For example,$5! = 5 \times 4 \times 3 \times 2 \times 1$.

:p Simplify $4!$.
??x
To simplify $4!$:

- Calculate the product of all positive integers less than or equal to 4:
$$4! = 4 \times 3 \times 2 \times 1 = 24$$

Therefore, the answer is 24.

x??

---

#### Cube Root and Exponents
Background context: The cube root of a number $n $(denoted as $\sqrt[3]{n}$) can be expressed using exponents. For example,$\sqrt[3]{64} = 4 $ because$4^3 = 64$.

:p Simplify $x^{3} \times x^{3}$.
??x
To simplify $x^3 \times x^3$:

- Use the exponent rule: $a^m \times a^n = a^{m+n}$:
$$x^3 \times x^3 = x^{3+3} = x^6$$

Therefore, the simplified form is $x^6$.

x??

---

#### Multimeter and Its Components
Background context explaining the multimeter, its components, and their functions. The multimeter is an electronic instrument that combines several test equipment pieces, including an ammeter, which measures inline current.

:p What is a multimeter and what does it include?
??x
A multimeter includes several test equipment pieces, one of which is an ammeter, used to measure inline current.
x??

---
#### Rotor Bars in AC and DC Motors
Explanation on the difference between rotor bars in AC induction motors and DC motors.

:p What are rotor bars found in, and what do they indicate about the motor?
??x
Rotor bars are only found in AC induction motors. This indicates that DC motors do not have rotor bars.
x??

---
#### NTSC vs Other Standards
Explanation on broadcast standards and their differences, including NTSC, RGB, SECAM.

:p What does NTSC stand for, and what is its significance?
??x
NTSC stands for National Television System Committee and represents the current broadcast standard in the U.S. It is gradually being replaced by ATSC (Advanced Television Systems Committee).
x??

---
#### Current Measurement Units
Explanation on the units of measure for electric current.

:p What unit measures electric current, and what does it mean?
??x
Ampères (or amps) measure electric current. The formula to find current is I = V/R, where V is voltage and R is resistance.
x??

---
#### Fuse Symbol Recognition
Explanation on the function of fuses in electrical circuits.

:p What symbol represents a fuse in an electrical diagram?
??x
The symbol for a fuse is typically depicted as a short circuit line with a small circle at each end, indicating that it will blow (melt) if current exceeds its rating.
x??

---
#### GFCI Protection for Outlets Near Sinks
Explanation on the National Electric Code and safety standards.

:p What does NEC code say about outlets near sinks?
??x
NEC code requires that outlets within 6 feet of a sink be GFCI protected to ensure safety.
x??

---
#### Lamp Symbol in Electrical Diagrams
Explanation on the role of lamps as transducers.

:p What symbol represents a lamp in an electrical diagram, and what does it do?
??x
The symbol for a lamp is a simple circle with lines radiating out from it. Lamps convert electrical energy into light.
x??

---
#### Circuit Breaker Reset Process
Explanation on how to reset a circuit breaker.

:p How do you reset an electrical circuit breaker that has tripped?
??x
To reset an electrical circuit breaker, first move the handle to the "off" position and then back to the "on" position. When tripped, the handle moves between these two positions.
x??

---
#### Wire Gauge Numbering System
Explanation on the wire gauge numbering system.

:p What does a smaller wire number signify?
??x
A smaller wire number signifies a larger diameter or thicker wire.
x??

---
#### Insulation Material Properties
Explanation on the properties of insulators and conductors.

:p What is an example of a good conductor that is not better than copper?
??x
Aluminum is a good conductor but not necessarily better than copper in all applications.
x??

---
#### Series Circuit Characteristics
Explanation on how series circuits function and provide examples.

:p In what scenario does a series circuit cease to work, and why?
??x
In a series circuit, if any part of the path breaks, electricity stops flowing. An example is a string of Christmas lights that fails when one bulb burns out.
x??

---
#### Power Calculation Formula
Explanation on how to calculate power using current.

:p How do you calculate power in an electrical system?
??x
Power can be calculated by multiplying current (I) with voltage (V): P = I * V. For example, if V = 120 volts and I = 10 amperes, then P = 1200 watts.
x??

---
#### Potential vs Voltage
Explanation on the difference between potential and voltage.

:p What is the term for voltage in an electrical system?
??x
Potential refers to voltage. Low potential is anything less than 600 volts.
x??

---
#### Insulators Examples
Explanation on common insulating materials.

:p Which of the following is a good insulator: plastic, wood, or aluminum?
??x
Plastic and wood are good insulators, while aluminum is a conductor but not necessarily better than copper for this purpose.
x??

---
#### Ground Wires Identification
Explanation on identifying ground wires in electrical systems.

:p What color is a ground wire in an electrical system?
??x
Ground wires are always green or green/yellow striped to distinguish them from other types of wires.
x??

---
#### AM Modulation Overview
Explanation on the history and use of amplitude modulation.

:p What was the first type of audio modulation used, and when did it become common?
??x
AM (Amplitude Modulation) was the first type of audio modulation used in radio. It became common for its effectiveness with high frequency (HF) transmissions and Morse code.
x??

---
#### Silver as a Conductor
Explanation on the properties of silver as a conductor.

:p Why is silver a better conductor but not always preferred?
??x
Silver is a better electrical conductor than copper, but it’s more brittle and more expensive, making it less preferable in most applications.
x??

---
#### Oscillator Functions
Explanation on the functions of various electronic components.

:p What do oscillators, amplifiers, regulators, and transformers do?
??x
Oscillators produce high frequencies. Amplifiers change the amplitude of a signal. Regulators maintain constant voltage. Transformers change (transform) input voltage to output voltage.
x??

---
#### DC on AC Appliances Effects
Explanation on the effects of applying DC to an AC appliance.

:p What happens when DC is applied to an AC appliance?
??x
When DC is applied to an AC appliances, less resistance results in more current flowing through the wire and heat buildup.
x??

---
#### Parts of a Car Suspension
Explanation on identifying parts within a car's suspension system.

:p What is the wishbone part of a car’s suspension called, and what does it do?
??x
The wishbone is a component found in a car’s suspension. It helps stabilize the wheel by connecting to the control arms.
x??

---
#### Battery Maintenance for Non-Sealed Batteries
Explanation on maintaining non-sealed batteries.

:p What should be done if electrolyte levels are low in a non-sealed battery?
??x
If electrolyte levels are low in a non-sealed battery, distilled water should be added to bring the level back up.
x??

---
#### Cooling System Maintenance
Explanation on maintaining an automobile's cooling system.

:p Why is it important to flush an automobile’s cooling system periodically?
??x
Flushing an automobile’s cooling system periodically ensures that any accumulated contaminants and corrosion are removed, maintaining optimal performance and preventing engine damage.
x??

---
#### Carburetor Function in Engines
Explanation on the purpose of a carburetor in engines.

:p What is the function of a carburetor in a combustion engine with a carburetor?
??x
The carburetor’s function in a combustion engine that has a carburetor is to mix fuel and air for optimal engine performance.
x??

---
#### NOS Automotive Parts
Explanation on what New Old Stock (NOS) means.

:p What does NOS stand for, and where might you encounter it?
??x
NOS stands for New Old Stock. It refers to parts that were originally manufactured but never installed in a vehicle, typically found in restorations.
x??

---
#### Spark Plug Operation
Explanation on how spark plugs work in vehicles.

:p What voltage is necessary for spark plugs to function correctly?
??x
Spark plugs require a very high electrical voltage supplied by the coil and breaker to fire properly.
x??

---
#### Tire Valve Identification
Explanation on identifying tire valves.

:p Which type of valve do you use when filling your car’s tires with air?
??x
The Schrader valve is used for inflating car tires. It has a small pin that can be removed to release pressure or allow inflation.
x??

---
#### Frame Deformation Effects
Explanation on the effects of frame deformation.

:p What happens if a vehicle's frame is bent improperly?
??x
Bending the frame can cause improper tracking, meaning the right angle between the centerline and axles is not maintained.
x??

---
#### Vehicle Performance Specifications
Explanation on H ratings in vehicles.

:p What does an "H" rating signify in a vehicle’s performance specification?
??x
The "H" rating signifies the maximum sustained speed at which the vehicle can maintain stability without compromising safety or handling.
x??

---
#### Torque Wrench Usage
Explanation on using torque wrenches for specific tightness requirements.

:p When is it necessary to use a torque wrench, and why?
??x
A torque wrench should be used when specific tightness of screws and/or bolts is required to ensure proper installation and prevention of stripping.
x??

---
#### Hammer and Sledgehammer Differences
Explanation on the differences between hammers and sledgehammers.

:p What do mallets and sledgehammers not have, and why?
??x
Mallets and sledgehammers do not have heads that are designed for hammering. They lack this feature due to their specific uses.
x??

---
#### Lamp Symbol in Circuit Diagrams
Explanation on recognizing lamp symbols.

:p What symbol represents a lamp in electrical circuit diagrams?
??x
A lamp in an electrical circuit diagram is represented by a simple circle with lines radiating from it, indicating its function of converting electrical energy into light.
x??

---
#### Circuit Breaker Reset Process
Explanation on the steps to reset a tripped circuit breaker.

:p How do you properly reset a tripped circuit breaker?
??x
To reset a tripped circuit breaker, move the handle to the "off" position and then back to the "on" position. This process helps ensure safe operation of the electrical system.
x??

---
#### Wire Gauge Numbers
Explanation on wire gauge numbering.

:p How do smaller numbers relate to wire gauges?
??x
Smaller numbers in wire gauges indicate thicker or larger diameter wires, while higher numbers denote thinner wires.
x??

---
#### Insulation Material Identification
Explanation on distinguishing between good conductors and insulators.

:p Which of the following is a good conductor but not necessarily better than copper: plastic, wood, aluminum?
??x
Aluminum can be a good conductor but is generally not preferred over copper due to its higher cost and brittleness.
x??

---
#### Ground Wires Identification
Explanation on identifying ground wires in electrical systems.

:p How are ground wires typically identified in an electrical system?
??x
Ground wires are usually green or green/yellow striped, distinguishing them from other types of wires in the system.
x??

--- 
Each card covers a single key concept and is separated by "---". These flashcards should help you review and understand the concepts outlined in the text.


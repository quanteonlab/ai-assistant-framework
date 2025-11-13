# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 85)

**Starting Chapter:** Perusing Perimeter and Area

---

#### Measuring Angles and Arcs of a Circle
Background context: In geometry, angles and arcs are measured using degrees. A full circle is 360°. Smaller arcs or angles can be measured in minutes (') and even seconds ("). One degree equals 60 minutes, and one minute equals 60 seconds.
:p What is the relationship between degrees, minutes, and seconds?
??x
Degrees, minutes, and seconds are related as follows: 
- 1° = 60'
- 1' = 60"
For example, if you need to convert 35.75° into degrees, minutes, and seconds:
- The integer part is 35°.
- To find the remaining minutes, calculate (0.75 × 60) = 45'.
So, 35.75° = 35°45'0" or simply 35.75°.

To convert from minutes and seconds to degrees:
- First, convert seconds into minutes: $\text{minutes} + \frac{\text{seconds}}{60}$
- Then, add the result to the integer part of the degree.
```java
public class AngleConversion {
    public static double convertToDegrees(double minutes, int seconds) {
        return (minutes / 60.0) + (seconds / 3600.0);
    }
}
```
x??

---

#### Sum of Angles in a Quadrilateral
Background context: A quadrilateral is any polygon with four sides and four angles. The sum of the interior angles of any quadrilateral is always 360°. This concept is useful for solving problems involving shapes like squares, rectangles, rhombuses, and trapezoids.
:p What is the sum of the interior angles in a quadrilateral?
??x
The sum of the interior angles in a quadrilateral is always 360°.

This can be verified using the formula:
$$\text{Sum of interior angles} = (n - 2) \times 180^\circ$$where $ n $is the number of sides. For a quadrilateral,$ n = 4$:
$$(4 - 2) \times 180^\circ = 360^\circ$$

This formula can be applied to verify or solve for missing angles in a quadrilateral.
```java
public class QuadrilateralAngles {
    public static int sumOfInteriorAngles(int sides) {
        return (sides - 2) * 180;
    }
}
```
x??

---

#### Geometry on the ASVAB Test
Background context: The ASVAB's Mathematics Knowledge subtest contains questions that cover a wide range of topics, including geometry. You are likely to encounter problems involving shapes, angles, and measurements within the time constraints.
:p How many minutes do you have per question in the CAT-ASVAB for Geometry?
??x
In the CAT-ASVAB, you have 23 minutes to complete 15 questions on the Mathematics Knowledge subtest. This means you typically have about 1 minute and 46 seconds (or approximately 1:46) per question.

For the paper-and-pencil version, you have 24 minutes for 25 questions, giving you slightly more time at about 96 seconds or 1 minute and 36 seconds per question.
x??

---

#### Perimeter of 2D Shapes
Background context: The perimeter is the distance around a shape, which for 2D objects like squares, rectangles, circles, and triangles is the sum of the lengths of their sides. For a circle, it's calculated using $P = 2\pi r $, where $ r$ is the radius.
:p What is the formula to calculate the perimeter of a square?
??x
The perimeter of a square can be found by adding all four equal sides, which simplifies to $P = 4s $, where $ s$ is the side length.

For example:
```java
public class SquarePerimeter {
    public static double calculatePerimeter(double sideLength) {
        return 4 * sideLength;
    }
}
```
x??

---

#### Area of 2D Shapes
Background context: The area measures the flat space within a shape. For squares and rectangles, it's $A = l \times w $, where $ l $ is the length and $ w $ is the width. For circles, it’s calculated using $ A = \pi r^2 $, with$ r$ as the radius.
:p What is the formula to calculate the area of a rectangle?
??x
The area of a rectangle can be found by multiplying its length ($l $) and width ($ w $), which gives us the formula:$ A = l \times w$.

For example:
```java
public class RectangleArea {
    public static double calculateArea(double length, double width) {
        return length * width;
    }
}
```
x??

---

#### Types of Angles
Background context: Angles are formed when two lines intersect. They can be classified based on their degree measurements and relationships with other angles.
:p What is the definition of a right angle?
??x
A right angle is exactly 90 degrees.

For example, if you have an angle that measures 90 degrees:
```java
public class RightAngle {
    public static boolean checkRightAngle(double angle) {
        return Math.abs(angle - 90.0) < 0.01; // Allowing some margin for floating point precision
    }
}
```
x??

---

#### Complementary and Supplementary Angles
Background context: Complementary angles add up to 90 degrees, while supplementary angles add up to 180 degrees.
:p What are complementary angles?
??x
Complementary angles are two angles that equal 90 degrees when added together.

For example:
```java
public class ComplementaryAngles {
    public static boolean checkComplementary(double angle1, double angle2) {
        return (Math.abs(angle1 + angle2 - 90.0) < 0.01); // Allowing some margin for floating point precision
    }
}
```
x??

---

#### Parallel Lines and Transversals
Background context: When two parallel lines are intersected by a transversal, several angles are formed that have specific relationships with each other.
:p What is the definition of corresponding angles?
??x
Corresponding angles are on the same side of the transversal and both are either above or below the parallel lines. They have the same measure if the lines are parallel.

For example:
```java
public class CorrespondingAngles {
    public static boolean checkCorresponding(double angle1, double angle2) {
        return Math.abs(angle1 - angle2) < 0.01; // Allowing some margin for floating point precision
    }
}
```
x??

---

#### Vertical Angles
Background context: When two lines cross each other, they form vertical angles that are always equal.
:p What is the definition of vertical angles?
??x
Vertical angles oppose each other when two lines cross. They are always equal.

For example:
```java
public class VerticalAngles {
    public static boolean checkVertical(double angle1, double angle2) {
        return Math.abs(angle1 - angle2) < 0.01; // Allowing some margin for floating point precision
    }
}
```
x??

---

#### Naming Angles
Angles can be named using a vertex and points on their rays. The middle letter typically represents the vertex, e.g., DRJ where R is the vertex.
:p How do you name an angle with its vertex and two points?
??x
To name an angle with its vertex and two points, place the vertex in the middle of the three letters, such as DRJ where R is the vertex. This naming convention helps distinguish between angles sharing a common vertex.
```java
public class AngleNaming {
    public static void main(String[] args) {
        String angleName = "DRJ"; // D is the vertex, RJ are the points on the rays
        System.out.println("The vertex of angle " + angleName + " is: " + angleName.charAt(1));
    }
}
```
x??

---

#### Triangle Types and Properties
Triangles can be classified by their angles or sides:
- **Isosceles triangle**: Two equal sides; opposite angles are also equal.
- **Equilateral triangle**: Three equal sides; each angle measures 60°.
- **Right triangle**: One right angle (90°); remaining two angles are complementary, adding up to 90°. The side opposite the right angle is the hypotenuse.
- **Obtuse triangle**: An angle greater than 90°.
- **Scalene triangle**: Three unequal sides.

The perimeter of a triangle is found by summing the lengths of all three sides. The area can be calculated using the formula: $\text{Area} = \frac{1}{2} \times \text{base} \times \text{height}$.

:p What is the formula for finding the area of a right triangle?
??x
The formula for finding the area of a right triangle is:
$$\text{Area} = \frac{1}{2} \times \text{base} \times \text{height}$$

This formula works because one side (the base) and the perpendicular distance from that side to the opposite vertex form two sides of a rectangle, with the area being half of that rectangle.
```java
public class RightTriangleArea {
    public static double findArea(double base, double height) {
        return 0.5 * base * height;
    }
}
```
x??

---

#### Pythagorean Theorem
The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides.
Formula:$c^2 = a^2 + b^2$, where:
- $c$ is the length of the hypotenuse,
- $a $ and$b$ are the lengths of the legs.

:p How do you find the length of the hypotenuse in a right triangle with sides 5 and 12?
??x
To find the length of the hypotenuse in a right triangle with sides 5 and 12, apply the Pythagorean theorem:
$$c^2 = a^2 + b^2$$

Given $a = 5 $ and$b = 12$:
$$c^2 = 5^2 + 12^2$$
$$c^2 = 25 + 144$$
$$c^2 = 169$$

Taking the square root of both sides:
$$c = \sqrt{169}$$
$$c = 13$$

Thus, the hypotenuse is 13.
```java
public class PythagoreanTheorem {
    public static double findHypotenuse(double a, double b) {
        return Math.sqrt(a * a + b * b);
    }
}
```
x??

---

#### Quadrilaterals and Their Areas
Quadrilaterals are shapes with four sides. Common types include squares, rectangles, parallelograms, rhombuses, and trapezoids.

- **Square**: All sides are equal; all angles are right angles.
- **Rectangle**: Opposite sides are equal and all angles are right angles.
- **Parallelogram**: Opposite sides are parallel and equal. Opposite angles are also equal.
- **Rhombus**: All four sides are equal, but angles do not have to be right angles.
- **Trapezoid**: Exactly two opposite sides are parallel.

Area formulas:
- Square/Rectangle:$\text{Area} = l \times w $- Parallelogram:$\text{Area} = b \times h$, where height is perpendicular from base to the opposite side.
- Rhombus: $\text{Area} = \frac{1}{2} \times p \times q $, where $ p $ and $ q$ are lengths of diagonals.
- Trapezoid: $\text{Area} = \frac{1}{2} \times (a + b) \times h $, where $ a $ and $ b $ are the lengths of parallel sides, and $ h$ is height.

:p How do you find the area of a rectangle with length 10 units and width 5 units?
??x
To find the area of a rectangle with length 10 units and width 5 units, use the formula:
$$\text{Area} = l \times w$$

Substituting the given values:
$$\text{Area} = 10 \times 5$$
$$\text{Area} = 50 \text{ square units}$$
```java
public class RectangleArea {
    public static int findArea(int length, int width) {
        return length * width;
    }
}
```
x??

---


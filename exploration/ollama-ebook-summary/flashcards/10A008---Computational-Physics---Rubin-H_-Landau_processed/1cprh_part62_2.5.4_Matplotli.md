# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 62)

**Starting Chapter:** 2.5.4 Matplotlibs Animations

---

#### Scatter Plots
Scatter plots are used to visualize the relationship between two variables by plotting individual data points on a coordinate system. In Python, scatter plots can be created using Matplotlib.

:p What is a scatter plot and how is it used in Python?
??x
A scatter plot is a graphical representation of the relationship between two variables, where each point represents an observation. In Python, you create a scatter plot using `ax.plot(ox, yo, 'bo', markersize=3)`, which adds blue points to the plot with specified size.

```python
import matplotlib.pyplot as plt

# Example data
ox = [1, 2, 3, 4]
yo = [2, 3, 5, 7]

fig, ax = plt.subplots()
ax.plot(ox, yo, 'bo', markersize=3)

plt.show()
```
x??

---

#### 3D Surface Plots
In Python, 3D surface plots can be created using Matplotlib's `Axes3D` toolkit. These plots are used to visualize functions of two variables or multivariate data in a three-dimensional space.

:p What is the purpose of creating a 3D surface plot and how do you create one using Matplotlib?
??x
The purpose of a 3D surface plot is to visualize functions of two variables or multivariate data in a three-dimensional space. You can create such plots by setting up an `Axes3D` object from the Matplotlib toolkit.

Here's how you can create a simple 3D plot with wireframe and surface plotting:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

# Plot the surface
ax.plot_surface(x, y, z, color='b', rstride=1, cstride=1, linewidth=0, antialiased=False)

# Create wireframe plot for comparison
ax.plot_wireframe(x, y, z, color='r')

plt.show()
```
x??

---

#### 3D Scatter Plots
Scatter plots can also be extended to three dimensions (3D scatter plots) when dealing with multivariate data. These plots help visualize the distribution of points in a 3D space.

:p What is a 3D scatter plot and how do you create one using Python?
??x
A 3D scatter plot visualizes multivariate data by plotting each point as an individual dot in three-dimensional space, where the axes represent different variables. You can create such plots using `mpl_toolkits.mplot3d.Axes3D` from Matplotlib.

Here's a sample code for creating a 3D scatter plot:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate random data
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)

# Plot the scatter plot
sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

plt.show()
```
x??

---

#### 3D Scatter Plot Using Matplotlib

Background context: This section explains how to create a 3D scatter plot using Python and Matplotlib. The `Scatter3dPlot.py` program demonstrates this process.

:p What is the purpose of the `Scatter3dPlot.py` program?
??x
The program aims to produce a 3D scatter plot, showcasing the plotting capabilities of Matplotlib in three dimensions. It utilizes various data points for visualization purposes.
x??

---

#### Fourier Reconstruction Using a Slider

Background context: The `FourierMatplot.py` script performs a Fourier reconstruction of a sawtooth wave, allowing users to interactively control the number of waves included via a slider.

:p What is the role of the `Slider` class in the `FourierMatplot.py` program?
??x
The `Slider` class from Matplotlib's widgets module is used to create an interactive interface. It allows the user to adjust the number of wave components that contribute to the Fourier synthesis, enabling real-time updates on the plot.

Example code:
```python
from matplotlib.widgets import Slider

# Create a slider widget for controlling the number of waves
airwaves = plt.axes([0.25, 0.1, 0.65, 0.03])
shortwaves = Slider(airwaves, '# Waves', 1, 20, valinit=1)
```
x??

---

#### Matplotlib Animations

Background context: This section introduces the capability of creating animations using Matplotlib, although not as straightforwardly as with Vpython. The example provided in Listing 2.10 demonstrates how to animate a heat equation.

:p What is an example scenario for using Matplotlib's animation capabilities?
??x
An example scenario involves simulating the heat equation, where changes over time are visualized through animations or sequences of static images. This can help illustrate how temperatures evolve across different points in space.
x??

---

#### Plotting Exercises

Background context: The text encourages readers to experiment with plotting commands and options using Matplotlib. It suggests exploring various features such as zooming, saving plots, printing graphs, adjusting subplot spacing, rotating surfaces, etc.

:p What are some tasks suggested for enhancing plot customization?
??x
The tasks include:
- Zooming in/out on sections of a plot.
- Saving plots to files in different formats.
- Printing up graphs.
- Utilizing pull-down menu options.
- Increasing the space between subplots.
- Rotating and scaling surfaces.

These exercises aim to familiarize users with Matplotlib's extensive documentation and functionalities.
x??

---

#### Beam Support Problem

Background context: This section describes a mechanical problem involving a beam supported at two points, with an object sliding along it. It also hints at extending the problem by adding a third support.

:p What is the scenario described for the first part of the exercise?
??x
The scenario involves a beam of length \(L = 10\) meters and weight \(W = 400N\), resting on two supports separated by a distance \(d = 2\) meters. A box with a weight \(W_b = 800N\) starts above the left support and slides frictionlessly to the right with a velocity of \(v = 7m/s\).

The task is to write a program that calculates the forces exerted on the beam by both supports as the box moves along it.
x??

---

#### Beam Support Problem (Animation)

Background context: The problem extends beyond just calculating static forces and includes visualizing the changes dynamically.

:p How can an animation be created for this scenario?
??x
To create an animation, you would write a program that simulates the movement of the box on the beam and updates the forces at each step. This could involve drawing the beam and supports multiple times with updated force values as the box's position changes.

Here is a simplified pseudocode example:
```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize plot elements
fig, ax = plt.subplots()
beam, support1, support2 = initialize_plot_elements()

def update_position(time):
    # Update the position of the box based on time or step
    x_box = calculate_new_position(time)
    
    # Calculate forces on supports
    force_left, force_right = calculate_forces(x_box)

    # Update plot elements with new positions and forces
    support1.set_xdata(support_position(force_left))
    support2.set_xdata(support_position(force_right))

ani = FuncAnimation(fig, update_position, interval=100)
plt.show()
```
x??

---

#### Extended Beam Support Problem

Background context: The initial problem is extended by adding a third support under the right edge of the beam.

:p How does extending the two-support problem to include a third support change the analysis?
??x
Extending the two-support problem to include a third support under the right edge complicates the force distribution and equilibrium calculations. You would need to:
- Recalculate the forces at all three supports.
- Ensure that the sum of moments around each point equals zero.
- Update the equations to account for the new support position.

This requires revising the statics equations to include the additional constraint provided by the third support.
x??

#### EasyVisual.py: 2D Plots Using Visual Package
Visual is a Python module for creating graphical plots and animations. In this script, it demonstrates how to create two different 2D plots using `gcurve` and `gdots` methods.

:p What does `EasyVisual.py` demonstrate?
??x
The script demonstrates the creation of two 2D plots using the Visual package in Python. It shows a graph curve (`gcurve`) plotting a function \(5 \times \cos(2x) \times e^{-0.4x}\) and a set of dots (`gdots`) plotted at points where the x-coordinate varies from -5 to +5.

```python
from visual.graph import *  # Import Visual

# Create a graph display with title, x-axis label, y-axis label, foreground color, and background color.
graph1 = gdisplay(width=600, height=450,
                  title='Visual 2-D Plot', xtitle='x', ytitle='f(x)',
                  foreground=color.black, background=color.white)

# Create a graph curve with white color for the function 5 * cos(2*x) * exp(-0.4*x).
Plot1 = gcurve(color=color.white)
for x in range(0, 8.1, 0.1): 
    Plot1.plot(pos=(x, 5. * cos(2. * x) * exp(-0.4 * x)))

# Create a set of dots for the function cos(x).
Plot2 = gdots(color=color.black)
for x in range(-5., +5, 0.1):
    Plot2.plot(pos=(x, cos(x)))
```
x??

---

#### 3GraphVisual.py: Multiple Plots Using Matplotlib and NumPy
This script illustrates how to produce a single figure with multiple types of plots using the `matplotlib` and `numpy` libraries in Python.

:p What does `3GraphVisual.py` demonstrate?
??x
The script demonstrates creating a single plot that includes three different types of data representations: a curve (`gcurve`), vertical bars (`gvbars`), and dots (`gdots`). It uses these methods to visualize the functions \( \sin^2(x) \), \( \cos^2(x) \), and \( \sin(x)\cos(x) \).

```python
from visual.graph import *
from visual import *

string = "blue: sinˆ2(x), white: cosˆ2(x), red: sin(x)*cos(x)"
graph1 = gdisplay(title=string, xtitle='x', ytitle='y')

# Create a curve with yellow color for the function \sin^2(x).
y1 = gcurve(color=color.yellow, delta=3)

# Create vertical bars with white color.
y2 = gvbars(color=color.white)

# Create dots with red color.
y3 = gdots(color=color.red, delta=3)

for x in range(-5, 5, 0.1):
    y1.plot(pos=(x, sin(x) * sin(x)))
    y2.plot(pos=(x, cos(x) * cos(x) / 3.))
    y3.plot(pos=(x, sin(x) * cos(x)))
```
x??

---

#### 3Dshapes.py: VPython 3D Shapes
This script showcases various 3D shapes that can be created using the `visual` module in Python, demonstrating a range of geometric objects.

:p What does `3Dshapes.py` demonstrate?
??x
The script demonstrates creating and displaying several 3D shapes using the `visual` library. It illustrates how to create spheres, arrows, cylinders, cones, helixes, rings, boxes, pyramids, and ellipsoids.

```python
from visual import *

graph1 = display(width=500, height=500, title='VPython 3-D Shapes', range=10)

# Create a green sphere at the origin.
sphere(pos=(0, 0, 0), radius=1, color=color.green)

# Create a red sphere at (0,1,-3) with a larger radius.
sphere(pos=(0, 1, -3), radius=1.5, color=color.red)

# Create a cyan arrow from (3,2,2) to its axis direction (3,1,1).
arrow(pos=(3, 2, 2), axis=(3, 1, 1), color=color.cyan)

# Create a yellow cylinder with specified position and axis.
cylinder(pos=(-3, -2, 3), axis=(6, -1, 5), color=color.yellow)

# Create a magenta cone with specified position and axis.
cone(pos=(-6, -6, 0), axis=(-2, 1, -0.5), radius=2, color=color.magenta)

# Create an orange helix with specified parameters.
helix(pos=(-5, 5, -2), axis=(5, 0, 0), radius=2, thickness=0.4, color=color.orange)

# Create a magenta ring with specified position and axis.
ring(pos=(-6, 1, 0), axis=(1, 1, 1), radius=2, thickness=0.3, color=(0.3, 0.4, 0.6))

# Create a box at (5, -2, 2) with specified dimensions.
box(pos=(5, -2, 2), length=5, width=5, height=0.4, color=(0.4, 0.8, 0.2))

# Create a pyramid at (-1, -7, 1) with specified size and color.
pyramid(pos=(-1, -7, 1), size=(4, 3, 2), color=(0.7, 0.7, 0.2))

# Create an ellipsoid with specified axis lengths and color.
ellipsoid(pos=(-1, -7, 1), axis=(2, 1, 3), length=4, height=2, width=5, color=(0.1, 0.9, 0.8))
```
x??

---

#### EasyMatPlot.py: Plotting a Function Using Matplotlib
This script illustrates how to plot a function using the `matplotlib` library in Python.

:p What does `EasyMatPlot.py` demonstrate?
??x
The script demonstrates plotting a mathematical function \(5 \times \cos(2x) \times e^{-0.4x}\) using matplotlib's `plot` method.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values.
x = np.linspace(0, 8, 100)

# Define the function to plot.
y = 5 * np.cos(2 * x) * np.exp(-0.4 * x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of 5*cos(2*x)*exp(-0.4*x)')
plt.show()
```
x??

---

#### PondMatPlot.py: Monte Carlo Integration Using von Neumann Rejection
This script illustrates the use of the `matplotlib` and `numpy` libraries to perform a Monte Carlo integration using the von Neumann rejection method.

:p What does `PondMatPlot.py` demonstrate?
??x
The script demonstrates visualizing an integral by plotting an integrand function \( x \sin^2(x) \) over the interval \([0, 2\pi]\). It then uses Monte Carlo integration via von Neumann rejection to estimate the area under the curve.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define parameters.
N = 100
Npts = 3000
analyt = np.pi ** 2

x1 = np.arange(0, 2 * np.pi + 2 * np.pi / N, 2 * np.pi / N)
xi = []
yi = []
xo = []
yo = []

# Set up the plot.
fig, ax = plt.subplots()
y1 = x1 * np.sin(x1) ** 2

ax.plot(x1, y1, 'c', linewidth=4)
ax.set_xlim((0, 2 * np.pi))
ax.set_ylim((0, 5))
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'2$\pi$'])
ax.set_ylabel(r'$f(x) = x\,\sin^2 x$', fontsize=20)
ax.set_xlabel('x', fontsize=20)
fig.patch.set_visible(False)

def fx(x):
    return x * np.sin(x) ** 2

# Perform Monte Carlo integration.
j = 0
for i in range(Npts):
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    if u1 < (fx(u2)) / analyt:
        xi.append(u2 * 2 * np.pi)
        yi.append(fx(u2))
        j += 1

ax.scatter(xi, yi, c='r', s=5)

plt.show()
```
x??

--- 

Each card is designed to help you understand the key concepts and how they are implemented in Python using relevant libraries. The questions prompt for detailed explanations of each concept's context and implementation.

#### Simple Monte Carlo Integration
Background context: This example demonstrates a simple Monte Carlo method for estimating the area under a curve. The area is calculated by randomly sampling points within a defined bounding box and determining how many fall below the curve.

:p What is the purpose of this code snippet?
??x
The purpose of this code snippet is to estimate the area under a specific curve using the Monte Carlo integration technique. It does so by generating random points within a predefined rectangular region and counting how many lie beneath the curve defined by `fx(xx[i])`.

```python
import numpy as np

Npts = 10000  # Number of random points to generate
xi, yi, xo, yo = [], [], [], []  # Lists to store coordinates of points inside and outside the curve
j = 0  # Counter for points below the curve

for i in range(1, Npts):
    x = np.pi * np.random.rand(Npts)  # Random x values between 0 and pi
    y = 5.0 * np.random.rand(Npts)   # Random y values between 0 and 5
    
    if (y[i] <= fx(x[i])):  # Check if the point is below the curve
        xi.append(x[i])
        yi.append(y[i])
        j += 1  # Increment count of points below the curve
        
    else:
        xo.append(x[i])
        yo.append(y[i])

boxarea = 2.0 * np.pi * 5.0  # Area of the bounding box
area = boxarea * j / (Npts - 1)  # Estimate area under the curve

print(f"Estimated area: {area}")
```
x??

---

#### 3D Surface Plot with Matplotlib
Background context: This example shows how to create a 3D surface plot using matplotlib's `Axes3D` toolkit. The plot can be interactively rotated and scaled by the user.

:p What is the objective of this code snippet?
??x
The objective of this code snippet is to generate a 3D surface plot for a mathematical function, specifically the function \( Z = \sin(X) \cdot \cos(Y) \). This example demonstrates the use of `matplotlib` and its 3D plotting capabilities.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the grid for X and Y coordinates
x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)

# Calculate Z values as a function of X and Y
Z = np.sin(X) * np.cos(Y)

# Create the figure and the 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z)
```
x??

---

#### 3D Scatter Plot with Matplotlib
Background context: This example illustrates how to create a 3D scatter plot using `matplotlib`'s `Axes3D` toolkit. The code generates random points and plots them in a 3-dimensional space.

:p What is the purpose of this code snippet?
??x
The purpose of this code snippet is to produce a 3D scatter plot of randomly generated points colored by different colors for different ranges. This example helps visualize data points distributed across three dimensions.

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define the range and number of random points to generate
n = 100
c, m, zl, zh = 'r', 'o', -50, -25  # Color, marker, lower and upper bounds for Z-axis

# Generate random data points
xs = np.random.rand(n) * 10 + 23
ys = np.random.rand(n) * 100
zs = np.random.rand(n) * (zh - zl) + zl

# Create a figure and add the subplot with 3D projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D space
ax.scatter(xs, ys, zs, c=c, marker=m)

# Set labels for axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
```
x??

---

#### Animation of Cooling Bar using Matplotlib
Background context: This example demonstrates how to animate a cooling bar's temperature over time. The simulation uses the finite difference method to approximate the solution to the heat equation.

:p What is the main goal of this code snippet?
??x
The main goal of this code snippet is to simulate and visualize the cooling process of a bar by solving the one-dimensional heat equation using the finite difference method. The animation updates the temperature distribution along the bar over time, showing how the temperature changes as the bar cools down.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx = 101  # Number of grid points
Dx = 0.01414  # Spatial step size
Dt = 0.6  # Time step size
KAPPA = 210.0  # Thermal conductivity
SPH = 900.0  # Specific heat
RHO = 2700.0  # Density

# Initialize the temperature array
T = np.zeros((Nx, 2), float)

def init():
    for i in range(1, Nx - 1):
        T[i, 0] = 100.0
    T[0, 0] = 0.0
    T[Nx-1, 0] = 0.0

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 105), ylim=(-5, 110.0))
ax.grid(True)
plt.ylabel("Temperature")
plt.title("Cooling of a bar")

line, = ax.plot(range(Nx), T[range(Nx), 0], "r", lw=2)

# Animation function
def animate(dum):
    for i in range(1, Nx - 1):
        T[i, 1] = (T[i, 0] + cons * (T[i+1, 0] + T[i-1, 0] - 2.0 * T[i, 0]))
    
    line.set_data(range(Nx), T[range(Nx), 1])
    for i in range(1, Nx - 1):
        T[i, 0] = T[i, 1]
        
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=500, interval=20)
plt.show()
```
x??

---

#### Input and Output Operations with Matplotlib
Background context: This example shows how to perform input/output operations using `matplotlib` for reading from and writing to a file or keyboard. The code demonstrates basic file handling and user input in Python.

:p What is the purpose of this code snippet?
??x
The purpose of this code snippet is to illustrate how to read data from and write results to files, as well as taking inputs directly from the user via the keyboard. This example covers essential I/O operations that are useful for processing and storing data.

```python
import numpy as np

# Example: Reading and writing data to a file
data = np.loadtxt('input_file.txt')  # Read data from file
np.savetxt('output_file.txt', data)  # Write data to file

# Example: Taking user input
user_input = input("Enter some text: ")
print(f"User entered: {user_input}")
```
x??

---
#### Python 2 vs. Python 3 Input Handling
Background context: The provided script demonstrates how to handle input and formatted output in both Python 2 and Python 3, using `raw_input` for Python 2 and `input` for Python 3.

:p How does the script switch between handling `raw_input` in Python 2 and `input` in Python 3?
??x
The script checks the Python version at runtime. If the major version number is greater than 2, it uses `input`, which works similarly to `raw_input` in Python 2 for numerical input but directly evaluates strings as expressions.

```python
if int(version[0]) > 2:
    raw_input = input
```
x??

---
#### File Input and Output in Canopy
Background context: The script illustrates reading from a file named `Name.dat`, where the first entry on each line is treated as a name, and the second entry is treated as a radius. It calculates the area of a circle using these values.

:p How does the script handle reading a file with tabular data?
??x
The script opens `Name.dat` in read mode (`'r'`) and processes it line by line:
1. Each line is split into components.
2. The first component (name) is printed formatted as "Hi [name]".
3. The second component (radius) is converted to a float and used to calculate the area of a circle.

```python
inpfile = open('Name.dat', 'r')
for line in inpfile:
    line = line.split()
    name = line[0]
    r = float(line[1])
    print(f'Hi {name}')
    print(f'r = {r:.5f}')
```
x??

---
#### Area Calculation and Output to File
Background context: The script calculates the area of a circle using the radius provided by the user or read from `Name.dat`. It then writes the results back to an output file named `A.dat`.

:p How does the script handle writing calculated values to a file?
??x
The script opens `A.dat` in write mode (`'w'`) and writes the radius and area of the circle to it:

```python
outfile = open('A.dat', 'w')
outfile.write(f'r = {r:.5f}\n')
outfile.write(f'A = {A:.5f}\n')
```

x??

---
#### Formatting Strings in Python
Background context: The script uses formatted string output, demonstrating various formatting directives and escape characters. It includes examples of formatted output using `print` statements.

:p How does the script use formatted output to print a floating-point number?
??x
The script formats floating-point numbers using Python’s f-string syntax:

```python
print(f'Hi {name}')
print(f'r = {r:.5f}')
```

The `.5f` directive specifies that the number should be printed with 5 digits after the decimal point.

x??

---
#### Machine Precision Determination
Background context: The script determines machine precision by halving a value repeatedly until the increment `eps` is so small that adding it to 1.0 no longer changes the result. This process helps in understanding numerical limits of floating-point arithmetic.

:p How does the script determine machine precision?
??x
The script iteratively halves the initial value `eps`, starting from 1.0, until adding `eps` to 1.0 no longer results in a change:

```python
N = 10
eps = 1.0
for i in range(N):
    eps = eps / 2
    one_Plus_eps = 1.0 + eps
```

The loop continues until the addition of `eps` to 1 does not produce a different result.

x??

---


# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 4)

**Starting Chapter:** 2.5.4 Matplotlibs Animations

---

#### Scatter Plots
Background context: Sometimes, we need to visualize data points using scatter plots. This is particularly useful when there's a need to observe the distribution and relationships between variables. In some cases, adding a curve can help identify trends.

Example code:
```python
# PondMapPlot.py in Listing 2.7
import matplotlib.pyplot as plt

ox = [1, 2, 3, 4]  # Example x coordinates
yo = [5, 6, 7, 8]  # Example y coordinates

fig, ax = plt.subplots()
ax.plot(ox, yo, 'bo', markersize=3)  # Adds blue points of size 3 to the plot

plt.show()
```

:p How do you create a scatter plot using Matplotlib in Python?
??x
You use the `plot` method from the `matplotlib.pyplot` library. The syntax is `ax.plot(x, y, 'bo', markersize=3)`, where `ox` and `yo` are the x and y coordinates of the points you want to plot, `'bo'` specifies that blue circular markers should be used, and `markersize=3` sets the size of these markers.

The `ax.plot()` method is called on an axes object (`ax`) created from `plt.subplots()`, which provides the coordinate system for plotting. The resulting scatter plot displays points based on the specified coordinates.
x??

---

#### 3D Surface Plots
Background context: For visualizing more complex potential fields, such as dipole potentials, a three-dimensional (3D) surface plot is necessary. This type of plot represents z-dimension values as heights above a plane defined by x and y axes.

Example code:
```python
# Simple3Dplot.py in Listing 2.8
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

ax.plot_wireframe(X, Y, Z)  # Creates a wire-frame plot
plt.show()
```

:p How do you create a 3D surface and wireframe plot using Matplotlib?
??x
To create a 3D surface and wireframe plot in Python with Matplotlib, follow these steps:

1. Import necessary libraries.
2. Define the x and y coordinates as arrays of floats.
3. Use `np.meshgrid` to generate a grid from these coordinate vectors.
4. Calculate the z values based on the x and y coordinates using vector operations.
5. Add an axes object with 3D projection to the figure.
6. Plot the wireframe or surface.

Here's the code:
```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define coordinate vectors
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

# Generate a grid from the x and y vectors
X, Y = np.meshgrid(x, y)

# Calculate z values using vector operations
Z = X**2 + Y**2

# Create figure and add 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the wireframe or surface plot
ax.plot_wireframe(X, Y, Z)  # Creates a wire-frame plot

plt.show()
```
x??

---

#### Scatter Plots in 3D
Background context: For visualizing data points in three-dimensional space, a scatter plot can be used. This is particularly useful when dealing with data of the form (xi, yj, zk).

Example code:
```python
# Scatter3dPlot.py in Listing 2.9
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate random data points
x = np.random.rand(50) * 2 - 1
y = np.random.rand(50) * 2 - 1
z = x**2 + y**2

# Create a scatter plot in 3D
ax.scatter(x, y, z)

plt.show()
```

:p How do you create a 3D scatter plot using Matplotlib?
??x
To create a 3D scatter plot in Python with Matplotlib, follow these steps:

1. Import necessary libraries.
2. Generate the x, y, and z coordinates for your data points. In this example, random values are used.
3. Create a figure and add an axes object with 3D projection.
4. Use `ax.scatter` to plot the points in 3D.

Here's the code:
```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Generate random data points
x = np.random.rand(50) * 2 - 1
y = np.random.rand(50) * 2 - 1
z = x**2 + y**2

# Create figure and add 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot in 3D
ax.scatter(x, y, z)

plt.show()
```
x??

#### 3D Scatter Plot
Background context: The program `Scatter3dPlot.py` uses Matplotlib to produce a 3D scatter plot. This type of visualization helps understand the distribution and relationships between three variables.

:p What is the primary purpose of using a 3D scatter plot in data analysis?
??x
The primary purpose of using a 3D scatter plot is to visualize the relationship among three variables, where each axis represents one variable. This allows for better understanding of how these variables interact with each other.
x??

---

#### Fourier Reconstruction
Background context: The program `FourierMatplot.py` performs a Fourier reconstruction of a sawtooth wave using Matplotlib. Users can control the number of waves included via a slider, allowing real-time visualization.

:p How does the `Slider` widget work in the `FourierMatplot.py` program?
??x
The `Slider` widget works by allowing users to adjust the number of waves included in the Fourier reconstruction through an interactive bar. The code snippet provided uses Matplotlib's `Slider` class, which updates the plot based on the current value set by the user.

Example code:
```python
from matplotlib.widgets import Slider

# Assuming airwaves is a previously defined axes object
shortwaves = Slider(ax=airwaves, label='# Waves', valmin=1, valmax=20, valinit=5)

def update(val):
    # Update function to be called when the slider value changes
    new_val = shortwaves.val
    # Perform Fourier reconstruction with `new_val` waves

snumwaves.on_changed(update)
```
x??

---

#### Matplotlib Animations
Background context: Matplotlib can create animations, though not as simply as VPython. The Matplotlib examples page provides several examples, and the `Codes` directory includes some animation codes.

:p What are some common methods for creating animations in Matplotlib?
??x
Some common methods for creating animations in Matplotlib include defining a function that updates the plot at each frame and using the `FuncAnimation` class from the `matplotlib.animation` module. This involves specifying an initialization function, update function, and frames.

Example code:
```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the figure and axes
fig, ax = plt.subplots()

# Create a line object for plotting
line, = ax.plot([], [], lw=2)

def init():
    # Initialization function to set up the plot
    return line,

def animate(i):
    # Update function called at each frame
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, animate, frames=400, init_func=init, blit=True)

plt.show()
```
x??

---

#### Plotting Exercises
Background context: The text suggests exploring various plotting functionalities in Matplotlib, such as zooming, saving plots, printing graphs, and adjusting subplot spacing.

:p What are some key plot manipulations that the exercise encourages?
??x
The exercise encourages you to explore several key plot manipulations:
- Zooming in and out on sections of a plot.
- Saving your plots to files in various formats.
- Printing up your graphs.
- Utilizing options available from pull-down menus.
- Increasing space between subplots.
- Rotating and scaling surfaces.

These operations can be achieved using specific Matplotlib functions such as `zoomed_in`, `savefig`, `print_figure`, `subplots_adjust`, and `set_aspect`.
x??

---

#### Beam Support Forces
Background context: A beam of length $L$ supported at two points with a sliding box on it is analyzed to calculate the forces exerted by each support.

:p How would you model the system where a box slides along a beam supported at two points?
??x
To model the system, we need to consider Newton's laws and the equilibrium conditions for the beam. The key steps are:
1. Define the positions of supports and the box.
2. Calculate the forces exerted by each support as the box moves.

Example pseudocode:
```pseudocode
function calculateForces(L, d, W, Wb, v):
    # L: Length of the beam
    # d: Distance between supports
    # W: Weight of the box initially above left support
    # Wb: Total weight of the box
    # v: Velocity of the box

    x = 0  # Initial position of the box
    while x <= L - d:
        force_left = (W + Wb) * (x / d)
        force_right = (W + Wb) * ((L - x) / d)
        print("Position: ", x, "Force Left: ", force_left, "Force Right: ", force_right)
        x += 0.1  # Increment position by small step
```
x??

---

#### Three-Support Beam Analysis
Background context: The problem is extended to include a third support under the right edge of the beam.

:p How would you extend the two-support problem to include a third support?
??x
To extend the two-support problem to include a third support, we need to consider additional equilibrium conditions for the system. Specifically:
1. Define positions and forces at all three supports.
2. Use moment balance equations around each support point.

Example pseudocode:
```pseudocode
function calculateThreeSupportForces(L, d1, d2, W, Wb, v):
    # L: Length of the beam
    # d1: Distance from left end to first support
    # d2: Distance between supports (from second to third)
    # W: Weight of the box initially above left support
    # Wb: Total weight of the box
    # v: Velocity of the box

    x = 0  # Initial position of the box
    while x <= L - d1:
        force_left, force_middle, force_right = calculateForces(L, d1, d2, W, Wb, v, x)
        print("Position: ", x, "Force Left: ", force_left, "Force Middle: ", force_middle, "Force Right: ", force_right)
        x += 0.1  # Increment position by small step
```
x??

#### EasyVisual.py - 2D Plotting Using Visual Package
Background context: This script demonstrates how to create simple 2D plots using Python's `visual` package. The first plot uses a smooth curve, and the second plot combines curves, dots, and vertical bars.

```python
from visual.graph import * # Import necessary modules from the visual package

# Create a graph display for plotting the first function.
Plot1 = gcurve(color=color.white)  # White curve for the function

# Loop over x values to plot the sine function with exponential decay.
for x in arange(0., 8.1, 0.1):
    Plot1.plot(pos=(x, 5.*cos(2.*x)*exp(-0.4*x)))  # Plot points on the curve

# Create a graph display for plotting multiple types of graphs.
graph1 = gdisplay(width=600, height=450,
                  title='Visual 2-D Plot', xtitle='x', ytitle='f(x)',
                  foreground=color.black, background=color.white)

# Create a dot plot and loop over x values to place the dots accordingly.
Plot2 = gdots(color=color.black)
for x in arange(-5., +5.1, 0.1):
    Plot2.plot(pos=(x, cos(x)))  # Plot black dots on the cosine function
```

:p What does the `gcurve` method do?
??x
The `gcurve` method creates a curve plot on the graph display and allows for plotting multiple points that form a smooth curve. It is used here to plot the function $f(x) = 5 \cdot \cos(2x) \cdot e^{-0.4x}$.
x??

---

#### 3GraphVisual.py - Multiple Plots Using Visual Package
Background context: This script shows how to use the `visual` package to create multiple types of plots in a single graph display, including curves, vertical bars, and dots.

```python
from visual import *
from visual.graph import *

string = "blue: sin^2(x), white: cos^2(x), red: sin(x)*cos(x)"

# Create a graph display with specific settings.
graph1 = gdisplay(title=string, xtitle='x', ytitle='y')

# Plot the first function as a curve in yellow color.
y1 = gcurve(color=color.yellow, delta=3)

# Plot the second function using vertical bars.
y2 = gvbars(color=color.white)

# Plot the third function using dots.
y3 = gdots(color=color.red, delta=3)

# Loop over x values and plot points for all three functions.
for x in arange(-5., 5.1, 0.1):
    y1.plot(pos=(x, sin(x) * sin(x)))  # Plot curve
    y2.plot(pos=(x, cos(x) * cos(x) / 3.))  # Plot vertical bars
    y3.plot(pos=(x, sin(x) * cos(x)))  # Plot dots
```

:p What is the purpose of using `gvbars` in this script?
??x
The `gvbars` method creates a plot with vertical bars at each specified point on the x-axis. It is used here to visually represent the function $y = \cos^2(x) / 3$ as vertical bars, providing a different visual representation compared to curves or dots.
x??

---

#### 3Dshapes.py - 3D Shapes Using VPython
Background context: This script demonstrates how to create and display various 3D shapes using the `visual` package. It includes spheres, cylinders, arrows, cones, helices, rings, boxes, pyramids, and ellipsoids.

```python
from visual import *

graph1 = display(width=500, height=500, title='VPython 3-D Shapes', range=10)

# Create a green sphere.
sphere(pos=(0,0,0), radius=1, color=color.green)

# Create a red sphere at (0,1,-3) with a larger radius.
sphere(pos=(0,1,-3), radius=1.5, color=color.red)

# Create a cyan arrow from (3,2,2) to a point defined by the axis vector (3,1,1).
arrow(pos=(3,2,2), axis=(3,1,1), color=color.cyan)

# Create a yellow cylinder with specified position and axis.
cylinder(pos=(-3,-2,3), axis=(6,-1,5), color=color.yellow)

# Create a magenta cone with specific dimensions and position.
cone(pos=(-6,-6,0), axis=(-2,1,-0.5), radius=2, color=color.magenta)

# Create an orange helix with specified parameters.
helix(pos=(-5,5,-2), axis=(5,0,0), radius=2, thickness=0.4, color=color.orange)

# Create a magenta ring with specific dimensions and position.
ring(pos=(-6,1,0), axis=(1,1,1), radius=2, thickness=0.3, color=(0.3,0.4,0.6))

# Create a yellow box with specified dimensions and position.
box(pos=(5,-2,2), length=5, width=5, height=0.4, color=(0.4,0.8,0.2))

# Create a green pyramid with specific dimensions and position.
pyramid(pos=(2,5,2), size=(4,3,2), color=(0.7,0.7,0.2))

# Create an orange ellipsoid with specified axis and position.
ellipsoid(pos=(-1,-7,1), axis=(2,1,3), length=4, height=2, width=5, color=(0.1,0.9,0.8))
```

:p How does the `box` object in VPython differ from a sphere?
??x
The `box` object in VPython is used to create a rectangular prism (3D box) with specified dimensions and position, unlike the `sphere` which creates a round 3D shape. The `box` allows for more control over the dimensions (length, width, height), whereas a sphere has uniform radius.
x??

---

#### PondMatPlot.py - Monte Carlo Integration via von Neumann Rejection
Background context: This script demonstrates how to perform Monte Carlo integration using the von Neumann rejection method in Python. It uses matplotlib for plotting and numpy for numerical operations.

```python
import numpy as np, matplotlib.pyplot as plt

N = 100; Npts = 3000; analyt = np.pi ** 2
x1 = np.arange(0, 2 * np.pi + 2 * np.pi / N, 2 * np.pi / N)
xi = []; yi = []; xo = []; yo = []

fig, ax = plt.subplots()
y1 = x1 * np.sin(x1) ** 2  # Define the integrand function

# Plot the curve of the integrand.
ax.plot(x1, y1, 'c', linewidth=4)
ax.set_xlim((0, 2 * np.pi))
ax.set_ylim((0, 5))
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(['0', '$\pi $', '2 $\pi$'])
ax.set_ylabel(r'$f(x) = x \,\sin^2 x$', fontsize=20)
ax.set_xlabel('x', fontsize=20)
fig.patch.set_visible(False)

# Define the integrand function as a Python function.
def fx(x): return x * np.sin(x) ** 2

j = 0  # Counter for points inside the curve
```

:p What is the role of `fx(x)` in this script?
??x
The `fx(x)` function defines the integrand $f(x) = x \cdot \sin^2(x)$, which represents the mathematical function to be integrated. This function is used later in the integration process, specifically when using the von Neumann rejection method for Monte Carlo simulation.
x??

--- 

Each flashcard covers a different aspect of the provided scripts, ensuring comprehensive understanding and familiarity with the concepts involved.

---
#### Monte Carlo Simulation for Area Calculation
Background context: This concept involves using a Monte Carlo method to estimate the area under a curve. The Monte Carlo method relies on random sampling and probability, making it suitable for problems where traditional integration methods might be difficult or impractical.

:p What is the purpose of this code snippet?
??x
The purpose of this code snippet is to approximate the area under a curve using the Monte Carlo method. By generating random points within a known area (in this case, a box) and determining how many fall below the curve, we can estimate the area under the curve.

```python
import numpy as np

Npts = 1000  # Number of random points to generate
fx = lambda x: np.sin(x / np.pi) * np.sqrt(2 - x ** 2)

# Generate random points
xx = np.pi * np.random.rand(Npts)
yy = 5 * np.random.rand(Npts)

j = 0
for i in range(1, Npts):
    if (yy[i] <= fx(xx[i])):  # Below curve
        if (i <= 100): xi.append(xx[i])
        if (i <= 100): yi.append(yy[i])
        j += 1
    else:
        if (i <= 100): yo.append(yy[i])
        if (i <= 100): xo.append(xx[i])

boxarea = 2 * np.pi * 5  # Box area is 2π × 5
area = boxarea * j / (Npts - 1)  # Area under the curve

ax.plot(xo, yo, 'bo', markersize=3)
ax.plot(xi, yi, 'ro', markersize=3)

plt.title('Answers: Analytic = {0:.5f}, MC = {1:.5f}'.format(analytic_value, area))
```
x??

---
#### 3D Surface Plot with Matplotlib
Background context: This code snippet demonstrates how to create a 3D surface plot using the `matplotlib` library. The process involves generating a grid of points, calculating the height at each point based on a mathematical function, and then plotting this data as a surface.

:p What is the purpose of the `plot_surface` method in this code?
??x
The purpose of the `plot_surface` method is to create a smooth 3D surface plot from a set of points generated by a meshgrid. This method takes the X, Y coordinates and corresponding Z values (heights) as inputs and renders them as a continuous surface.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.1
x = np.arange(-3., 3., delta)
y = np.arange(-3., 3., delta)
X, Y = np.meshgrid(x, y)

Z = np.sin(X) * np.cos(Y)  # Surface height

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)  # Surface
ax.plot_wireframe(X, Y, Z, color='r')  # Add wireframe

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
```
x??

---
#### 3D Scatter Plot with Matplotlib
Background context: This code snippet illustrates how to create a 3D scatter plot using the `matplotlib` library. The data points are generated randomly, and each point is plotted in three-dimensional space. Different colors represent different categories of data.

:p What does this loop do in the code?
??x
This loop generates random X, Y, and Z coordinates for multiple points and plots them as colored markers on a 3D scatter plot.

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = np.random.uniform(23, 32, n)
    ys = np.random.uniform(0, 100, n)
    zs = np.random.uniform(zl, zh, n)

    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```
x??

---
#### Animation of Cooling Bar
Background context: This code snippet demonstrates how to animate a 1D cooling bar using the `matplotlib` library. The process involves simulating heat diffusion over time and updating the plot at each step.

:p What does the `animate` function do in this code?
??x
The `animate` function updates the temperature distribution of the bar over successive frames, simulating the cooling process. It calculates new temperatures based on finite difference equations and redraws the plot with updated values.

```python
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nx = 101
Dx = 0.01414
Dt = 0.6
KAPPA = 210.0  # Thermal conductivity
SPH = 900.0  # Specific heat
RHO = 2700.0  # Density

cons = KAPPA / (SPH * RHO) * Dt / (Dx * Dx)

T = zeros((Nx, 2), float)  # Temperature at first two time steps

def init():
    for i in range(1, Nx - 1): T[i, 0] = 100.0
    T[0, 0] = 0.0; T[0, 1] = 0.0
    T[Nx - 1, 0] = 0.0; T[Nx - 1, 1] = 0.0

init()

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 105), ylim=(-5, 110))
ax.grid()
plt.ylabel("Temperature")
plt.title("Cooling of a bar")

line, = ax.plot(range(Nx), T[range(Nx), 0], "r", lw=2)
plt.plot([1, 99], [0, 0], "r", lw=10)
plt.text(45, 5, 'bar', fontsize=20)

def animate(dum):
    for i in range(1, Nx - 1): T[i, 1] = (T[i, 0] + cons * (T[i + 1, 0] + T[i - 1, 0] - 2.0 * T[i, 0]))
    line.set_data(range(Nx), T[range(Nx), 1])
    for i in range(1, Nx - 1): T[i, 0] = T[i, 1]
    return line,

ani = animation.FuncAnimation(fig, animate, interval=25)

plt.show()
```
x??

---

#### Python 2 vs. Python 3 Input Handling
Python versions 2 and 3 differ in their handling of keyboard input, with `raw_input` being used in Python 2 and `input` in Python 3. The provided script demonstrates this transition by using a conditional statement to switch between the two based on the version of Python.

:p How does the script handle keyboard input for different versions of Python?
??x
The script uses an if-else statement to check which version of Python is being used and adjusts the input function accordingly. If the version number is greater than 2, `input` is used instead of `raw_input`. This ensures compatibility with both Python 2 and 3.

```python
if int(version[0]) > 2:  # Python 3 uses input, not raw_input
    raw_input = input
```
x??

---
#### Formatted Output in Python
The script demonstrates how to use formatted output using the `print` function with various formatting directives. It shows how to format floating-point numbers and strings.

:p How can you print a floating-point number with specific precision in Python?
??x
You can use the `%f` directive within the string passed to the `print` function along with the `percent` method to specify the desired precision for floating-point numbers.

```python
radius = 3.14159
print('you entered radius= %8.5f' % radius)
```
x??

---
#### Reading from a File in Python
The script illustrates how to read data from a file and process it line by line, splitting each line into components and performing operations based on the content of those lines.

:p How do you open and read a file in Python?
??x
To open a file for reading, use the `open` function with the appropriate mode ('r' for read). The script reads each line from the file, splits it using `split()`, and processes the components accordingly.

```python
inpfile = open('Name.dat', 'r')
for line in inpfile:
    line = line.split()
    name = line[0]
    r = float(line[1])
```
x??

---
#### Writing to a File in Python
The script demonstrates how to write formatted data back to a file, converting floating-point numbers and other types of variables into strings.

:p How do you write formatted output to a file in Python?
??x
To write formatted output to a file, use the `write` method with appropriate formatting directives. The script converts the radius and area (A) values to strings using the `%f` directive before writing them to the file.

```python
outfile = open('A.dat', 'w')
outfile.write('r= %13.5f' % r)
outfile.write('A = %13.5f' % A)
```
x??

---
#### Escape Characters in Python Strings
The script shows how to use escape characters within strings, such as `\t` for tab and `\\` for a literal backslash.

:p What are some common escape sequences used in Python strings?
??x
Some common escape sequences in Python include:
- `\t`: Tab character
- `\\`: Backslash character
- `\"`: Double quote character

Example usage:

```python
print("hello\tit’s me")
print("shows a backslash \\")
```
x??

---
#### Machine Precision Determination
The script illustrates how to determine the machine precision by halving an initial value repeatedly until the addition of this value to 1.0 no longer affects the result.

:p How does the script determine the approximate machine precision?
??x
The script initializes `eps` to 1.0 and repeatedly halves it while adding the current value of `eps` to 1.0. It continues this process until the addition no longer changes the value, indicating that further halving would result in a loss of significance.

```python
N = 10
eps = 1.0
for i in range(N):
    eps = eps / 2
    one_Plus_eps = 1.0 + eps
print('eps = ', eps, ', one + eps = ', one_Plus_eps)
```
x??

---


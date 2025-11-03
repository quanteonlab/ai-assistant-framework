# Flashcards: Game-Engine-Architecture_processed (Part 23)

**Starting Chapter:** 10 Tools for Debugging and Development. 10.1 Logging and Tracing

---

#### Logging and Tracing: Introduction
Background context explaining the importance of logging and tracing in game development. Debugging tools are crucial for making the game development process easier and less error-prone.

:p What is logging and tracing used for in game development?
??x
Logging and tracing are essential techniques to monitor and understand the state and behavior of a game application during development, helping developers identify and fix bugs more efficiently.
x??

---

#### Print Statements for Debugging: Basic Usage
Explanation on using print statements (printf) as a debugging tool. This method is still widely used in real-time programming due to its simplicity and effectiveness.

:p How do you use printf() for debugging?
??x
You can use `printf()` within your code to dump the internal state of your program, which helps identify issues during development.
For example:
```cpp
int x = 10;
int y = 20;
printf("The value of x is %d and the value of y is %d", x, y);
```
x??

---

#### Print Statements in Different Environments: Windows Console Applications
Explanation on how to use printf() for debugging in console applications under Linux or Win32 environments.

:p How do you print debug information in a console application using printf()?
??x
In a console application written in C/C++ running under Linux or Win32, you can produce output by printing to `stdout` or `stderr` via `printf()` or the C++ standard library’s iostream interface.
Example:
```cpp
#include <iostream>

int main() {
    int x = 10;
    std::cout << "The value of x is: " << x << std::endl;
    return 0;
}
```
x??

---

#### Debugging in Windowed Applications: Using OutputDebugString()
Explanation on how to use `OutputDebugString()` for debugging in windowed applications under Win32 environments.

:p How do you print debug information using OutputDebugString() in a windowed application?
??x
`OutputDebugString()` is useful for printing debugging information to Visual Studio's Debug Output window. However, it does not support formatted output and can only print raw strings.
Example:
```cpp
#include <windows.h>
int main() {
    char msg[] = "This is a debug message";
    OutputDebugStringA(msg);
    return 0;
}
```
x??

---

#### Custom Formatted Output Function: VDebugPrintF()
Explanation on creating a custom function to handle formatted output using `vsnprintf()`.

:p How do you create a custom function for formatted debugging in Windows game engines?
??x
You can wrap `OutputDebugString()` with a custom function that supports formatted output. Here’s an example implementation:
```cpp
#include <stdio.h>
#include <windows.h>

int VDebugPrintF(const char* format, va_list argList) {
    const U32 MAX_CHARS = 1024;
    static char s_buffer[MAX_CHARS];
    int charsWritten = vsnprintf(s_buffer, MAX_CHARS, format, argList);
    OutputDebugStringA(s_buffer);
    return charsWritten;
}

int DebugPrintF(const char* format, ...) {
    va_list argList;
    va_start(argList, format);
    int charsWritten = VDebugPrintF(format, argList);
    va_end(argList);
    return charsWritten;
}
```
x??

---

#### Verbosity Control Mechanism
Verbosity control allows developers to adjust the level of detail printed during debugging. The simplest implementation stores the verbosity level in a global integer variable, and then uses this value to conditionally print messages.

:p How does the `VerboseDebugPrintF` function manage verbosity levels?
??x
The `VerboseDebugPrintF` function checks if the provided verbosity level is greater than or equal to the current `g_verbosity`. If it is, it proceeds to print the message. Here's a pseudocode implementation:

```c
int g_verbosity = 0; // Global variable for verbosity

void VerboseDebugPrintF(int verbosity, const char* format, ...) {
    va_list argList;
    
    if (g_verbosity >= verbosity) { // Check if verbosity level is high enough
        va_start(argList, format); 
        VDebugPrintF(format, argList); // Print the message using a helper function
        va_end(argList);
    }
}
```

This ensures that print statements are only output when necessary. The `g_verbosity` variable can be adjusted via command-line arguments or runtime settings to control what information is printed.

x??

---

#### Channel-Based Debug Output System
Debug messages can often be categorized into different channels based on the system they originate from (e.g., animation, physics). This allows developers to focus on specific areas of interest without being overwhelmed by irrelevant data.

:p How does a channel-based debug output system work?
??x
In a channel-based system, each message is associated with one or more channels. The `VerboseDebugPrintF` function can include an additional argument for the channel and consults a list of active channels to decide whether to print the message.

Here's an example implementation:

```c
enum Channel { ANIMATION, PHYSICS, AI, RENDERING, NUM_CHANNELS };

int g_verbosity = 0; // Global verbosity variable

void VerboseDebugPrintF(int verbosity, int channel, const char* format, ...) {
    va_list argList;
    
    if (g_verbosity >= verbosity) { 
        va_start(argList, format);
        
        bool activeChannel = false;
        for (int i = 0; i < NUM_CHANNELS; ++i) {
            if ((1 << i) & g_active_channels_mask) // Check if channel is active
                activeChannel = true;
        }
        
        if (activeChannel)
            VDebugPrintF(format, argList);
        
        va_end(argList);
    }
}
```

The `g_active_channels_mask` is a bitmask where each bit corresponds to a channel. By checking the bits in this mask, you can determine which channels are active and whether to print the message.

x??

---

#### Logging and Tracing
Logging allows developers to record debug output for later analysis. This is particularly useful when dealing with issues that occur after runtime or in environments where direct debugging is difficult.

:p Why is it important to log debug output?
??x
Logging ensures that critical information can be captured even if the application crashes or the issue occurs outside of normal operating conditions. By maintaining a record, developers can diagnose problems more effectively later on.

For example, using a logging system in C++:

```c
void LogDebugMessage(int verbosity, int channel, const char* format, ...) {
    va_list argList;
    
    if (g_verbosity >= verbosity) { 
        va_start(argList, format);
        
        bool activeChannel = false;
        for (int i = 0; i < NUM_CHANNELS; ++i) {
            if ((1 << i) & g_active_channels_mask)
                activeChannel = true;
        }
        
        if (activeChannel)
            VLogPrintF(format, argList); // Print the message to a file or log

        va_end(argList);
    }
}
```

This function logs messages only when necessary and ensures that important data is not lost during crashes.

x??

---

#### Mirroring Output to Files
Mirroring debug output to files helps in diagnosing issues by providing a persistent record of all debug information, independent of the current verbosity settings or active channels.

:p Why should you mirror debug output to log files?
??x
Mirroring debug output to log files is crucial because it allows developers to analyze problems that occur after runtime. By maintaining log files, critical data can be captured even if the application crashes or if issues are not immediately apparent during normal operation.

Here's an example of how to implement this in C++:

```c
void LogDebugMessageToFile(int verbosity, int channel, const char* format, ...) {
    va_list argList;
    
    if (g_verbosity >= verbosity) { 
        va_start(argList, format);
        
        bool activeChannel = false;
        for (int i = 0; i < NUM_CHANNELS; ++i) {
            if ((1 << i) & g_active_channels_mask)
                activeChannel = true;
        }
        
        if (activeChannel) {
            FILE* logFile = fopen("debug.log", "a"); // Open or create the log file
            if (logFile != NULL) {
                fprintf(logFile, format, argList); // Write to the log file
                fclose(logFile);
            }
        }
        
        va_end(argList);
    }
}
```

This function ensures that all debug messages are written to a log file, regardless of verbosity settings or active channels.

x??

---

#### Crash Reports
Crash reports are a crucial tool for identifying and fixing issues in games. When a game crashes, these reports can provide detailed information that helps developers understand why and how it failed.

These reports often contain several pieces of useful data:
- Current level(s) being played at the time of the crash.
- Player character's world-space location when the crash occurred.
- The animation/action state of the player during the crash.
- Gameplay scripts running at the time of the crash, which can help pinpoint the source of the issue.
- A stack trace showing the call sequence leading to the crash.
- Memory allocator states (free memory, fragmentation level).
- Any other relevant information that might be helpful.

:p What kind of information should be included in a crash report?
??x
The answer includes several key pieces of data:
- Current level(s)
- Player's world-space location and state
- Running gameplay scripts
- Stack trace for call sequence
- Memory allocator states
- Other relevant details
```java
// Example function to generate a basic crash report
public void generateCrashReport() {
    System.out.println("Current Level: " + getCurrentLevel());
    System.out.println("Player Location (World Space): " + getPlayerLocation());
    System.out.println("Player State: " + getPlayerState());
    // Print running scripts or stack trace
}
```
x??

---

#### Debug Drawing Facilities
Debug drawing facilities are essential tools for visualizing mathematical and logical errors during development. Modern games rely heavily on mathematical operations to position, orient, move objects, test collisions, cast rays, and more.

These debug drawings include colored lines, simple shapes, 3D text, etc., which are removed before shipping the game. They help developers quickly identify issues by providing a visual representation of complex calculations.

:p What is the purpose of a debug drawing facility in game development?
??x
The primary purpose of a debug drawing facility is to visualize complex mathematical operations and logical errors during development. This makes it easier to spot issues that would otherwise be difficult to understand from code or numbers alone.
```java
// Example function to draw a line representing projectile trajectory
public void drawProjectileTrajectory(Vector3 start, Vector3 end) {
    // Code to draw the line in 3D space
}
```
x??

---

#### Stack Trace Information
Stack traces are crucial for understanding the sequence of function calls leading up to an error. Most operating systems provide a mechanism for generating stack traces during crashes.

:p What is a stack trace and why is it important?
??x
A stack trace is a report that shows the sequence of function calls leading up to a particular point in the code, often at the time of a crash or exception. It is important because it helps developers understand the context in which an error occurred.
```java
// Example pseudo-code for generating a stack trace
function generateStackTrace() {
    StackTraceElement[] elements = Thread.currentThread().getStackTrace();
    for (StackTraceElement element : elements) {
        System.out.println(element);
    }
}
```
x??

---

#### Memory Allocator States
Memory allocators manage the allocation and deallocation of memory. Information about these allocators can be very useful when debugging issues related to low memory conditions.

:p What information should be included in a crash report regarding memory allocators?
??x
Information to include in a crash report for memory allocators might include:
- Amount of free memory
- Degree of fragmentation
```java
// Example function to get memory allocator state
public MemoryState getMemoryAllocatorState() {
    MemoryState state = new MemoryState();
    state.freeMemory = getFreeMemorySize();
    state.fragmentationLevel = calculateFragmentationLevel();
    return state;
}
```
x??

---

#### Screenshot of the Game at Crash Time
Screenshots can provide a visual reference to understand what was happening in the game when it crashed.

:p Should screenshots be included in crash reports, and why?
??x
Including screenshots in crash reports is beneficial because they provide a visual reference of what was happening in the game just before or during the crash. This helps developers correlate the state of the game with the stack trace and other data.
```java
// Example function to capture a screenshot
public void takeScreenshot(String filename) {
    // Code to capture a screenshot and save it as filename
}
```
x??

#### Visualizing NPC Perception and Line of Sight

Background context: In video game development, especially for games like "The Last of Us: Remastered," developers use visual aids to understand how non-playable characters (NPCs) perceive the player. This is crucial for debugging and ensuring that NPCs behave as intended.

:p How does an NPC in "The Last of Us: Remastered" visualize the player's position when line of sight is broken?

??x
In "The Last of Us: Remastered," an NPC visualizes the player's last known location even if the line of sight (LOS) is broken. This visualization helps developers ensure that NPCs react appropriately to changes in visibility, such as when a player sneaks away unnoticed.

For example, consider this pseudo-code snippet:
```java
public class Npc {
    private Vector3 lastKnownPosition;

    public void updatePerception(Vector3 playerPosition, boolean hasLOS) {
        if (!hasLOS) {
            // Update the NPC's perception to the last known position of the player.
            lastKnownPosition = playerPosition;
        }
    }
}
```
The `updatePerception` method updates the NPC's perception based on whether it still has line of sight. If LOS is broken, the NPC retains the last known player position.

x??

---

#### Visualizing Explosion Blast Radius

Background context: The Naughty Dog engine uses wireframe spheres to visualize the dynamic explosion radius. This helps developers understand and tweak the blast effects during development without requiring physical testing.

:p How does a wireframe sphere help in visualizing an explosion's blast radius?

??x
A wireframe sphere is used as a visualization tool for the expanding blast radius of an explosion. By dynamically scaling the size of this sphere, developers can see how the blast affects different areas within the game world.

Here’s a simplified pseudo-code example:
```java
public class Explosion {
    private float radius;

    public void updateBlastRadius(float playerDistance) {
        // Example logic: scale up the blast radius based on the distance from the player.
        if (playerDistance > 10.0f && playerDistance < 20.0f) {
            radius = 5.0f;
        } else if (playerDistance >= 20.0f) {
            radius = 7.0f;
        }
    }

    public void render() {
        // Render the wireframe sphere with updated blast radius.
        drawWireframeSphere(radius);
    }
}
```
In this code, `updateBlastRadius` adjusts the blast radius based on the player's distance from the explosion center, and `render` method draws a wireframe sphere at that radius.

x??

---

#### Visualizing Drake’s Ledge Hang and Shimmy System

Background context: In the "Uncharted" series, visual aids such as spheres and vectors are used to debug and develop the AI system for characters like Drake. This helps in fine-tuning movements like hanging from ledges and shimming along them.

:p How do circles and lines help visualize the gameplay mechanics related to ledge hang and shimmy?

??x
Circles represent the different radii used by Drake to search for ledges he can climb or hang onto, while a line shows which specific ledge he is currently attached to. This visualization aids developers in debugging and refining the AI's decision-making process during gameplay.

Here’s an example of how this might be implemented:
```java
public class LedgeHangSystem {
    private Circle searchRadius;
    private Line currentLedgeLine;

    public void updateSearchRadius(float distance) {
        // Example logic to set the search radius based on game state.
        if (distance > 5.0f && distance < 10.0f) {
            searchRadius = new Circle(3.0f);
        } else if (distance >= 10.0f) {
            searchRadius = new Circle(4.0f);
        }
    }

    public void setLedge(Ledge ledge) {
        // Set the current line based on the selected ledge.
        currentLedgeLine = new Line(ledge.position, ledge.attachmentPoint);
    }

    public void render() {
        // Render both search radius and current ledge line.
        drawCircle(searchRadius);
        drawLine(currentLedgeLine);
    }
}
```
In this example, `updateSearchRadius` and `setLedge` methods adjust the visual elements based on the game state, while `render` method draws them for inspection.

x??

---

#### Debug Drawing API Requirements

Background context: A debug drawing API is essential in video game development as it allows developers to visualize various aspects of gameplay mechanics. This helps in debugging and optimizing performance without disrupting normal game operations.

:p What are the key requirements for a debug drawing API?

??x
A debug drawing API should be simple, easy to use, and support a wide range of primitives such as lines, spheres, points, coordinate axes, bounding boxes, and formatted text. It must offer flexibility in how these primitives can be drawn, including options for color, line width, sphere radii, point size, axis length, and more.

Additionally, the API should provide both world-space and screen-space drawing capabilities:
- **World-Space Drawing:** Useful for annotating objects in a 3D scene.
- **Screen-Space Drawing:** Helpful for displaying debugging information on a heads-up display (HUD) independent of camera position or orientation.

Here is an example of how such a debug drawing API might be structured:
```java
public interface DebugDrawer {
    void drawLine(Vector3 start, Vector3 end, Color color, float lineWidth);

    void drawSphere(Vector3 center, float radius, Color color);

    void drawText(String text, Vector3 position, Color color, float fontSize);
    
    // More methods for drawing other primitives...
}

public class GameDebugger {
    private DebugDrawer debugDraw;

    public void setupDebugDraw(DebugDrawer drawer) {
        this.debugDraw = drawer;
    }

    public void drawWorldSpacePrimitives() {
        debugDraw.drawLine(Vector3.ZERO, new Vector3(10.0f, 20.0f, 5.0f), Color.RED, 1.0f);
        debugDraw.drawSphere(new Vector3(5.0f, 6.0f, 7.0f), 2.0f, Color.BLUE);
    }

    public void drawScreenSpacePrimitives() {
        debugDraw.drawText("Player Health: 100%", new Vector2(10.0f, 10.0f), Color.GREEN, 36.0f);
    }
}
```
This example demonstrates how a `DebugDrawer` interface can be implemented with methods to draw various primitives in both world and screen spaces.

x??

---

#### Primitive Drawing with Depth Testing
Background context: This concept discusses how to draw primitives (basic shapes and lines) in a 3D scene, allowing for both depth testing and its disabling. Enabling depth testing ensures that only visible parts of the primitives are drawn, while disabling it makes the drawing more straightforward but may hide parts of the primitives behind objects.

:p How does enabling or disabling depth testing affect primitive drawing?
??x
When depth testing is enabled, primitives will be occluded by real objects in your scene. This means they can be hidden and difficult to see. However, this helps visualize their correct depth within the scene. Disabling depth testing makes the primitives appear on top of all other objects, ensuring visibility but making it harder to determine their actual depth.

```cpp
// Example code snippet for enabling/disabling depth testing in a rendering API
void enableDepthTesting(bool enable) {
    if (enable) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}
```
x??

---

#### Drawing Primitives Anywhere in Code
Background context: Debugging often requires drawing primitives at various points within the code. Most rendering engines have specific phases for submitting geometry, but a flexible API allows calls from anywhere.

:p Can the drawing API be called from any part of the code?
??x
Yes, the drawing API should allow function calls from any point in the code to facilitate debugging. For instance, if you need to draw a line segment or a sphere at runtime based on certain conditions, this flexibility is crucial. However, such drawing calls need to be queued up and processed during a specific rendering phase, typically at the end of each frame.

```cpp
// Example of adding a debug line in code
void someFunction() {
    // Some logic that might require debugging
    g_debugDrawMgr.AddLine(Vector3(0, 0, 0), Vector3(10, 0, 0), Color::Red, 2.0f);
}
```
x??

---

#### Lifetime of Debug Primitives
Background context: The lifetime of a debug primitive determines how long it remains on-screen after being requested. This is crucial for maintaining consistent and useful debugging information without causing visual flicker.

:p What controls the duration or lifetime of debug primitives?
??x
The lifetime of a debug primitive can be controlled by setting a duration when adding the primitive to the drawing queue. If the code that draws the primitive calls frequently (e.g., every frame), the duration might be set to 1 frame so the primitive is refreshed continuously. For less frequent or intermittent calls, primitives should have a longer lifetime, often several frames.

```cpp
// Example of setting a non-zero duration for a debug sphere
g_debugDrawMgr.AddSphere(Vector3(0, 0, 0), 1.0f, Color::Blue, 5.0f);
```
x??

---

#### Efficient Handling of Multiple Debug Primitives
Background context: Managing a large number of debug primitives efficiently is essential to ensure the game remains usable even when debugging features are active.

:p How does an efficient debug drawing API handle multiple primitives?
??x
An efficient debug drawing API should queue up all incoming debug drawing requests and process them during a specific phase, usually at the end of each frame. This ensures that rendering performance is not significantly affected by debugging needs. For instance, Naughty Dog's engine uses such a system where `AddLine`, `AddSphere`, etc., add primitives to a queue, which are then drawn in a unified pass.

```cpp
// Example of adding multiple debug lines
void drawDebugLines() {
    g_debugDrawMgr.AddLine(Vector3(0, 0, 0), Vector3(10, 0, 0), Color::Red, 2.0f);
    g_debugDrawMgr.AddLine(Vector3(5, -5, 0), Vector3(5, 5, 0), Color::Green, 1.5f);
}
```
x??

---

#### Debug Draw Manager API Usage
Background context: The provided text describes how a debug draw manager is used within game code to visualize elements such as velocity vectors and names of objects. It highlights that these drawings are typically added to a list rather than drawn immediately, allowing efficient rendering at specific times during the game loop.

:p How does the `DebugDrawManager` API work in the context of adding visual elements like lines or text?
??x
The `DebugDrawManager` adds visual elements to an internal list rather than drawing them immediately. These elements are then rendered together efficiently by the engine later in the game loop, which optimizes performance and simplifies the rendering pipeline.

```cpp
extern DebugDrawManager g_debugDrawMgr2D;

void Vehicle::Update() {
    // Do some calculations...
    const Point& start = GetWorldSpacePosition();
    Point end = start + GetVelocity();
    
    // Add a red line to visualize velocity.
    g_debugDrawMgr.AddLine(start, end, kColorRed);
    
    // Do some other calculations...
    {  // Scope for the string
        char buffer[128];
        sprintf(buffer, "Vehicle %s: passengers %d", GetName(), GetNumPassengers());
        
        const Point& pos = GetWorldSpacePosition();
        
        // Add text to display name and passenger count.
        g_debugDrawMgr.AddString(pos, buffer, kColorWhite, 0.0f, false);
    }
}
```
x??

---

#### In-Game Menu System Overview
Background context: The text explains the importance of in-game menus for configuring game settings during development. These menus allow programmers and artists to adjust global settings without stopping the game or modifying source code.

:p What is the primary purpose of an in-game menu system?
??x
The primary purpose of an in-game menu system is to provide a way for developers, artists, and designers to configure various game engine options and features while the game is running. This allows quick adjustments without needing to stop the game or recompile code.

:p How does bringing up in-game menus typically affect gameplay?
??x
Bringing up in-game menus usually pauses the game temporarily. This pause allows developers to examine and adjust settings more effectively, as they can play through a scenario until just before an issue occurs, then use the menus to visualize problems clearly without the interruption of normal gameplay.

:x??

---

#### Example In-Game Menu Structure
Background context: The text discusses the hierarchical structure of in-game menus, showing how submenus are organized for easy navigation. It mentions examples from The Last of Us: Remastered, highlighting major subsystems like rendering and mesh options.

:p How do in-game menu systems typically organize their options?
??x
In-game menu systems usually organize their options hierarchically, with main top-level menus leading to submenus that represent different subsystems or features. This structure makes it easy for users to navigate through various settings related to game mechanics, rendering, physics, and other aspects.

:p Can you give an example of a potential in-game menu entry?
??x
A typical in-game menu entry might allow toggling global Boolean settings, adjusting integer values, calling functions, or opening submenus. For instance:
- Toggle "Show Debug Lines" (Boolean)
- Set "Anti-Aliasing Level" (Integer)
- Call function to adjust camera position

:x??

---

#### Detailed Menu System Functionality
Background context: The text explains the flexibility of in-game menus, detailing how they can handle complex tasks such as visualizing debug information or pausing gameplay.

:p What features do in-game menu systems typically support?
??x
In-game menu systems typically support a wide range of functionalities:
- Toggling global Boolean settings (e.g., "Show Debug Lines")
- Adjusting global integer and floating-point values (e.g., "Anti-Aliasing Level", "Render Distance")
- Calling arbitrary functions to perform specific tasks within the engine
- Bringing up submenus for hierarchical navigation

:x??

---

#### Example In-Game Menu Display in Naughty Dog Engine
Background context: The text provides screenshots from The Last of Us: Remastered, illustrating the structure and appearance of in-game menus. It shows top-level and submenus that represent different subsystems.

:p What do the screenshots illustrate about in-game menus?
??x
The screenshots from The Last of Us: Remastered demonstrate how in-game menus are structured and displayed. They show a main development menu with submenus for various game systems such as rendering, mesh options, and more, allowing developers to configure settings while playing.

:x??

---

#### Summary of In-Game Menus and Debug Draw
Background context: The text covers the integration of debug draw features and in-game menus within game engines. It explains how these tools help developers quickly visualize and adjust various aspects of gameplay.

:p How do debug draw functions and in-game menus enhance game development?
??x
Debug draw functions and in-game menus significantly enhance game development by allowing real-time visualization of elements like vectors, text, and other visual primitives directly within the game. In-game menus provide an intuitive way to configure these visuals and settings without stopping gameplay or modifying code.

:x??

---


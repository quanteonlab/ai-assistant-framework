# Flashcards: Game-Engine-Architecture_processed (Part 54)

**Starting Chapter:** 10 Tools for Debugging and Development. 10.1 Logging and Tracing

---

#### Logging and Tracing
Background context explaining the use of logging for debugging purposes, especially in game development. It mentions how printf debugging is still a valid method even though debuggers are available.

:p What is printf debugging?
??x
Printf debugging involves using print statements to output the internal state of your program during runtime. This method helps identify issues by displaying relevant data at specific points in the code execution, especially useful for real-time programming scenarios where timing-dependent bugs occur.
??x

---

#### Debugging with OutputDebugString()
Background context on how `OutputDebugString()` function works in Windows SDK and its limitations.

:p How does `OutputDebugString()` work?
??x
`OutputDebugString()` is a Windows API function used to send debugging information to the Visual Studio's Debug Output window. It can only print raw strings and doesn't support formatted output, unlike `printf()`. Therefore, it often requires wrapping with custom functions that handle formatting.
??x

---

#### Custom Formatted Output Function
Explanation on creating a custom function in C for formatted output using `OutputDebugString()`.

:p What is the purpose of the `VDebugPrintF` and `DebugPrintF` functions?
??x
These functions are created to provide formatted output similar to `printf()`, but still send the data through `OutputDebugString()`. `VDebugPrintF` takes a `va_list` argument, while `DebugPrintF` uses variable-length arguments. This design allows for easier expansion and reuse of formatting logic.
??x

---

#### Code Example: Custom Formatted Output Functions
Code example demonstrating how to implement the custom formatted output functions.

:p Provide code for implementing `VDebugPrintF` and `DebugPrintF`.
??x
```c
#include <stdio.h> // for va_list et al
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h> // for OutputDebugString()

int VDebugPrintF(const char* format, va_list argList) {
    const U32 MAX_CHARS = 1024;
    static char s_buffer[MAX_CHARS];
    int charsWritten = vsnprintf(s_buffer, MAX_CHARS, format, argList);
    OutputDebugString(s_buffer); // Send the formatted string to Visual Studio's Debug Output window
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
??x

---

#### Differentiating Game Engine Logging
Explanation on why most game engines provide advanced logging facilities beyond simple `printf()` statements.

:p Why do game engines often go beyond basic `printf()` for logging?
??x
Game engines typically offer more sophisticated logging mechanisms that are better suited for the complex and real-time nature of game development. These tools help in tracing bugs, monitoring performance, and managing large-scale data output efficiently.
??x

---

#### Console and TTY Output Devices
Explanation on using console or teletype (TTY) output devices for debugging purposes.

:p What are some examples of console or TTY output devices used in game development?
??x
Examples include:
- In a C/C++ application running under Linux or Win32, `stdout` or `stderr` can be printed via `printf()` or `fprintf()`.
- For windowed applications under Win32, the Visual Studio debugger provides a debug console accessible through `OutputDebugString()`.
- On PlayStation 3 and 4, the Target Manager runs on your PC and allows printing messages to TTY output windows.
??x

---

#### Verbosity Control
Verbosity control allows developers to manage the level of detail printed by debug statements. This is particularly useful for troubleshooting issues that may reoccur, as it enables printing more detailed information when needed.

Background context: Typically, a global integer variable `g_verbosity` stores the current verbosity level. A function `VerboseDebugPrintF()` is provided, which checks if the message should be printed based on this verbosity level.
:p How does the `VerboseDebugPrintF()` function control verbosity?
??x
The `VerboseDebugPrintF()` function ensures that only messages with a verbosity level equal to or higher than the current setting are printed. Here's how it works:

```c
int g_verbosity = 0; // Global variable to store the verbosity level

void VerboseDebugPrintF(int verbosity, const char* format, ...) {
    va_list argList;
    va_start(argList, format); // Initialize variable argument list

    if (g_verbosity >= verbosity) { // Check if message should be printed
        VDebugPrintF(format, argList); // Print the formatted string
    }

    va_end(argList); // Clean up after using the variable arguments
}
```

x??

---

#### Channels in Debugging
Channels allow developers to categorize their debug output for easier filtering and analysis. Each channel can contain messages related to a specific part of the system, such as animation or physics.

Background context: Different platforms support varying numbers of channels (e.g., 14 on PlayStation 3). Some systems like Windows might have only one console, but even in such cases, it's beneficial to divide output into distinct channels for better organization.
:p How do you implement channels in a debugging system?
??x
Channels can be implemented by adding an additional channel argument to the debug printing function. Here’s how:

```c
enum Channels { ANIMATION_CHANNEL, PHYSICS_CHANNEL, AI_CHANNEL, ... };

void VerboseDebugPrintF(int verbosity, Channel channel, const char* format, ...) {
    va_list argList;
    va_start(argList, format);

    if (g_verbosity >= verbosity && active_channels & (1 << channel)) {
        VDebugPrintF(format, argList);
    }

    va_end(argList);
}
```

Here, `active_channels` is a bitmask where each bit represents the status of a channel. If a particular bit is set to 1, the corresponding channel is considered active.

x??

---

#### Logging and Tracing
Logging and tracing involve storing all debug output in log files for post-mortem analysis. This ensures that even if issues are not caught during development, they can still be diagnosed later by examining logs.

Background context: Log files should contain every single debug message, independent of the verbosity level or active channels mask. Flushing buffers after each call to ensure recent data is captured.
:p How do you implement logging in a debugging system?
??x
Implementing logging involves mirroring all debug output into one or more log files. Here’s an example approach:

```c
void DebugPrintToFile(const char* channel, const char* format, ...) {
    va_list argList;
    va_start(argList, format);

    // Open file for writing (or append)
    FILE* fp = fopen("log.txt", "a");

    if (fp != NULL) {
        vfprintf(fp, format, argList); // Print formatted string to file
        fclose(fp);
    }

    va_end(argList);
}
```

This function opens a log file and appends debug messages. Ensure that the buffer is flushed after each write to avoid losing data in case of crashes.

x??

---

#### Mirroring Output to File
Mirroring output to a file ensures that all debug information is recorded, independent of verbosity settings or active channels. This allows for post-mortem analysis of issues even if they were not visible during development.

Background context: Flushing the log file after each call can be expensive but is necessary to capture the latest data before a crash.
:p How do you ensure that the last output buffer full of debug messages is captured in a log file?
??x
To ensure that the most recent debug messages are included in the log, flush the file buffer after each write. Here’s an example:

```c
void DebugPrintToFile(const char* channel, const char* format, ...) {
    va_list argList;
    va_start(argList, format);

    // Open file for writing (or append)
    FILE* fp = fopen("log.txt", "a");

    if (fp != NULL) {
        vfprintf(fp, format, argList); // Print formatted string to file
        fflush(fp); // Flush buffer immediately
        fclose(fp);
    }

    va_end(argList);
}
```

Flushing the buffer ensures that data is written to disk before closing the file. This helps prevent data loss in case of crashes.

x??

---

---
#### Crash Report Information
Crash reports can provide valuable insights into game crashes by including various pieces of information. This is particularly useful for identifying issues, such as memory leaks or script errors.

:p What are some common pieces of information that should be included in a crash report?
??x
Common pieces of information include:
- Current level(s) being played.
- Player's world-space location when the crash occurred.
- Player’s animation/action state at the time of the crash.
- Gameplay scripts running during the crash.
- Stack trace and memory allocation details.

For example, you might log something like this in a crash report:

```plaintext
Crash Report:
Level: Level1
Player Location: (10.5, 234.7, -89.3)
Animation State: Idle
Running Scripts: MovementScript, EnemyAI
Stack Trace:
[1] Game::Update()
[2] Game::Tick()
[3] Memory::Alloc()
```
x??

---
#### Debug Drawing Facilities
Debug drawing facilities allow developers to visualize mathematical calculations and logic errors in real-time during game development. This can significantly speed up the debugging process.

:p What is a debug drawing facility, and why is it useful?
??x
A debug drawing facility is an API that allows developers to draw colored lines, simple shapes, or 3D text within the game environment for visualization purposes. It helps identify logical and mathematical errors quickly by providing visual feedback.

Example usage in C++:
```cpp
void DebugDrawLine(Vector2 start, Vector2 end) {
    // Logic to draw a line between two points.
}
```

Using this facility can greatly improve debugging efficiency. For instance, instead of deciphering debug logs, you can simply visualize the trajectory of projectiles or paths to identify issues.

For example:
```cpp
DebugDrawLine(playerPosition, targetPosition);
```
x??

---

#### Visualizing Player Perception for NPCs
Background context: In game development, visualizing how non-player characters (NPCs) perceive the player is crucial for debugging and ensuring AI behaviors are as intended. This can be achieved by showing the location of the player relative to the NPC's perception.

:p How does an enemy NPC visualize the player in terms of line of sight?
??x
The NPC perceives the player based on their last known position when the line of sight is broken. The "stick man" figure represents this last known position, even if the player has moved away.
```java
// Pseudocode for visualizing player perception
if (!player.inLineOfSight(npc)) {
    npc.playerLocation = player.lastKnownPosition;
}
```
x??

---

#### Visualizing Explosions in Game Development
Background context: Understanding how explosions interact with game objects and players is essential. This can be facilitated by using a wireframe sphere to represent the expanding blast radius.

:p How does the wireframe sphere help in visualizing explosion effects?
??x
The wireframe sphere dynamically expands to show the area affected by an explosion, allowing developers to see where damage will occur.
```java
// Pseudocode for updating the explosion sphere
void updateExplosionSphere(float timeSinceLastExplosion) {
    // Calculate the radius based on time since the last explosion
    float radius = initialRadius + (timeSinceLastExplosion * expansionRate);
    // Update the sphere to reflect the new radius
}
```
x??

---

#### Visualizing Drakewhile Hanging from Ledges
Background context: To ensure smooth gameplay, developers need tools to visualize how characters interact with environment elements like ledges. This helps in debugging and improving animations.

:p How are circles used to visualize Drakewhile searching for ledges?
??x
Circles represent the radii used by Drake when searching for ledges. A line shows the ledge he is currently hanging from, helping developers understand his movement paths.
```java
// Pseudocode for visualizing ledge hang radius
void drawLedgeHangRadius(Drake drake) {
    float radius = drake.getLedgeSearchRadius();
    // Draw circles representing the search area
    drawCircle(drake.position.x, drake.position.y + radius, radius);
}
```
x??

---

#### Debugging Mode for AI Characters
Background context: Developers often need full control over NPC behavior to test and debug. This can be achieved by placing characters in a special debugging mode where their actions can be controlled via a heads-up display.

:p How does the debugging mode allow developers to control NPC movements?
??x
In debugging mode, NPCs can be manually directed to walk, run, or sprint to specified points. Developers can also instruct them to enter cover, fire weapons, and more.
```java
// Pseudocode for controlling an NPC in debug mode
void controlNPCDebugMode(NPC npc) {
    if (developerCommand == "walk") {
        npc.walkTo(targetPoint);
    } else if (developerCommand == "fireWeapon") {
        npc.fireWeaponAt(targetEnemy);
    }
}
```
x??

---

#### Debug Drawing API Requirements
Background context: A debug drawing API is essential for developers to visualize game elements and behaviors. It should be simple, flexible, and support various types of primitives.

:p What are the key requirements for a debug drawing API?
??x
A debug drawing API must:
- Be simple and easy to use.
- Support lines, spheres, points, coordinate axes, bounding boxes, and text.
- Allow customization of properties like color, line width, sphere radius, etc.
- Draw primitives in world space or screen space with flexibility.

Example: 
```java
// Pseudocode for a debug drawing function
void drawLine(Vector3 start, Vector3 end) {
    // Logic to draw the line between two points
}
```
x??

---

#### Drawing Primitives with Depth Testing
Background context: In rendering engines, depth testing is a technique used to ensure that only the closest objects are visible. When drawing debug primitives, it's useful to have control over whether or not depth testing is enabled so you can visualize them according to your needs.

:p How does enabling and disabling depth testing affect the visibility of debug primitives?
??x
Enabling depth testing means that the debug primitives will be occluded by real objects in the scene. This makes their depth easy to visualize, but it also means they may sometimes be difficult to see or totally hidden by the geometry. Disabling depth testing causes the primitives to "hover" over the real objects, making it harder to gauge their actual depth but ensuring that no primitive is ever hidden from view.

```cpp
void AddLine(const Point& fromPosition, const Point& toPosition, Color color, float lineWidth = 1.0f, float duration = 0.0f, bool depthEnabled = true);
```
In this function, the `depthEnabled` parameter determines whether or not depth testing is applied.

x??

---

#### Debug Drawing API Flexibility
Background context: The debug drawing system should be flexible and allow for a variety of primitive types to be drawn at any point in the code. This flexibility helps developers easily visualize different elements within their scene without strict limitations on when they can call the rendering functions.

:p How does the `DebugDrawManager` class provide flexibility in drawing primitives?
??x
The `DebugDrawManager` class provides a variety of methods to add different types of debug primitives, such as lines, crosses, spheres, circles, axes, triangles, axis-aligned bounding boxes (AABBs), oriented bounding boxes (OBBs), and text strings. Each method allows customization through parameters like color, size, duration, and depth testing.

```cpp
class DebugDrawManager {
public:
    void AddLine(const Point& fromPosition, const Point& toPosition, Color color, float lineWidth = 1.0f, float duration = 0.0f, bool depthEnabled = true);
    // Other methods for adding different primitives...
};
```
This design ensures that the debug drawing API can be used in any part of the code and provides options like `depthEnabled` to control how these primitives interact with other objects in the scene.

x??

---

#### Primitive Lifetimes
Background context: The concept of a primitive lifetime allows developers to set how long a debug primitive should remain on-screen. This is particularly useful for debugging information that needs to persist across multiple frames without flickering or disappearing too quickly.

:p What is the purpose of setting a duration when adding a debug primitive?
??x
Setting a duration when adding a debug primitive gives control over its visibility. If the code drawing the primitive is called every frame, the default lifetime can be one frame, ensuring the primitive remains on screen as it gets refreshed each frame. However, for less frequent calls (e.g., calculating initial velocities), setting a longer lifetime ensures that the primitive stays visible until it's no longer needed.

```cpp
void AddLine(const Point& fromPosition, const Point& toPosition, Color color, float lineWidth = 1.0f, float duration = 0.0f, bool depthEnabled = true);
```
In this function, setting a non-zero `duration` keeps the primitive visible for that number of frames.

x??

---

#### Handling Multiple Primitives Efficiently
Background context: When dealing with many game objects and their associated debug primitives, it's crucial to manage performance efficiently. The system must be able to handle a large number of primitives without significantly impacting overall game performance when debug drawing is enabled.

:p How does the `DebugDrawManager` ensure efficient handling of multiple debug primitives?
??x
The `DebugDrawManager` ensures efficiency by queuing up all incoming debug drawing requests and submitting them during an appropriate phase of the game loop, usually at the end of each frame. This approach allows for batch processing of multiple primitives in a single call, reducing overhead.

```cpp
class DebugDrawManager {
public:
    void AddLine(const Point& fromPosition, const Point& toPosition, Color color, float lineWidth = 1.0f, float duration = 0.0f, bool depthEnabled = true);
    // Other methods for adding different primitives...
};
```
Internally, the manager collects all added primitives into a queue and processes them together at an appropriate time.

x??

---

#### Debug Drawing Manager Usage
Background context: The text describes how to use a debug drawing manager (g_debugDrawMgr2D) within game code. This function is used for visualizing various aspects of the game, such as velocity vectors and textual information about entities.

:p How does the `AddLine` and `AddString` functions in `DebugDrawManager` work?
??x
The `AddLine` and `AddString` functions allow adding visual elements to a list that will be drawn at a later time. This approach is used for efficiency, especially in high-speed 3D rendering engines.

```cpp
void Vehicle::Update() {
    // Do some calculations...
    
    // Debug-draw my velocity vector.
    const Point& start = GetWorldSpacePosition();
    Point end = start + GetVelocity();
    g_debugDrawMgr.AddLine(start, end, kColorRed);
    
    // Do some other calculations...
    
    // Debug-draw my name and number of passengers.
    {
        char buffer[128];
        sprintf(buffer, "Vehicle  percents:  percentd passengers", GetName(), GetNumPassengers());
        const Point& pos = GetWorldSpacePosition();
        g_debugDrawMgr.AddString(pos, buffer, kColorWhite, 0.0f, false);
    }
}
```
x??

---

#### In-Game Menus
Background context: The text explains the purpose and functionality of in-game menus in game development. These menus allow programmers, artists, and designers to configure various settings while the game is running without needing to recompile or relink the executable.

:p What are the primary purposes of in-game menus?
??x
In-game menus serve multiple purposes:
- Toggling global Boolean settings.
- Adjusting global integer and floating-point values.
- Calling arbitrary functions, which can perform any task within the engine.
- Bringing up submenus, allowing for hierarchical organization to simplify navigation.

These features help reduce debugging time and facilitate quick adjustments during game development.
x??

---

#### Menu System in Naughty Dog Engine
Background context: The text provides an example of how the menu system works in the Naughty Dog engine. It includes top-level menus with submenus for different subsystems, such as rendering and physics.

:p How does bringing up the in-game menu typically affect gameplay?
??x
Bringing up the in-game menu usually pauses the game. This allows developers to stop the game at the precise moment a problem occurs, adjust engine settings, and then resume play to inspect the issue more thoroughly.
x??

---

#### Submenus in In-Game Menus
Background context: The text mentions that submenus can be used for more detailed configuration within the main menu structure.

:p What are some things an item on an in-game menu might do?
??x
An item on an in-game menu can perform various tasks, including:
- Toggling global Boolean settings.
- Adjusting global integer and floating-point values.
- Calling arbitrary functions to execute specific tasks.
- Bringing up submenus for further configuration.

These capabilities allow for flexible and dynamic game development without the need for recompilation.
x??

---

#### Example of In-Game Menu Usage
Background context: The text provides an example from "The Last of Us: Remastered" showing how top-level menus and submenus are used in practice.

:p How can developers effectively use in-game menus during debugging?
??x
Developers can use in-game menus to pause the game, adjust settings, visualize issues more clearly, and then resume gameplay to inspect problems. This approach minimizes disruptions to normal play while facilitating efficient problem-solving.
x??

---

#### Visualizing Game Elements with Debug Draw Manager
Background context: The text explains how a debug draw manager is used for visualizing game elements like velocity vectors and textual information.

:p How does the debug draw manager manage the drawing of visual elements?
??x
The debug draw manager adds visual elements to a list that will be drawn later. This method ensures efficient rendering, especially in high-speed 3D environments where all visual elements need to be managed through a scene data structure.
x??

---

#### Menu Navigation and Organization
Background context: The text describes the hierarchical organization of menus for easy navigation.

:p How are submenus typically organized within an in-game menu system?
??x
Submenus in an in-game menu system can be organized hierarchically, allowing for clear and intuitive navigation. This structure enables developers to quickly access specific settings or configurations related to different game subsystems.
x??

---


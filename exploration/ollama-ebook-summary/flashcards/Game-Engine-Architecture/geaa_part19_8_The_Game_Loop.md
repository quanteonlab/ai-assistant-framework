# Flashcards: Game-Engine-Architecture_processed (Part 19)

**Starting Chapter:** 8 The Game Loop and Real-Time Simulation. 8.2 The Game Loop

---

#### Rendering Loop Overview
Real-time 3D computer graphics require a different approach compared to static GUI rendering. Instead of rectangle invalidation, real-time graphics present a series of still images rapidly on-screen, simulating motion and interactivity through a loop.

The simplest form of a render loop in a real-time application is structured as follows:
```cpp
while (!quit) {
    // Update the camera transform based on interactive inputs or predefined path.
    updateCamera();
    
    // Update positions, orientations, and relevant visual state of any dynamic elements in the scene.
    updateSceneElements();
    
    // Render a still frame into an off-screen framebuffer known as the "back buffer".
    renderScene();
    
    // Swap the back buffer with the front buffer to make the most recently rendered image visible on-screen. 
    // Or, in windowed mode, copy (blit) the back buffer's contents to the front buffer.
    swapBuffers();
}
```
:p How is a real-time rendering loop structured for 3D graphics?
??x
The render loop updates camera transformations and scene elements before rendering a frame into an off-screen framebuffer called the "back buffer". After rendering, it swaps or copies this back buffer with the on-screen front buffer. This process creates a continuous flow of images that simulate motion.
```cpp
// Example of updating the camera in the render loop
void updateCamera() {
    // Update based on user input or predefined path
}
```
x??

---

#### Game Loop Overview
Games are composed of many interacting subsystems, each requiring periodic servicing at different rates. The game loop is the master loop that services all these systems.

The simplest implementation of a game loop might look like this:
```cpp
void main() {
    initGame();
    
    while (true) { // Game loop
        readHumanInterfaceDevices();
        
        if (quitButtonPressed()) {
            break; // Exit the game loop
        }
        
        movePaddles(); 
        moveBall();
        collideAndBounceBall();
        
        if (ballImpactedSide(LEFT_PLAYER)) {
            incrementScore(RIGHT_PLAYER);
            resetBall();
        } else if (ballImpactedSide(RIGHT_PLAYER)) {
            incrementScore(LEFT_PLAYER);
            resetBall();
        }
        
        renderPlayfield();
    }
}
```
:p What is the basic structure of a game loop?
??x
The basic structure involves initializing the game, entering an infinite loop where it reads input devices, performs necessary updates (like moving paddles and ball), checks collisions, increments scores, and renders the playfield. This ensures all subsystems are serviced at appropriate intervals.
```cpp
// Example of updating human interface devices in a game loop
void readHumanInterfaceDevices() {
    // Read from control wheels, joysticks or other input devices
}
```
x??

---

#### Pong Game Loop Details
Pong is an early example illustrating the core logic of a game loop. The loop continuously reads player inputs, updates game state (paddle and ball positions), checks for collisions, updates scores, and renders the scene.

Here’s how the core logic might look:
```cpp
void main() {
    initGame();
    
    while (true) { // Game loop
        readHumanInterfaceDevices();
        
        if (quitButtonPressed()) {
            break;
        }
        
        movePaddles(); 
        moveBall();
        collideAndBounceBall();
        
        if (ballImpactedSide(LEFT_PLAYER)) {
            incrementScore(RIGHT_PLAYER);
            resetBall();
        } else if (ballImpactedSide(RIGHT_PLAYER)) {
            incrementScore(LEFT_PLAYER);
            resetBall();
        }
        
        renderPlayfield();
    }
}
```
:p What does the core logic of a Pong game loop involve?
??x
The core logic involves reading player inputs, updating paddle and ball positions, checking for collisions, incrementing scores when necessary, and rendering the playfield. This ensures continuous interaction and display of the game state.
```cpp
// Example of moving paddles in a game loop
void movePaddles() {
    // Adjust paddle positions based on current input from control wheels or joysticks
}
```
x??

---


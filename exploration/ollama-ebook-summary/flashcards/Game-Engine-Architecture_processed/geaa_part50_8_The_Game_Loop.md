# Flashcards: Game-Engine-Architecture_processed (Part 50)

**Starting Chapter:** 8 The Game Loop and Real-Time Simulation. 8.2 The Game Loop

---

---
#### Rendering Loop Concept
Background context: In real-time 3D computer graphics, the screen's contents change continually as the camera moves through a 3D scene. The traditional technique of rectangle invalidation used for static GUIs no longer applies here because every part of the screen changes frequently. Instead, a rapid sequence of still images is presented to simulate motion and interactivity.

:p What does the render loop in real-time 3D graphics look like?
??x
The render loop is structured as follows:

```cpp
while (.quit) {
    // Update the camera transform based on interactive inputs or predefined paths.
    updateCamera();

    // Update positions, orientations, and other relevant visual states of dynamic elements in the scene.
    updateSceneElements();

    // Render a still frame into an off-screen framebuffer known as the "back buffer".
    renderScene();

    // Swap the back buffer with the front buffer, making the most recently rendered image visible on-screen. 
    swapBuffers();
}
```

x??
This pseudocode demonstrates the basic structure of a real-time rendering loop in 3D graphics. The camera transform is updated first based on interactive inputs or predefined paths to reflect the movement of the camera. Next, positions and orientations of dynamic elements are updated. Then, a still frame (rendered image) is drawn into an off-screen framebuffer known as the "back buffer". Finally, this back buffer is swapped with the front buffer (which displays on the screen) to show the newly rendered image.

---
#### Game Loop Concept
Background context: A game's engine consists of many interacting subsystems such as device I/O, rendering, animation, collision detection and resolution, rigid body dynamics simulation, multiplayer networking, audio, etc. These subsystems require periodic servicing but at varying rates. The game loop is a master loop that services every subsystem in the game.

:p What does the game loop look like in pseudocode?
??x
The simplest form of a game loop can be structured as:

```java
void main() {
    initGame();
    while (true) {
        readHumanInterfaceDevices();
        
        if (quitButtonPressed()) {
            break;
        }
        
        movePaddles(); // Adjust paddle positions based on control inputs.
        moveBall();     // Update ball position using its velocity vector.
        collideAndBounceBall(); // Handle collisions and bounces of the ball.

        if (ballImpactedSide(LEFT_PLAYER)) {
            incrementScore(RIGHT_PLAYER);
            resetBall();
        } else if (ballImpactedSide(RIGHT_PLAYER)) {
            incrementScore(LEFT_PLAYER);
            resetBall();
        }

        renderPlayfield(); // Draw the entire screen.
    }
}
```

x??
This pseudocode outlines a basic game loop. The `initGame()` function initializes the necessary systems like graphics, input devices, audio, etc. The loop runs indefinitely until the player presses the "quit" button, which causes an exit via a break statement. Inside the loop, human interface devices are read to get current inputs. Paddle and ball positions are updated based on these inputs, collision checks and bounces are handled, scores are incremented when necessary, and finally, the playfield is rendered.

---
#### Pong Game Loop Concept
Background context: The classic game of Pong involves a ball bouncing between two paddles controlled by human players. The game loop in such a simple game can be quite straightforward.

:p What does the core structure of a pong game loop look like?
??x
The core structure of a pong game loop might look something like this:

```java
void main() {
    initGame();
    
    while (true) { // Infinite game loop.
        readHumanInterfaceDevices(); // Read player inputs from control wheels or other devices.

        if (quitButtonPressed()) {
            break; // Exit the game loop when "quit" button is pressed.
        }

        movePaddles(); // Adjust paddle positions based on control wheel deflection.
        moveBall();     // Update ball position using its velocity vector.
        collideAndBounceBall(); // Handle collisions and bounces of the ball.

        if (ballImpactedSide(LEFT_PLAYER)) {
            incrementScore(RIGHT_PLAYER);
            resetBall();
        } else if (ballImpactedSide(RIGHT_PLAYER)) {
            incrementScore(LEFT_PLAYER);
            resetBall();
        }

        renderPlayfield(); // Draw the entire screen.
    }
}
```

x??
This pseudocode demonstrates a simplified game loop for Pong. The main function initializes the game and then enters an infinite loop until the player presses the "quit" button, causing a break in the loop. Inside the loop, inputs from human interface devices (like control wheels) are read to adjust paddle positions. Ball position is updated based on its velocity vector, and collision checks and bounces are handled. Scores are incremented when necessary, and finally, the playfield is rendered.

---
#### Difference Between Rendering Loop and Game Loop
Background context: The rendering loop and game loop serve different purposes but both are crucial for real-time simulations in games. The rendering loop focuses on drawing images onto the screen at a high frame rate to simulate motion, while the game loop manages all subsystems of the game, ensuring they operate as required.

:p How do the rendering loop and game loop differ?
??x
The rendering loop is focused specifically on rendering images onto the screen at a desired frame rate. It updates the camera position, object states, and renders new frames into an off-screen buffer that gets swapped with the front buffer to display the image. 

On the other hand, the game loop manages all aspects of the game, including device I/O (like player inputs), updating scene elements (such as paddles and ball positions in Pong), collision detection, scoring, and rendering.

```java
// Rendering Loop Example
while (.quit) {
    updateCamera();
    updateSceneElements();
    renderScene();
    swapBuffers();
}

// Game Loop Example
void main() {
    initGame();
    
    while (true) { 
        readHumanInterfaceDevices();
        
        if (quitButtonPressed()) {
            break;
        }

        movePaddles();
        moveBall();
        collideAndBounceBall();

        incrementScore(leftOrRightPlayer);
        resetBall();

        renderPlayfield(); 
    }
}
```

x??
The rendering loop and game loop differ in their focus. The rendering loop updates the camera and scene elements but does not handle higher-level game logic like scoring or player inputs. It is primarily concerned with drawing images onto the screen at a high frame rate to simulate motion.

In contrast, the game loop manages all aspects of the game. It reads human interface devices for input, updates positions and states of dynamic objects (like paddles in Pong), handles collisions and bounces, increments scores based on ball impacts, and renders the entire playfield. This ensures that the game logic runs smoothly alongside the visual representation.

---


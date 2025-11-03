# Flashcards: Game-Engine-Architecture_processed (Part 74)

**Starting Chapter:** 16.6 Updating Game Objects in Real Time

---

#### Specialized Data Structures for Game Object Queries
Background context: In game development, efficient data structures are crucial to handle various types of queries on game objects. This is especially important given the dynamic nature of games and the need for real-time performance.

The text describes several specialized data structures that can be used to accelerate specific types of queries. For example:
- **Finding Game Objects by Unique ID**: Using hash tables or binary search trees.
- **Iterating Over All Objects That Meet a Particular Criterion**: Sorting objects into linked lists based on various criteria (e.g., object type, position).
- **Collision Detection and Line of Sight Queries**: Leveraging collision systems to perform fast ray casts and other shape intersections.

:p What is the purpose of using specialized data structures for game queries?
??x
To optimize performance by accelerating specific types of queries that are frequently needed during development. These optimizations help maintain real-time responsiveness, which is crucial in games.
x??

---

#### Updating Game Objects in Real Time
Background context: Every game engine needs to update the internal state of every game object over time. The state of a game object can be thought of as its configuration at one specific instant in time.

The text explains that this process involves determining the current state \( S_i(t) \) given the previous state \( S_i(t - \Delta t) \). This updating is typically done via a single master loop called the game loop, which runs continuously throughout the game's runtime.

:p What does it mean when we say "game objects' states change discretely rather than continuously"?
??x
It means that while in reality the state of an object might change smoothly over time, in the context of game development, these changes are sampled at discrete points. This approach simplifies calculations and ensures consistent performance.
x??

---

#### Game Object State Representation
Background context: A game object's state can be described as a collection of attribute values (properties or data members). For example, in Pong, the ball's position and velocity define its state.

The text uses vector notation to represent this concept, indicating that each state is like an n-dimensional vector containing various types of information. However, this representation helps in understanding the heterogeneity of game object states.

:p How is a game object's state typically represented?
??x
A game object’s state can be thought of as an \( n \)-dimensional vector where each dimension corresponds to one of the attributes (properties or data members) of the game object. For instance, in Pong, the state might include position coordinates and velocity components.

```java
class GameObj {
    int x, y; // Position
    float dx, dy; // Velocity

    public Vector2 getState() {
        return new Vector2(x, y); // Example of a 2D vector for position
    }
}
```
x??

---

#### Game Loop and Object Updating
Background context: Most game engines use a single master loop called the game loop to update the state of all game objects. This process involves determining the current state based on the previous state.

The text emphasizes that updating is done periodically, often within this main game loop, treating each subsystem (like rendering or physics) as requiring periodic servicing.

:p What role does the game loop play in managing game object states?
??x
The game loop acts as a central mechanism for determining and updating the current state of all game objects. It ensures that the engine continuously checks and updates these states based on their previous values, maintaining the dynamic nature of the game world.

```java
public class GameLoop {
    public void run() {
        while (gameRunning) {
            updateAllObjects();
            renderScene();
            handleInput();
        }
    }

    private void updateAllObjects() {
        for (GameObj obj : objects) {
            obj.updateState(getPreviousState(obj)); // Example method call
        }
    }
}
```
x??

---

#### Discrete vs Continuous Time in Games
Background context: The text explains that while the state of a game object is described as changing discretely, it’s often useful to think of these changes as continuous for practical reasons. This approach helps developers avoid common pitfalls.

:p Why might we prefer to think about game objects’ states as changing continuously?
??x
Thinking of game object states as changing continuously can help in designing more realistic and smooth animations and interactions within the game environment, even though the actual updates occur discretely. This abstraction simplifies development by allowing us to focus on the flow of events rather than their precise timing.

```java
// Pseudocode for a continuous state transition
public void updateState(float deltaTime) {
    x += dx * deltaTime; // Continuous position update
    y += dy * deltaTime; // Continuous velocity-based update
}
```
x??

---

#### Game Object Updating Systems
Background context explaining that game object updating systems are a dynamic, real-time agent-based computer simulation. They involve updating game objects over time, which can be challenging due to the need for precise timing and synchronization with real-world events.

Game object updating is crucial in games as it affects how game elements behave based on time passing. Understanding this system helps developers create realistic and engaging gameplay experiences.
:p What is a game object updating system?
??x
A game object updating system is an example of a dynamic, real-time agent-based computer simulation used to manage the state transitions of game objects over time. This process involves iterating through each game object within the engine's main loop and calling an `Update()` function that performs necessary tasks to advance the state of the object.
??x
---

#### Time Deltas in Game Object Updating
Explanation about why using a time delta is important for precise timing in game object updates.

Using a time delta (`dt`) allows objects to adjust their behavior based on the actual elapsed time since the last frame, rather than assuming fixed-time steps. This ensures that physics calculations and animations are accurate and smooth.
:p Why do we use time deltas when updating game objects?
??x
We use time deltas because they allow each object's update function to account for the exact amount of time that has passed since the last frame. This is crucial for maintaining accuracy in physics calculations, animations, and other timing-dependent actions.
??x
---

#### Monolithic Object Hierarchy
Explanation about a monolithic object hierarchy where each game object is represented by a single class instance.

In a monolithic object hierarchy, all game objects are instances of a single class. This simplifies the engine design but may limit flexibility in some cases.
:p What does "monolithic object hierarchy" mean?
??x
A monolithic object hierarchy means that every game object is an instance of a single class. While this approach simplifies the engine's structure, it can be limiting if different types of objects require different behaviors or properties.
??x
---

#### Update Function Signature
Explanation about the typical signature of the `Update()` function.

The `Update()` function typically takes a parameter representing the time delta (`dt`) and is called once per frame during each iteration of the main game loop. This allows objects to update their state based on real-time conditions.
:p What does the `Update(float dt)` function do?
??x
The `Update(float dt)` function updates the state of a game object based on the time delta (`dt`) since the last frame. This ensures that all game objects can adjust their states accurately in response to real-world timing.
??x
---

#### Common Problems and Solutions
Explanation about recurring issues faced by game developers when implementing game object updating systems.

Game developers often encounter problems such as synchronization issues, performance bottlenecks, and state inconsistencies. Addressing these requires careful design patterns and optimization techniques.
:p What are some common problems in game object updating?
??x
Common problems include:
- Synchronization issues where objects don't update correctly due to timing discrepancies.
- Performance bottlenecks caused by poorly optimized `Update()` functions.
- State inconsistencies when multiple systems or objects interact improperly.

To solve these, developers implement robust design patterns and optimize performance through techniques like fixed time steps and efficient state management.
??x
---

#### Maintaining a Collection of Active Game Objects
In object-oriented game development, managing the collection of active game objects is crucial for efficient gameplay. Typically, this collection is maintained by a singleton manager class such as `GameWorld` or `GameObjectManager`. The collection must be dynamic to handle spawning and destruction of game objects during gameplay.

This often involves using data structures like linked lists, smart pointers, or handles. However, some engines may use statically sized arrays for fixed-sized object management.

:p How should the collection of active game objects typically be managed in a game?
??x
The collection is usually maintained by a singleton manager class such as `GameWorld` or `GameObjectManager`. This manager uses dynamic data structures like linked lists, smart pointers, or handles to efficiently manage the lifecycle of game objects. In some cases, engines might use statically sized arrays for fixed-size object management.
x??

---

#### Responsibilities of the Update() Function
The `Update()` function in a game object is responsible for determining its state at each discrete time index based on its previous state. This involves various tasks such as applying rigid body dynamics, sampling animations, and reacting to events.

While it might seem intuitive to update all engine subsystems directly from within the `Update()` function of a game object, this approach is generally not recommended in commercial-grade engines due to performance and maintainability issues.

:p What are some common responsibilities of an Update() function for a game object?
??x
The `Update()` function primarily handles tasks such as:
- Updating the state of the game object itself (e.g., moving a tank, deflecting its turret).
- Reacting to events that occurred during the current time step.
- Interacting with engine subsystems like rendering, audio, physics, and collision.

Here's an example of how not to update all subsystems from within `Update()`:
```cpp
virtual void Tank::Update(float dt) {
    // Update the state of the tank itself.
    MoveTank(dt);
    DeflectTurret(dt);
    FireIfNecessary();

    // This is generally NOT a good idea...
    m_pAnimationComponent->Update(dt);  // Updates animation component
    m_pCollisionComponent->Update(dt);  // Updates collision component
    m_pPhysicsComponent->Update(dt);    // Updates physics component
    m_pAudioComponent->Update(dt);      // Updates audio component
    m_pRenderingComponent->draw();     // Draws rendering component
}
```
x??

---

#### Driving the Game Loop with Object Updates
The game loop can be driven by updating game objects, but this approach is generally not recommended in commercial-grade engines due to performance and maintainability issues.

Typically, a more complex data structure than a simple linked list is used. The game loop would poll input, calculate delta time, update each game object, and then swap the rendering buffers.

:p How could the game loop be structured using updates of game objects?
??x
A typical game loop might look like this:
```cpp
while (true) {
    PollJoypad();  // Polls for input events
    float dt = g_gameClock.CalculateDeltaTime();  // Calculates delta time

    for (each gameObject) {
        // Update each game object's state.
        gameObject.Update(dt);  // Updates the game object itself and its components
    }

    g_renderingEngine.SwapBuffers();  // Swaps rendering buffers to display updated frames
}
```
This structure is less common in commercial engines due to performance concerns. More complex systems manage state updates more efficiently.

x??

#### Performance Constraints and Batched Updates
In game development, engine systems often face stringent performance constraints due to the need for real-time updates on a large amount of data. To meet these requirements, batched updating is frequently used to optimize efficiency.

Batching involves grouping operations together to reduce overhead and improve performance. For instance, it's more efficient to update multiple animations in one go rather than interleaving them with other unrelated operations like collision detection or rendering.

Most commercial game engines handle updates through a main game loop that directly or indirectly calls the Update() functions of engine subsystems. Game objects typically request services from these subsystems when needed, such as requesting a mesh instance for rendering purposes.

The rendering engine maintains internal collections of these instances and manages them to optimize performance. Game objects control their appearance by manipulating properties like visibility but do not directly update the subsystems themselves.
:p How does batched updating work in game engines?
??x
Batched updating works by grouping similar operations together, reducing overhead and improving performance. For example, all visible mesh instances are drawn in one efficient batch instead of updating each animation individually during every frame.

The main game loop typically handles this process, calling the Update() functions for various subsystems like rendering, physics, or animations. Game objects interact with these subsystems through specific methods, such as requesting a mesh instance or controlling rendering properties without directly updating the subsystem.
```java
public class Tank {
    // Example of how a tank object might update itself in a game loop

    public void Update(float dt) {
        // Update internal state and behavior
        MoveTank(dt);
        DeflectTurret(dt);
        FireIfNecessary();

        // Control rendering properties without directly updating the subsystems
        if (justExploded) {
            m_pAnimationComponent->PlayAnimation("explode");
        }
        if (isVisible) {
            m_pCollisionComponent->Activate();
            m_pRenderingComponent->Show();
        } else {
            m_pCollisionComponent->Deactivate();
            m_pRenderingComponent->Hide();
        }
    }

    private void MoveTank(float dt) { /* Logic for moving the tank */ }
    private void DeflectTurret(float dt) { /* Logic for turrets deflecting */ }
    private void FireIfNecessary() { /* Logic to fire if conditions met */}
}
```
x??

---
#### Engine Subsystem Management
Game engines manage engine subsystems through a main game loop that updates them rather than updating each object's Update() function. This approach ensures efficient use of resources by batch processing operations.

For instance, a game object might request specific services from the rendering or animation systems to optimize performance. The game object controls how these services are used but does not directly update the subsystems themselves.
:p How do game engines manage engine subsystem updates?
??x
Game engines typically manage engine subsystem updates through a main game loop that calls Update() functions for each subsystem rather than directly within each game object's Update() function.

This approach allows for more efficient and coordinated updates by batch processing operations. For example, game objects can request specific services like rendering or animations from the relevant subsystems but do not directly control their internal states. Instead, they manipulate properties to influence how these systems operate.
```java
public class MainGameLoop {
    public void GameLoop() {
        while (true) {
            PollJoypad();
            float dt = g_gameClock.CalculateDeltaTime();

            // Update all game objects
            for (GameObject gameObject : GameObjects) {
                gameObject.Update(dt);
            }

            // Update specific subsystems in batches
            g_animationEngine.Update(dt);
            g_physicsEngine.Simulate(dt);
            g_collisionEngine.DetectAndResolveCollisions(dt);
            g_audioEngine.ProcessAudio(dt);
        }
    }
}
```
x??

---
#### Game Object Requesting Subsystem Services
Game objects can request services from engine subsystems when needed, such as rendering or animation. These requests are made through specific methods provided by the subsystem.

For instance, a game object might ask for a mesh instance to be created for rendering purposes, and it controls how this instance is used without directly updating the subsystem.
:p How do game objects request services from engine subsystems?
??x
Game objects can request services from engine subsystems using specific methods provided by those subsystems. For example, if a game object wants to use a triangle mesh for rendering, it might call a method like `m_pRenderingComponent->Show()` or `m_pCollisionComponent->Activate()`.

These methods allow the game object to influence how the subsystem operates without directly updating its internal state. The game object manipulates properties and behaviors through these requests but does not control the actual update process.
```cpp
// Example of requesting services from rendering component
if (isVisible) {
    m_pRenderingComponent->Show(); // This method might activate the render instance
} else {
    m_pRenderingComponent->Hide(); // This method might deactivate the render instance
}
```
x??

---

---
#### Batched Updating Benefits
Batched updating refers to a technique where game objects are updated as part of a batch rather than individually. This approach offers several performance benefits and is often necessary for correct game state updates.

:p What are the primary performance benefits of batched updating?
??x
The primary performance benefits include maximal cache coherency, minimal duplication of computations, reduced reallocation of resources, and efficient pipelining. These benefits arise because:
- **Maximal Cache Coherency**: Batched data can be arranged in a single, contiguous region of RAM, leading to better cache utilization.
- **Minimal Duplication of Computations**: Global calculations are done once and reused for multiple objects, reducing redundant computations.
- **Reduced Reallocation of Resources**: Resources like memory or other assets are allocated once per frame and reused across all objects in the batch, minimizing reallocations.
- **Efficient Pipelining**: The scatter/gather approach can be employed to divide large workloads across multiple CPU cores, enhancing parallelism.

For example:
```java
// Pseudocode for a batched update function
public void BatchUpdate(float deltaTime) {
    // Update all objects in one pass
    for (Object obj : objectBatch) {
        Update(obj, deltaTime);
    }
    g_renderingEngine.RenderFrameAndSwapBuffers();
}
```
x??

---
#### Interdependencies in Game Object Updates
In game development, not only performance but also the logical interdependence of game objects and engine subsystems necessitates a batched update approach. For instance, when updating a character holding a cat, the world-space pose of the cat's skeleton must be calculated after that of the human.

:p Why is the order in which objects are updated important?
??x
The order in which objects are updated is crucial because some game elements depend on others to function correctly. For example:
- **Character and Cat Example**: To calculate the world-space pose of a cat held by a human, the human’s world-space pose must be calculated first.
- **Animation and Physics Interdependency**: The animation system produces intermediate poses that are then applied in the physics simulation. This requires the physics to update before the animation can finalize its calculations.

```java
// Pseudocode for updating objects with interdependencies
public void UpdateGameObjects(float deltaTime) {
    // Update human first
    UpdateHuman(deltaTime);
    
    // Then update cat based on human's pose
    UpdateCatBasedOnHumanPose();
    
    // Physics and Animation Interdependency
    UpdatePhysicsSystem(deltaTime);  // Updates rigid bodies
    ApplyJointTransformsToSkeleton(); // Converts joint transforms to world space
    UpdateAnimationSystem();         // Calculates final world-space poses
}
```
x??

---
#### Collision Resolution in Dynamic Rigid Bodies
When dealing with dynamic rigid bodies, simple per-object collision detection and resolution may not suffice. Instead, interpenetrations between multiple objects must be resolved collectively.

:p Why can't collisions between dynamic rigid bodies be handled by processing each object individually?
??x
Collisions involving multiple dynamic rigid bodies cannot be effectively managed through individual processing because:
- **Interpenetration Issues**: Individual processing does not account for the combined effects of multiple bodies. Objects may interpenetrate, and resolving these issues requires a group approach.
- **Iterative or Linear System Solving**: Effective collision resolution often involves iterative methods or solving linear systems that consider the interactions between all involved objects.

For example:
```java
// Pseudocode for iterative collision resolution
public void ResolveCollisions(float deltaTime) {
    // Group of dynamic rigid bodies
    List<RigidBody> rigidBodies = GetAllDynamicRigidBodies();
    
    // Iteratively resolve collisions
    while (!AreAllObjectsNonPenetrating()) {
        for (int i = 0; i < rigidBodies.size(); ++i) {
            for (int j = i + 1; j < rigidBodies.size(); ++j) {
                if (rigidBodies[i].Intersects(rigidBodies[j])) {
                    // Apply collision response
                    rigidBodies[i].ResolveCollisionWith(rigidBodies[j]);
                    rigidBodies[j].ResolveCollisionWith(rigidBodies[i]);
                }
            }
        }
    }
}
```
x??

---

#### Phased Updates in Game Engines
Phased updates are a common technique used to manage inter-subsystem dependencies within game engines. By explicitly defining the order of subsystem updates, developers can ensure that data and state changes are applied correctly at the right time.

In this context, the main game loop often involves multiple stages where different subsystems perform their calculations or updates. This is particularly important when dealing with complex interactions between systems like animation, physics (ragdoll), and collision detection.

:p How does a phased update system work in managing inter-subsystem dependencies?
??x
A phased update system works by breaking down the main game loop into smaller phases, each corresponding to a specific subsystem or process. For instance, in an engine with an animation system and ragdoll physics, one phase might handle the calculation of intermediate poses, while another applies those poses to the ragdolls.

Here is an example of how this might look in pseudocode:
```pseudocode
while (true) // main game loop
{
    g_animationEngine.CalculateIntermediatePoses(dt);
    
    for each gameObject {
        gameObject.PreAnimUpdate(dt);
    }
    
    for each gameObject {
        gameObject.PostAnimUpdate(dt);
    }

    g_ragdollSystem.ApplySkeletonsToRagDolls();
    
    g_physicsEngine.Simulate(dt); // runs ragdolls too
    g_collisionEngine.DetectAndResolveCollisions(dt);

    g_ragdollSystem.ApplyRagDollsToSkeletons();

    g_animationEngine.FinalizePoseAndMatrixPalette();
    
    for each gameObject {
        gameObject.FinalUpdate(dt);
    }
}
```
x??

---

#### Multiple Update Phases in Game Objects
Game objects often require updates at multiple points during the frame, especially when they depend on calculations performed by various engine subsystems. This is particularly true for systems like animation blending and final pose generation.

Many game engines provide mechanisms to handle these requirements efficiently. For example, Naughty Dog's engine allows game objects to run update logic in three different phases: before animation blending, after animation blending but before final pose generation, and after final pose generation.

:p How does the Naughty Dog engine manage multiple update phases for game objects?
??x
The Naughty Dog engine manages multiple update phases by providing each game object class with virtual functions that act as "hooks." These hooks are called at specific points in the main game loop, allowing game objects to run their logic during different stages of the frame.

Here is an example of how this might be implemented:
```pseudocode
while (true) // main game loop
{
    for each gameObject {
        gameObject.PreAnimUpdate(dt);
    }
    
    g_animationEngine.CalculateIntermediatePoses(dt);
    
    for each gameObject {
        gameObject.PostAnimUpdate(dt);
    }

    g_ragdollSystem.ApplySkeletonsToRagDolls();
    
    g_physicsEngine.Simulate(dt); // runs ragdolls too
    g_collisionEngine.DetectAndResolveCollisions(dt);

    g_ragdollSystem.ApplyRagDollsToSkeletons();

    g_animationEngine.FinalizePoseAndMatrixPalette();
    
    for each gameObject {
        gameObject.FinalUpdate(dt);
    }
}
```
x??

---

#### Iteration Efficiency in Game Objects
When implementing multiple update phases, it is crucial to optimize the iteration over game objects. Directly iterating over all game objects and calling virtual functions can be expensive, especially if only a small percentage of objects need to perform any logic.

To improve efficiency, some engines use techniques like batch processing or dynamically selecting which objects require updates based on their state or needs.

:p Why is direct iteration over game objects inefficient in multiple update phases?
??x
Direct iteration over all game objects can be inefficient because it involves calling virtual functions for every object, even if only a small percentage of them need to perform any logic. This can lead to unnecessary CPU usage and performance degradation.

To improve efficiency, the engine might use batch processing or dynamic selection techniques. For example, the engine could selectively call update phases based on whether objects have changed state since the last frame.

Here is an example of optimized iteration:
```pseudocode
while (true) // main game loop
{
    for each gameObject that needs PreAnimUpdate {
        gameObject.PreAnimUpdate(dt);
    }
    
    g_animationEngine.CalculateIntermediatePoses(dt);
    
    for each gameObject that needs PostAnimUpdate {
        gameObject.PostAnimUpdate(dt);
    }

    g_ragdollSystem.ApplySkeletonsToRagDolls();
    
    g_physicsEngine.Simulate(dt); // runs ragdolls too
    g_collisionEngine.DetectAndResolveCollisions(dt);

    g_ragdollSystem.ApplyRagDollsToSkeletons();

    g_animationEngine.FinalizePoseAndMatrixPalette();
    
    for each gameObject that needs FinalUpdate {
        gameObject.FinalUpdate(dt);
    }
}
```
x??

---

#### Inflexible Design for Game Object Updates

Background context explaining why an inflexible design is problematic, especially when game objects need to interact with other engine systems like particle systems during animation phases.

:p What are the limitations of a design that only supports updating game objects and not other engine systems like particle systems?
??x
This design limitation means that if you want to update a particle system during the post-animation phase, there is no mechanism provided for this. The design would require custom handling or breaking encapsulation to allow such updates, leading to less flexibility.

To illustrate, consider a scenario where particles need to be updated based on an object's final world-space pose, which might not always align with the standard animation update phases:

```java
// Pseudocode example showing how an inflexible design might struggle

class ParticleSystem {
    void updateAfterAnimation(GameObject obj) {
        // This method cannot exist in a system that only updates game objects.
        // Instead, this logic would need to be integrated elsewhere, leading to coupling issues.
    }
}
```
x??

---

#### Generic Callback Mechanism for Updates

Background context explaining the benefits of using a generic callback mechanism to maximize performance and flexibility.

:p How does a generic callback mechanism differ from an inflexible design in terms of updating game objects?
??x
A generic callback mechanism allows any client code, not just game objects, to register callbacks for different update phases (pre-animation, post-animation, final). This approach maximizes both flexibility and performance by only calling registered clients when necessary.

```java
// Pseudocode example showing a simple callback registration and invocation

class AnimationSystem {
    private List<Callback> preAnimationCallbacks = new ArrayList<>();
    private List<Callback> postAnimationCallbacks = new ArrayList<>();
    private List<Callback> finalCallbacks = new ArrayList<>();

    void registerPreAnimationCallback(Callback callback) {
        preAnimationCallbacks.add(callback);
    }

    void registerPostAnimationCallback(Callback callback) {
        postAnimationCallbacks.add(callback);
    }

    void registerFinalCallback(Callback callback) {
        finalCallbacks.add(callback);
    }

    void performUpdate() {
        for (Callback cb : preAnimationCallbacks) {
            cb.preAnimation();
        }
        // Animation update
        for (Callback cb : postAnimationCallbacks) {
            cb.postAnimation();
        }
        for (Callback cb : finalCallbacks) {
            cb.finalUpdate();
        }
    }

    interface Callback {
        void preAnimation();
        void postAnimation();
        void finalUpdate();
    }
}
```
x??

---

#### Bucketed Updates Technique

Background context explaining how inter-object dependencies affect the order of updating and need to be handled.

:p How does the phased updates technique handle inter-object dependencies in game object updates?
??x
The phased updates technique needs adjustment when dealing with inter-object dependencies because these dependencies can create conflicts in the update order. For example, if one object depends on another for its final world-space pose, the animation system must ensure that dependent objects are updated only after their dependencies.

To manage this, objects can be grouped into buckets based on their dependency depth. The first bucket contains all root objects (those with no parents), the second contains their direct children, and so forth.

```java
// Pseudocode example showing how to group objects in buckets

class GameObject {
    List<GameObject> children = new ArrayList<>();
    boolean hasParent;
}

List<Bucket> buckets = new ArrayList<>();

for (GameObject obj : gameObjects) {
    if (!obj.hasParent) {
        buckets.add(0, obj); // Add root objects
    } else {
        for (GameObject parent : obj.children) {
            int bucketIndex = buckets.indexOf(parent);
            if (bucketIndex == -1) {
                buckets.add(bucketIndex + 1, obj); // Add children in correct order
            }
        }
    }
}

for (Bucket bucket : buckets) {
    performUpdateOn(bucket);
}
```
x??

---

#### Inter-Object Dependency Visualization

Background context explaining how inter-object dependencies can be visualized as a forest of dependency trees.

:p How are inter-object dependencies visualized, and what do the root objects represent in this model?
??x
Inter-object dependencies can be visualized using a forest of dependency trees. Each tree in the forest represents a set of interconnected game objects where each object depends on another (or its parent). The root objects are those that have no parents and thus form the base of each tree.

Here’s how you might visualize it:

```java
// Pseudocode example showing visualization of inter-object dependencies

class GameObject {
    List<GameObject> children = new ArrayList<>();
    boolean hasParent;
}

List<GameObject> gameObjects = ...; // Assume this list is populated with game objects and their parent relationships

List<Tree> dependencyTrees = new ArrayList<>();

for (GameObject obj : gameObjects) {
    if (!obj.hasParent) { // If the object has no parent, it's a root
        Tree tree = new Tree();
        tree.root = obj;
        dependencyTrees.add(tree);
    } else { // Otherwise, add to one of the trees in the forest based on its parent
        for (Tree tree : dependencyTrees) {
            if (tree.root.children.contains(obj)) {
                tree.addChild(obj);
                break;
            }
        }
    }
}
```
x??

---

#### Bucketed Update Loop
This section describes how a game engine might use a bucketed, phased, batched update loop to manage different types of game objects. The loop is structured into multiple buckets for different object categories, such as vehicles and platforms, characters, and attached objects.

In the provided code, the `UpdateBucket` function updates one specific type of object in each pass through the loop. After updating all necessary objects, the rendering engine renders the scene.
:p What does the `UpdateBucket` function do?
??x
The `UpdateBucket` function updates a specific bucket (category) of game objects. It follows these steps:
1. Pre-animates the update for each object in the bucket.
2. Calculates intermediate poses using the animation engine.
3. Post-animates the update for each object.
4. Applies skeletons to ragdolls.
5. Simulates physics for the specified bucket.
6. Detects and resolves collisions.
7. Applies ragdolls to skeletons.
8. Finalizes pose and matrix palette.
9. Performs final updates for each object.

This ensures that objects of the same type are updated in a similar manner, reducing complexity and potential issues related to state inconsistencies.

Code example:
```cpp
void UpdateBucket(Bucket bucket) {
    // Pre-animates update for each game object in the bucket
    for (auto gameObject : objectsInBucket[bucket]) {
        gameObject.PreAnimUpdate(dt);
    }

    // Calculates intermediate poses using the animation engine
    g_animationEngine.CalculateIntermediatePoses(bucket, dt);

    // Post-animates update for each game object in the bucket
    for (auto gameObject : objectsInBucket[bucket]) {
        gameObject.PostAnimUpdate(dt);
    }

    // Applies skeletons to ragdolls
    g_ragdollSystem.ApplySkeletonsToRagDolls(bucket);

    // Simulates physics for the specified bucket
    g_physicsEngine.Simulate(bucket, dt);

    // Detects and resolves collisions
    g_collisionEngine.DetectAndResolveCollisions(bucket, dt);

    // Applies ragdolls to skeletons
    g_ragdollSystem.ApplyRagDollsToSkeletons(bucket);

    // Finalizes pose and matrix palette
    g_animationEngine.FinalizePoseAndMatrixPalette(bucket);

    // Performs final updates for each game object in the bucket
    for (auto gameObject : objectsInBucket[bucket]) {
        gameObject.FinalUpdate(dt);
    }
}
```
x??

---
#### State Vector Update
This concept describes how the state of a game object changes over time, with an emphasis on the differences between sequential and parallel updates. In practice, when a single-threaded update loop runs, only one game object is updated at a time.

The current state vector \(S_i(t_2)\) of a game object i at time \(t_2\) can be derived from its previous state vector \(S_i(t_1)\).

:p What happens during the `UpdateBucket` function when using a single-threaded update loop?
??x
During the `UpdateBucket` function in a single-threaded update loop, each game object is updated one by one. This means that if there are 100 objects and the loop has processed half of them (50 objects), only those 50 objects will have their states updated to \(S_i(t_2)\). The remaining 50 objects will still be in their previous state \(S_i(t_1)\).

This sequential update can lead to inconsistencies if multiple game objects need to reference the current time or each other's states during updates. For instance, if a character object queries its position and an attached vehicle object simultaneously, they might receive different states because one of them has not yet been updated.

Code example:
```cpp
void UpdateGameObjects() {
    // Single-threaded update loop for objects in kBucketVehiclesAndPlatforms bucket
    for (auto gameObject : vehiclesAndPlatforms) {
        gameObject.PreAnimUpdate(dt);
    }

    g_animationEngine.CalculateIntermediatePoses(kBucketVehiclesAndPlatforms, dt);

    for (auto gameObject : vehiclesAndPlatforms) {
        gameObject.PostAnimUpdate(dt);
    }

    // Simulate physics and other processes in the same manner as described earlier
}
```
x??

---
#### Object State Inconsistencies
This section discusses potential issues that arise when game objects are updated sequentially within a single-threaded update loop. Specifically, it highlights how different game objects might have different states if they query their state during an update.

:p How can object state inconsistencies occur in a single-threaded update loop?
??x
Object state inconsistencies can occur because the state of all game objects is not updated simultaneously but rather one after another in a single-threaded context. This means that some objects may be in a partially updated state while others are still in their initial or previous states.

For example, consider two objects: Object A and Object B. If Object A queries its state before Object B has been fully updated, it might receive an outdated state. Conversely, if Object B needs to reference the current state of Object A during its update but Object A's state is not yet final, Object B may also receive a stale or partially updated state.

This can lead to issues like "one-frame-off" lag, where objects are in slightly different states compared to their true current state due to the sequential nature of updates.

Code example:
```cpp
class GameObject {
public:
    void Update() {
        // Assume this method is called during the single-threaded update loop
        if (IsUpdatingState()) {
            PreAnimUpdate(dt);
            PostAnimUpdate(dt);
        } else {
            FinalUpdate(dt);
        }
    }

private:
    bool IsUpdatingState() {
        // This function checks if the object's state has been updated yet
        return currentStateUpdated;
    }

    void PreAnimUpdate(float dt) { /* ... */ }
    void PostAnimUpdate(float dt) { /* ... */ }
    void FinalUpdate(float dt) { /* ... */ }
};
```
x??

---

#### Partially Updated Game Object States
Background context: This section discusses how game objects may be in a partially updated state during an update loop, leading to inconsistencies. It explains that while some states like animation pose blending might have been applied, others such as physics and collision resolution might not yet have occurred.

:p What is the main issue with partially updating game object states?
??x
The main issue is that different objects in the game may be at different stages of their update process during a single frame. This can lead to inconsistencies where an object might think it's in state \( t2 \) while another thinks it’s still in state \( t1 \). For example, if object B needs information from object A, and object A has not yet updated its physics, this could cause problems.
x??

---
#### Rule of Consistency During Update Loop
Background context: The text outlines a fundamental rule that the states of all game objects are consistent before and after an update loop but may be inconsistent during it. This is critical to understand when dealing with interdependent game objects.

:p What does the rule state about object states before, during, and after the update loop?
??x
The rule states that:
- Before the update loop starts, all game objects are in a consistent state.
- During the update loop, there can be inconsistencies as some objects might have completed certain updates (like animation) while others are still updating (e.g., physics).
- After the update loop completes, all game objects will again be in a consistent state.

This rule is important for debugging and ensuring that interdependent game objects behave predictably.
x??

---
#### Update Order Problem
Background context: The text explains how querying object states during the update loop can lead to issues if the order of updates is not managed correctly. This is particularly problematic when an object depends on the state of another object.

:p What happens in an "update order problem"?
??x
In an update order problem, a game object B might need information from another object A at time \( t \). If A has been updated to state \( SA(t2) \), but B is still using the old state \( SA(t1) \), it can lead to inconsistencies. This manifests as one-frame-off flags where an object's state lags behind its peers, causing synchronization issues.

Example:
```java
if (objectB.velocity == objectA.velocity) {
    // Code that depends on consistent states
}
```
If the update order is incorrect, `objectB` might use \( SA(t1) \)'s velocity when it should be using \( SA(t2) \), leading to bugs.
x??

---
#### Object State Caching
Background context: To mitigate the issues with partial updates and inconsistent states during an update loop, the text suggests caching the previous state vectors of game objects. This ensures that objects can always query a consistent state regardless of their current position in the update sequence.

:p What is object state caching?
??x
Object state caching involves storing each object's previous state vector \( Si(t1) \) while it calculates its new state vector \( Si(t2) \). This allows any object to safely query another object’s previous state without worrying about the current update order. It ensures that a totally consistent state is always available, even during the calculation of the new state.

Example:
```java
public class GameObject {
    private Vector3 currentState;
    private Vector3 previousState;

    public void update() {
        // Calculate new state Si(t2)
        currentState = calculateNewState();

        // Cache old state before updating
        previousState = currentState.clone();
    }

    public Vector3 getPreviousState() {
        return previousState;
    }
}
```
x??

---
#### Linear Interpolation Between States
Background context: State caching enables linear interpolation between the previous and next states, allowing for smooth transitions over time. This is particularly useful in physics engines like Havok.

:p How can we use state caching for linear interpolation?
??x
By maintaining both the current and previous states of objects, we can perform linear interpolation to approximate an object's state at any point in between its last two updates. For example:
```java
public Vector3 interpolateState(float t) {
    float alpha = t - (int)t; // Alpha is the fractional part of t
    return currentState.add(previousState.subtract(currentState).multiply(alpha));
}
```
This method allows for smooth animations and transitions, making the game feel more fluid.

Example:
```java
public class InterpolationExample {
    GameObject objectA;

    public void update(float deltaTime) {
        // Update logic goes here

        Vector3 interpolatedPosition = objectA.interpolateState(deltaTime);
        // Use interpolatedPosition for rendering or physics calculations
    }
}
```
x??

#### Time-Stamping for Game Object States
Game developers face challenges ensuring consistency when multiple threads or processes update game objects. One approach is to apply time-stamping to game object states, allowing them to track their historical and current configurations.

:p How does time-stamping help maintain consistency in game object updates?
??x
Time-stamping helps by adding a timestamp to each state of the game object. When querying the state during an update loop, developers can easily check if the queried state corresponds to the previous or current time. This prevents inconsistencies that might arise from concurrent access.

For example:
```java
class GameObject {
    private Timestamp lastUpdatedTimestamp;
    private State currentState;

    public void setState(Timestamp timestamp, State newState) {
        this.lastUpdatedTimestamp = timestamp;
        this.currentState = newState;
    }

    public boolean isStateCurrent() {
        return getCurrentTimestamp().equals(lastUpdatedTimestamp);
    }
}
```
x??

---

#### Concurrency in Game Engine Subsystems
Concurrency can significantly enhance performance by utilizing multiple cores and threads for tasks like rendering, animation, audio processing, and physics calculations.

:p How does concurrency benefit game engine subsystems?
??x
Concurrency benefits game engine subsystems by allowing them to execute independently or concurrently, which can reduce overall execution time. For instance, an animation system can process animations in parallel across multiple cores.

Example of a concurrent job system:
```java
public class JobSystem {
    public void submitJob(Runnable task) {
        // Submit the task to be executed on available core(s)
    }
}

// An example of submitting jobs for each game object requiring animation blending.
jobSystem.submitJob(() -> {
    GameObject obj = getGameObjectById(id);
    if (obj.requiresAnimationBlending()) {
        blendAnimations(obj);
    }
});
```
x??

---

#### Performance-Critical Engine Systems
The most performance-critical parts of the engine, such as rendering, animation, audio, and physics, should be designed to support concurrent execution.

:p Which subsystems in a game engine are likely to benefit the most from parallel processing?
??x
Rendering, animation, audio, and physics are the subsystems that will benefit the most from parallel processing. For example, an animation system can handle multiple animations for different objects concurrently, thereby improving performance by utilizing all available cores.

Example job submission logic:
```java
public class AnimationSystem {
    private JobSystem jobSystem;

    public void updateAnimations() {
        List<GameObject> animatedObjects = getAnimatedObjects();
        for (GameObject obj : animatedObjects) {
            jobSystem.submitJob(() -> {
                blendAnimations(obj);
                computeWorldMatrices(obj);
                skinMeshes(obj);
            });
        }
    }
}
```
x??

---

#### Example of Concurrent Job Submission
In a game engine, various subsystems can be designed to submit jobs for parallel execution.

:p How does the `submitJob` method work in a job system?
??x
The `submitJob` method allows tasks to be submitted for concurrent or parallel execution. This method is crucial for managing and distributing workload across multiple cores efficiently.

Example usage of `submitJob`:
```java
public class JobSystem {
    private List<Runnable> jobs = new ArrayList<>();

    public void submitJob(Runnable task) {
        // Add the task to the job queue.
        jobs.add(task);
        
        // Optionally, kick off a thread to process the tasks in parallel.
        if (!jobs.isEmpty()) {
            Thread thread = new Thread(() -> {
                for (Runnable job : jobs) {
                    job.run();
                }
            });
            thread.start();
        }
    }
}
```
x??

---

---
#### Ensuring Thread-Safety through Locking Mechanisms
In concurrent programming, ensuring thread-safety is crucial to prevent data races. When dealing with low-level engine systems, it's essential that their interfaces are thread-safe to avoid conflicts between external clients and internal subsystem operations.

For user-level threads (coroutines or fibers), spin locks can be used because these threads run at the same level as the calling context.
:p What is a suitable locking mechanism for user-level threads in concurrent programming?
??x
Spin locks are appropriate for user-level threads since they allow the thread to wait on a lock without yielding control to the operating system, maintaining responsiveness. The logic involves checking if the lock is available; if not, the thread will repeatedly check until it can acquire the lock.
```java
// Pseudo-code example of using spin lock
while (!spinLock.tryLock()) {
    // Thread checks and retries acquiring the lock without yielding
}
try {
    // Critical section code here
} finally {
    spinLock.unlock();
}
```
x??

---
#### Lock-Free Data Structures in Engine Subsystems
Implementing lock-free data structures can enhance performance by avoiding explicit locking mechanisms. However, designing such data structures is complex and not all types of data can be implemented without locks.

This approach is recommended for subsystems with high performance requirements.
:p When would you prefer using a lock-free implementation over traditional locking?
??x
Lock-free implementations are suitable when the overhead of locking becomes a bottleneck in highly concurrent environments, especially in critical sections of the engine that require frequent and fast access. However, implementing these structures requires deep understanding of atomic operations and memory barriers to ensure consistency.

Here is an example of using lock-free techniques with atomic operations:
```java
import java.util.concurrent.atomic.AtomicReference;

public class LockFreeStack {
    private final AtomicReference<Node> head = new AtomicReference<>(null);

    public void push(T item) {
        Node newNode = new Node(item);
        Node prevHead;
        do {
            prevHead = head.get();
            newNode.next = prevHead;
        } while (!head.compareAndSet(prevHead, newNode));
    }

    // More methods for pop and peek would be defined here
}
class Node {
    T item;
    Node next;

    public Node(T item) {
        this.item = item;
    }
}
```
x??

---
#### Asynchronous Program Design in Game Object Updates
Asynchronous programming is essential when interfacing with concurrent engine subsystems to avoid blocking the calling thread. This approach involves sending requests and processing results asynchronously.

In an asynchronous design, a function sends a request without waiting for it to complete.
:p What is the difference between synchronous and asynchronous function calls in the context of game object updates?
??x
Synchronous functions block the calling thread until they return, whereas asynchronous functions send a request immediately and do not wait for it to complete. Instead, they allow the calling thread to continue with other tasks.

Here's an example of how you might design an asynchronous ray-casting function:
```java
public interface RayCastRequest {
    void onResults(RayCastResult result);
}

public class GameObjUpdate {
    public void update() {
        // ... other game object updates ...
        
        // Asynchronous request for casting a ray
        requestRaycast(playerPos, enemyPos, new RayCastRequest() {
            @Override
            public void onResults(RayCastResult result) {
                if (result.hitSomething() && isEnemy(result.getHitObject())) {
                    // Player can see the enemy. Proceed with further logic.
                }
            }
        });
        
        // ... other game object updates ...
    }

    private void requestRaycast(Vector3 playerPos, Vector3 enemyPos, RayCastRequest callback) {
        // Enqueue the ray cast job and return immediately
        ThreadManager.enqueue(new RayCastJob(playerPos, enemyPos, callback));
    }
}
```
x??

---

---
#### Asynchronous Ray Casting in Game Loops
In games, using asynchronous ray casting can help maintain smooth gameplay by allowing other tasks to be performed while a ray cast is being processed. This approach helps in keeping the main thread busy with other work and avoids idle waiting times.

The example shows how to use `requestRayCast` to initiate a ray cast query and then `waitForRayCastResults` to retrieve the results later.
:p How does asynchronous ray casting benefit game development?
??x
Asynchronous ray casting benefits game development by allowing the main thread to continue processing other tasks while waiting for the result of a ray cast. This reduces idle time, improves frame rate stability, and enhances overall performance.

For instance, if you request a ray cast from `playerPos` to `enemyPos`, your game can still process other updates or perform additional tasks during this wait period. Once the results are ready, it can then process them without stalling.
x??

---
#### Deferring Request Results Processing
Sometimes, the processing of a request's result can be deferred until later in the frame update loop to reduce unnecessary computation and improve efficiency.

In the example provided, if you have already initiated a ray cast job in the previous frame, your current task might only involve waiting for its results. Once the job is complete, you process them.
:p When should the processing of a request's result be deferred?
??x
The processing of a request's result can often be deferred until later in the update loop if it does not need to happen immediately. This helps reduce unnecessary computations and improves overall performance by allowing the main thread to focus on other tasks.

For example, if you have initiated a ray cast job in one frame (`rayJobPending = true;`), you might wait for its results only when necessary, like during `SomeGameObject::Update()` where you check if there's pending data from the previous frame.
x??

---
#### Determining When to Make Asynchronous Requests
To optimize performance and resource utilization, it’s crucial to determine the optimal time within the game loop to make and process asynchronous requests.

The example demonstrates how to balance between making a request early in the update cycle and waiting for results later. This involves evaluating when you have enough information to initiate a request and how long you can afford to wait before using its results.
:p How do you decide the optimal time during the game loop to make an asynchronous request?
??x
The decision on when to make an asynchronous request should be based on when you have sufficient data or conditions to kick off the request. The earlier you initiate the request, the more likely it is to complete by the time you need its results, maximizing CPU utilization.

For example, in `SomeGameObject::Update()`, you might evaluate if there's enough information available to start a ray cast and then wait for its results later using `rayJobPending` and `waitForRayCastResults`.

```cpp
SomeGameObject::Update() {
    // ... other work ...
    
    // Check if we have pending data from the previous frame.
    if (rayJobPending) {
        waitForRayCastResults(&r);
        
        // Process the results...
        if (r.hitSomething() && isEnemy(r.getHitObject())) {
            // Player can see the enemy.
            // ... 
        }
    }

    // Cast a new ray for the next frame.
    rayJobPending = true;
    requestRayCast(playerPos, enemyPos, &r);

    // Continue with other work...
}
```
x??

---
#### Job Dependencies and Degree of Parallelism
To fully utilize parallel computing platforms, it’s essential to ensure that all cores remain busy. This can be achieved by designing job systems that run hundreds or thousands of jobs concurrently.

In a game engine using such a system, each iteration of the game loop could consist of many concurrent jobs, helping to maximize CPU utilization.
:p How do you design a job system for parallel computing in games?
??x
Designing a job system for parallel computing in games involves structuring tasks into small, independent jobs that can run concurrently across multiple cores. Each job should be designed to be as lightweight and self-contained as possible.

For example, you might break down the game loop into numerous tasks such as ray casting, physics simulations, AI updates, etc., each running in parallel. This ensures all cores remain busy and no idle time is wasted.

```cpp
class JobManager {
public:
    void addRayCastJob(Vector3 playerPos, Vector3 enemyPos) {
        // Add a job to cast a ray from player to enemy.
    }

    bool hasJobsComplete() {
        // Check if any jobs are complete.
        return false;
    }
};

// Example of adding and checking jobs in SomeGameObject::Update()
SomeGameObject::Update() {
    JobManager manager;

    // ... other work ...

    // Add a ray cast job
    manager.addRayCastJob(playerPos, enemyPos);

    // Do other work...

    // Check if the ray cast job is complete.
    while (!manager.hasJobsComplete()) {
        // Wait for jobs to complete.
    }

    RayCastResult r;
    requestRayCast(playerPos, enemyPos, &r);
    
    // Process results...
}
```
x??

---

#### Degree of Parallelism (DOP) and Dependency Graphs

Background context: The degree of parallelism (DOP), also known as the degree of concurrency (DOC), measures how many jobs can run in parallel. It is determined by drawing a dependency graph where nodes represent jobs, and arrows show dependencies between them. The number of leaf nodes gives the DOP.

:p What does DOP measure in a system?
??x
DOP measures the theoretical maximum number of jobs that can be running in parallel at any given moment.
x??

---

#### Job Dependency Trees

Background context: Job dependency trees illustrate how jobs depend on each other, forming a hierarchy. The number of leaf nodes indicates the DOP. For instance, if there are 4 leaf nodes, the system’s DOP is 4.

:p How can you determine the DOP from a job dependency tree?
??x
The DOP is determined by counting the number of leaf nodes in the job dependency tree.
x??

---

#### Synchronization Points (SyncPoints)

Background context: A synchronization point occurs when one job depends on another. This creates an opportunity for idle time if a job waits for its dependent jobs to complete. These points can waste valuable CPU resources.

:p What is a synchronization point?
??x
A synchronization point is introduced whenever one job is dependent upon the completion of one or more other jobs, causing potential idle time and wasted CPU resources.
x??

---

#### Maximizing Hardware Utilization

Background context: To achieve full utilization of CPU resources in an explicitly parallel computer, we aim to match or exceed the number of available cores with our DOP. If the DOP equals the number of cores, throughput is maximized. If it's higher, some jobs must run serially; if lower, some cores will be idle.

:p How can you increase the DOP in a system?
??x
To increase the DOP, we can reduce dependencies between jobs or find unrelated work to do during idle periods.
x??

---

#### Deferring Synchronization Points

Background context: By deferring synchronization points, we can avoid idle time. For example, if job D depends on jobs A, B, and C, scheduling it before these are completed will cause it to wait, but scheduling it later ensures no waiting.

:p What is the benefit of deferring a synchronization point?
??x
Deferring a synchronization point reduces or eliminates idle time caused by waiting for dependent jobs, thereby improving CPU utilization.
x??

---

#### Code Example: Deferring Jobs

Background context: The following code example demonstrates how to defer a job's execution until all its dependencies are completed.

:p Show an example of deferring job D in Java.
??x
```java
public class JobScheduler {
    private final List<Job> jobs = new ArrayList<>();

    public void addJob(Job job) {
        jobs.add(job);
        for (Job dependency : job.getDependencies()) {
            if (!jobs.contains(dependency)) {
                // Schedule the dependency first
                addJob(dependency);
            }
        }
    }

    public void executeJobs() {
        while (!jobs.isEmpty()) {
            List<Job> toExecute = new ArrayList<>();
            for (Job job : jobs) {
                if (allDependenciesCompleted(job)) {
                    toExecute.add(job);
                }
            }
            // Execute all ready jobs
            for (Job job : toExecute) {
                job.execute();
            }
            // Remove executed jobs from the list
            jobs.removeIf(Job::isExecuted);
        }
    }

    private boolean allDependenciesCompleted(Job job) {
        return job.getDependencies().stream()
                .allMatch(dependency -> dependency.isExecuted());
    }
}
```
This code ensures that a job is only executed once all its dependencies have been completed, effectively deferring the job if necessary.
x??

---

---
#### Game Object Interdependencies and Parallelization Challenges
Background context explaining why game objects are challenging to parallelize. Game objects often interact with each other and depend on various engine subsystems, leading to complex inter-object dependencies that can make concurrent processing difficult.

:p What are some reasons why game object models are hard to parallelize?
??x
Game objects tend to be highly interdependent and typically rely on numerous engine subsystems like animation, audio, collision/physics, rendering, file I/O, etc. These interactions frequently occur multiple times during the update loop and can be unpredictable due to player inputs and events in the game world.
x??

---
#### Job System for Game Object Updates
Explanation of using a job system to parallelize game object updates. Describes how this approach was implemented by Naughty Dog in The Last of Us: Remastered.

:p How did Naughty Dog implement concurrent game object model updates?
??x
Naughty Dog implemented game object updates as jobs, which are scheduled by the job system across available cores. This allows for better utilization of CPU resources compared to single-threaded processing.
x??

---
#### Bucketing Game Objects for Parallelization
Explanation of bucketing objects based on their dependencies to achieve parallelism.

:p How do you divide game objects into buckets for parallel updates?
??x
Game objects are divided into buckets (Nbuckets) such that each object in a given bucket depends only on objects in the preceding buckets. This ensures that objects can be processed in an order that respects inter-object dependencies.
x??

---
#### Scheduling Jobs Across Cores
Explanation of how to schedule jobs across multiple cores effectively.

:p How does the job system help in scheduling game object updates across cores?
??x
The job system schedules jobs across available cores, allowing for efficient parallel processing. By intelligently managing the order and timing of job invocations, idle time can be minimized.
x??

---
#### Example Job Scheduling Scenario
Illustration of a specific scenario where job D depends on A, B, and C.

:p What is an example scenario to illustrate job scheduling?
??x
In the provided diagram, if job D is scheduled immediately after job C on Core 2, it will sit idle waiting for job B to complete. However, delaying the invocation of job D until well after jobs A, B, and C have completed allows Core 2 to run other jobs, avoiding idle time.
```
Diagram:
Sync point creates idle time on Core 2
Core 0: F A I
Core 1: E B J
Core 2: G H C D (idle)
```

x??

---
#### Code Example for Job Scheduling
Pseudocode example to illustrate job scheduling logic.

:p How can we implement the job scheduling logic in pseudocode?
??x
```pseudocode
function scheduleJobs(jobs) {
    let buckets = partitionJobsIntoBuckets(jobs);
    for (bucket in buckets) {
        for (job in bucket) {
            if (dependenciesMet(job, completedJobs)) {
                executeJob(job);
                addJobToRunningList(job);
            }
        }
    }
}

function dependenciesMet(job, completedJobs) {
    for (dependency in job.dependencies) {
        if (!completedJobs.includes(dependency.id)) return false;
    }
    return true;
}
```

x??

---
#### Conclusion on Game Object Parallelization
Summary of the challenges and solutions discussed.

:p What are the key takeaways from this section about game object parallelization?
??x
Key takeaways include understanding that while it's possible to update game objects concurrently, doing so requires careful management of interdependencies. Techniques like bucketing and job systems can help achieve efficient parallel processing but require complex coordination.
x??

---

#### Asynchronous Game Object Updates
Background context: In game development, especially when dealing with large numbers of objects or complex interactions, updates to game objects are often handled asynchronously. This means that instead of waiting for an update to complete before proceeding, the system schedules tasks to run later and continues execution.

Relevant formulas: None specific, but understanding how parallel processing and task scheduling affect performance is key.

If applicable, add code examples with explanations.
:p What is the primary method used in asynchronous game object updates?
??x
The primary method involves kicking off a job or task for updating a game object, which runs asynchronously. The main thread continues execution without waiting for this task to complete. Later, results from these tasks can be fetched and acted upon.

For example:
```cpp
void UpdateGameObject(GameObject* obj) {
    // Schedule an asynchronous update job
    job::Declaration decl;
    decl.m_pEntryPoint = UpdateGameObjectJob;
    decl.m_param = reinterpret_cast<uintptr_t>(obj);
    
    // Schedule the job to run later
    job::KickJobsAndWait(decl, 1);
}

// Later in the frame or next frame
void HandleUpdateResults() {
    // Fetch and process results from previous asynchronous tasks
}
```
x??

---

#### Dependency Graph for Game Objects
Background context: In complex game scenarios, dependencies between game objects can create a dependency graph. This graph helps in understanding which updates need to be performed first based on the dependencies.

Relevant formulas: Not specific, but understanding topological sorting and cycle detection algorithms is helpful.

If applicable, add code examples with explanations.
:p How do we handle game object updates when there are inter-object dependencies?
??x
When there are inter-object dependencies, we can use a dependency graph to manage the order of updates. We start by updating all objects that have no incoming edges (i.e., nodes without any outgoing dependencies). As each update completes, we check its dependents and ensure they get updated only after their dependencies are completed.

Here is an example:
```cpp
void UpdateGameObjects() {
    const int count = GetNumGameObjects();
    
    for (int i = 0; i < count; ++i) {
        if (!HasIncomingEdges(i)) { // Check if the object has no incoming edges
            job::Declaration decl;
            decl.m_pEntryPoint = UpdateGameObjectJob;
            decl.m_param = reinterpret_cast<uintptr_t>(GetGameObject(i));
            
            job::KickJobsAndWait(decl, 1);
        }
    }
    
    while (!AllUpdatesComplete()) { // Wait until all updates are complete
        for (int i = 0; i < count; ++i) {
            if (HasIncomingEdges(i)) {
                if (AreDependenciesComplete(i)) {
                    job::Declaration decl;
                    decl.m_pEntryPoint = UpdateGameObjectJob;
                    decl.m_param = reinterpret_cast<uintptr_t>(GetGameObject(i));
                    
                    job::KickJobsAndWait(decl, 1);
                }
            }
        }
    }
}
```
x??

---

#### Handling Cycles in Dependency Graphs
Background context: When the dependency graph contains cycles (i.e., circular dependencies), it becomes challenging to determine a clear order for updating game objects. This is because circular dependencies indicate that no object can be updated without waiting for another, creating an infinite loop.

Relevant formulas: None specific, but cycle detection algorithms are crucial here.

If applicable, add code examples with explanations.
:p How do we handle cycles in the dependency graph of game objects?
??x
Handling cycles in the dependency graph involves isolating cyclically-dependent game objects and updating them serially. This means that a group of objects that depend on each other cannot be updated concurrently; instead, they are treated as separate units.

Here’s an approach to handle this:
```cpp
void HandleCyclesInGraph() {
    // Detect cycles in the dependency graph
    if (DetectCycle()) {
        // Isolate and update cyclically-dependent game objects serially
        for (auto& cycle : GetCycles()) {
            for (GameObject* obj : cycle) {
                job::Declaration decl;
                decl.m_pEntryPoint = UpdateGameObjectJob;
                decl.m_param = reinterpret_cast<uintptr_t>(obj);
                
                job::KickJobsAndWait(decl, 1); // Update each object in the cycle
            }
        }
    } else {
        // If no cycles, use normal update process
        UpdateGameObjects();
    }
}
```
x??

---

#### Blocking Calls in Job Systems
Background context: While asynchronous updates are common to avoid blocking, there might be cases where blocking calls can still be useful. This is especially true when using user-level threads like coroutines or fibers.

Relevant formulas: None specific, but understanding the concept of cooperative multitasking and yielding is key.

If applicable, add code examples with explanations.
:p How do we handle blocking calls in a job system based on coroutines/fibers?
??x
In a job system that uses user-level threads (coroutines or fibers), blocking calls can be used because these threads have the ability to yield control and continue execution later. This means that a task can pause midway through its execution, allowing other tasks to run before resuming.

Here’s an example:
```cpp
void SomeGameObject::Update() {
    // Cast a ray to see if the player has line of sight to the enemy
    RayCastResult result = castRayAndWait(playerPos, enemyPos);
    
    // Process the result without blocking further updates
}
```
In this case, `castRayAndWait` is a function that blocks until it gets the result. However, since coroutines or fibers can yield control and resume execution later, this does not block other tasks from running.

x??

---

#### Job System Execution with User-Level Threads
Background context explaining how user-level threads can interrupt and resume job execution, allowing blocking function calls to be made within a job. This mechanism helps implement functions like `WaitForCounter()` or `KickJobsAndWait()`, which block jobs temporarily.

:p How does the job system handle blocking function calls during its execution?
??x
The job system can pause a job part-way through its execution when a blocking function call is encountered, such as a ray cast. Once the blocking operation completes, the job can resume from where it was paused.

```java
// Pseudocode example for handling blocking operations in jobs
void processJob(Job j) {
    if (j.isBlockingOperationNeeded()) {
        j.pause();
        performBlockingOperation(); // e.g., ray casting
        j.resume();
    } else {
        j.execute();
    }
}
```
x??

---

#### Bucketed Updates and Inter-Object Dependencies
Background context explaining how bucketed updates address the problem of inter-object dependencies by ensuring objects are updated in a global order, thus helping with state queries. However, there is still a one-frame-off issue when querying objects within the same bucket.

:p How do bucketed updates help manage inter-object dependencies?
??x
Bucketed updates help by dividing game object updates into buckets based on a global order (e.g., updating train cars before objects sitting on them). This ensures that an object in bucket B can safely query the state of objects in buckets B−∆ or B+∆ without interference, as we know those objects won't be concurrently updated.

```java
// Pseudocode example for bucketed updates
void updateGameObjects() {
    for (int i = 0; i < numBuckets; ++i) {
        for (GameObject obj : buckets[i]) {
            obj.update();
        }
    }
}
```
x??

---

#### Locking During Game Object Updates
Background context explaining the need to avoid data races when game objects in the same bucket interact or query one another. Introducing a single global lock could resolve this but would serialize all updates within that bucket, defeating the purpose of concurrent updates.

:p How can we safely handle interactions between game objects within the same bucket without using a global lock?
??x
To safely handle interactions between game objects in the same bucket without causing data races, we can use temporal queries. Objects can query the state of other objects from previous frames or future frames based on their bucket order, avoiding concurrent updates.

```java
// Pseudocode example for safe inter-object communication
void updateObject(GameObject obj) {
    // Safe to query objects in B−∆ and B+∆ without locks
    GameObject otherObj = obj.bucket.getOtherObject();
    if (obj.frameNumber > otherObj.lastUpdateFrame) {
        // Perform safe queries here
        obj.doSomethingWith(otherObj);
    }
}
```
x??

---

#### Global Locking System
Background context: The global locking system was an approach used early at Naughty Dog to manage concurrency in game object updates. However, it only acquired locks when dereferencing a game object handle within its update function. Game objects are referenced by handles rather than raw pointers to support memory defragmentation.
:p What is the primary issue with the initial global locking system?
??x
The main issue was that the locking mechanism was overly complex and difficult to work with, leading to inefficient use of CPU cores during bucket updates.
x??

---

#### Object Snapshots
Background context: Many interactions between game objects are state queries rather than direct modifications. To optimize this, snapshots were introduced—read-only copies of a game object's state that can be queried without locks or fear of data races.
:p How do snapshots help in managing concurrency?
??x
Snapshots allow multiple game objects to query the state of others concurrently and safely, as they are read-only. This avoids the need for locks during state queries, improving efficiency.
```cpp
// Example of how a snapshot might be used in C++
class GameObject {
public:
    // Method to generate a snapshot
    Snapshot getSnapshot() const {
        return Snapshot(*this);
    }

    // Snapshot class representing a read-only copy
    class Snapshot {
        // State information from the original object
    };
};
```
x??

---

#### Handling Inter-Object Mutation
Background context: While snapshots help with state queries, they do not address inter-object mutation within the same bucket. To handle this, Naughty Dog used techniques such as minimizing mutations and deferring them to later buckets.
:p What are some methods used to handle inter-object mutations?
??x
Methods include:
1. Minimizing inter-object mutation where possible.
2. Requesting mutations through a queue that is protected by a lock and handled after the bucket update completes.
3. Spawning jobs in the next bucket to synchronize these operations, ensuring objects are not concurrently updating during such actions.
```cpp
// Example of request queue handling
class MutationRequestQueue {
public:
    void addMutationRequest(GameObject* object1, GameObject* object2) {
        // Add mutation request to the queue
    }

    void handleRequests() {
        // Handle all requests after bucket update completion
    }
};
```
x??

---

#### Future Improvements
Background context: The bucketed updating system has room for improvement. Each transition between buckets introduces sync points in the game loop, causing some CPU cores to idle while waiting for all objects in a bucket to complete their updates.
:p What are some potential improvements for the bucketed update system?
??x
Potential improvements include optimizing the transitions between buckets to reduce sync points and ensure more efficient use of CPU cores. This could involve more sophisticated job scheduling or other concurrency techniques.
```java
// Example of optimizing transitions in Java
public class BucketManager {
    // Method to manage transitions between buckets efficiently
    public void optimizeTransitions() {
        // Logic to minimize idle time during bucket transitions
    }
}
```
x??

---


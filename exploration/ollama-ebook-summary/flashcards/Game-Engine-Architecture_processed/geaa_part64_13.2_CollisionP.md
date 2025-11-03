# Flashcards: Game-Engine-Architecture_processed (Part 64)

**Starting Chapter:** 13.2 CollisionPhysics Middleware

---

#### Collision/Physics Middleware Overview
Background context: The text discusses the integration of collision and physics middleware into game development, highlighting its significance and various impacts. Middleware is essential for handling complex simulations like collisions and rigid body dynamics, which can be challenging to implement from scratch.

:p What are some key reasons why teams choose to integrate collision and physics middleware in their games?
??x
The decision to use collision and physics middleware is driven by several factors:

1. **Complexity**: Implementing a robust collision system and physics engine from scratch requires significant expertise.
2. **Cost and Resources**: Developing such systems can add considerable costs, both financial and in terms of development time.
3. **Interdisciplinary Collaboration**: Integrating these systems often necessitates close cooperation between different teams (engineering, art, audio, design).
4. **Risk Management**: There is an increased risk associated with developing complex physics systems.

These factors make middleware a practical choice for most game projects.

x??

---

#### Open Dynamics Engine (ODE)
Background context: ODE is described as an open-source collision and rigid body dynamics SDK that offers similar features to commercial products like Havok. It's particularly beneficial due to its free availability, which is advantageous for smaller studios and educational projects.

:p What are the key benefits of using ODE in game development?
??x
ODE provides several key benefits:

1. **Free Availability**: It is a free product, making it accessible to small game studios and educational projects.
2. **Full Source Code**: The availability of full source code simplifies debugging and allows for customization tailored to specific needs.
3. **Feature Set Similarity**: Its feature set is comparable to commercial products like Havok.

These benefits make ODE a compelling choice for developers looking for robust physics capabilities without the cost.

x??

---

#### Bullet
Background context: Bullet is an open-source collision detection and physics library used in both game and film industries. It integrates its collision engine with dynamics simulation but provides flexibility through hooks, supporting continuous collision detection (CCD), which can be particularly useful for fast-moving objects.

:p What is a key feature of the Bullet SDK that makes it suitable for games involving small, fast-moving objects?
??x
A key feature of the Bullet SDK is **continuous collision detection (CCD)**, also known as time of impact (TOI) collision detection. This feature is crucial for games with fast-moving and potentially small objects because:

- **Preventing Penetration**: CCD helps prevent objects from penetrating each other when they collide.
- **Smoothing Animations**: It ensures smoother animations by accurately detecting the point of contact.

This makes Bullet particularly suitable for scenarios where precise handling of moving objects is necessary, such as in real-time strategy games or first-person shooters.

x??

---

#### TrueAxis
Background context: TrueAxis is a collision/physics SDK that is free for non-commercial use. The text does not provide extensive details on its features or performance, but it highlights its availability for non-commercial projects.

:p What type of project would benefit most from using TrueAxis?
??x
TrueAxis would be most beneficial for **non-commercial projects** such as personal projects, student work, and smaller indie games where budget constraints are a significant factor. Its free nature makes it an attractive choice for developers who do not have the financial resources to invest in more expensive commercial middleware.

x??

---

#### PhysX (NVIDIA)
Background context: PhysX started as Novodex, produced by Ageia, and was later acquired by NVIDIA. It supports both CPU and GPU execution, with a free version available for non-commercial use. The SDK is now part of APEX, NVIDIA’s scalable multiplatform dynamics framework.

:p What are the key benefits of using PhysX (NVIDIA) in game development?
??x
PhysX offers several key benefits:

1. **Free Version**: It provides a free version for non-commercial projects.
2. **Multiplatform Support**: It supports multiple platforms including Windows, Linux, Mac, Android, Xbox 360, PlayStation 3, Xbox One, PlayStation 4, and Wii.
3. **GPU Acceleration**: The ability to run on GPUs can significantly enhance performance.
4. **Customizability**: Developers have the option to pay for full source code and customization capabilities.

These features make PhysX a versatile and powerful choice for game development projects.

x??

---

#### Havok
Background context: Havok is described as one of the gold standards in commercial physics SDKs, offering rich feature sets and excellent performance across all supported platforms. It includes various optional add-ons such as vehicle physics systems and destructible environment modeling.

:p What are some notable features of Havok?
??x
Havok stands out with several notable features:

1. **Rich Feature Set**: Provides one of the most comprehensive feature sets among commercial physics SDKs.
2. **Performance**: Offers excellent performance characteristics on all supported platforms.
3. **Optional Add-ons**: Includes add-ons like a vehicle physics system and destructible environment modeling.
4. **Cost**: It is the most expensive solution but offers extensive capabilities.

These features make Havok a premium choice for developers needing advanced physics simulations, even though it comes with a higher price tag.

x??

---

#### Collision Detection System Overview

Background context: The primary purpose of a game engine's collision detection system is to determine whether any objects in the game world have come into contact. This involves representing each logical object by one or more geometric shapes, which are usually simple (spheres, boxes) but can be complex.

:p What is the main function of a game engine's collision detection system?
??x
The collision detection system determines whether any of the objects in the game world have come into contact at any given moment. This involves representing each logical object by one or more geometric shapes to check for intersections.
??x

---

#### Collision Detection Shapes

Background context: Collisions are detected between simple geometric shapes such as spheres, boxes, and capsules. More complex shapes can also be used but the basic principle remains the same.

:p What types of shapes are commonly used in collision detection?
??x
Commonly used shapes include spheres, boxes (cuboids), and capsules. These are simple and efficient to work with for collision detection.
??x

---

#### Collision Detection in Game Development

Background context: The collision system not only checks for intersection but also provides information about the nature of each contact. This is crucial for preventing visual anomalies like interpenetration.

:p What does a collision detection system provide beyond just determining if objects are intersecting?
??x
A collision detection system provides relevant information about the nature of each contact, which can be used to prevent unrealistic visual anomalies and handle various game mechanics.
??x

---

#### Rigid Body Dynamics Simulation

Background context: A rigid body dynamics simulation uses the collision system to mimic physically realistic behaviors like bouncing, rolling, sliding, and coming to rest. These simulations are often demanding on the collision system.

:p How does a rigid body dynamics simulation use the collision detection system?
??x
A rigid body dynamics simulation uses the collision detection system to handle physical interactions such as collisions, which can cause objects to bounce, roll, slide, or come to rest.
??x

---

#### Digital Molecular Matter (DMM)

Background context: Pixelux Entertainment's DMM engine simulates the dynamics of deformable bodies and breakable objects using finite element methods. It has both an offline and runtime component.

:p What is Digital Molecular Matter (DMM)?
??x
Digital Molecular Matter (DMM) is a physics engine that uses finite element methods to simulate the dynamics of deformable bodies and breakable objects, with applications in games like Star Wars: The Force Unleashed.
??x

---

#### Physics Abstraction Layer (PAL)

Background context: PAL is an open-source library allowing developers to work with multiple physics SDKs on a single project. It provides hooks for various physics engines.

:p What is the purpose of the Physics Abstraction Layer (PAL)?
??x
The Physics Abstraction Layer (PAL) allows developers to use more than one physics SDK in a single project by providing hooks and interfaces that abstract away specific implementation details.
??x

---

#### Collidable Entities

Background context: To make an object capable of colliding with others, it needs a collision representation defining its shape, position, and orientation. This is separate from the gameplay and visual representations.

:p What does a "collidable entity" need to have for collision detection?
??x
A collidable entity needs a collision representation that defines its shape, position, and orientation in the game world. This is distinct from its gameplay and visual representations.
??x

---

#### Simple Shapes for Collision Detection
Simple shapes like spheres, boxes, and capsules are used to approximate object volumes for collision detection. These shapes simplify the computational requirements while still providing reasonable accuracy.

:p Why are simple shapes preferred over more complex ones for collision detection?
??x
Simple shapes are favored because they reduce the computational complexity involved in detecting collisions. They are easier to work with mathematically and geometrically, making it faster and less resource-intensive to determine if two objects intersect or collide.
For example, checking whether a point lies inside a sphere is straightforward:
```java
public boolean isPointInsideSphere(Vector3 point, Vector3 center, float radius) {
    // Calculate the distance from the point to the center of the sphere
    return Vector3.distance(point, center) <= radius;
}
```
x??

---

#### Collidables and Their Components
A collidable entity in a game consists of two primary components: a shape and a transform. The shape defines the geometric form and size (e.g., dimensions or radius), while the transform positions and orients the shape within the game world.

:p What are the two basic pieces of information contained in a collidable?
??x
A collidable contains a shape, which describes the object's geometry and size, and a transform, which defines the position and orientation of the shape in the game world.
```java
// Example in C++ (assuming Havok API)
hkpCollidable collidable;
collidable.setShape(myShape);
collidable.setTransform(myTransform);
```
x??

---

#### Reasons for Using Transforms
There are three main reasons why transforms are necessary: they help position and orient the shape correctly, allow efficient movement of complex shapes, and enable multiple collidables to share a single shape.

:p Why do we need transforms in a game?
??x
Transforms are needed because:
1. They position and orient the shape appropriately in the world space.
2. Moving complex shapes through space is inexpensive with a transform.
3. Multiple collidables can share a single shape, reducing memory usage.
```java
// Example in C++ (assuming Havok API)
hkpTransform myTransform;
myTransform.setTranslation(myPosition);
myTransform.setRotation(myOrientation);

hkpCollidable::setTransform(myTransform);
```
x??

---

#### Collision World and Singleton Design
A collision world is a singleton data structure used to manage all collidables in the game. It represents the complete game world for use by the collision detection system.

:p What is the purpose of a collision world?
??x
The collision world serves as a central entity that keeps track of all collidable objects and provides a complete representation of the game world specifically tailored for the collision detection system.
```java
// Example in C++
class hkpWorld {
public:
    static hkpWorld& getInstance() {
        // Ensure only one instance exists
        static hkpWorld instance;
        return instance;
    }

    void addCollidable(hkpCollidable* collidable);
};
```
x??

---

#### Havok's Collision World Implementation
Havok implements the collision world using a class called `hkpWorld`, which is responsible for managing all collidables and ensuring efficient collision detection.

:p How does Havok represent the game world in terms of collision detection?
??x
Havok represents the game world through an instance of the `hkpWorld` class, which manages all collidables. This allows it to efficiently handle and detect collisions between objects.
```java
// Example in C++ (assuming Havok API)
class hkpWorld {
public:
    void addCollidable(hkpCollidable* collidable) {
        // Add the collidable to the world for collision detection
    }
};
```
x??

---

#### Collision World vs. Game Object Collision Information

Collision information is managed separately from game objects to optimize performance and maintainability.

:p How does separating collision information from game objects benefit a game's performance and maintenance?
??x
Separating collision information from game objects has several advantages:
1. **Reduced Iteration Over Irrelevant Data**: Only relevant collidable shapes are stored in the collision world, reducing unnecessary computations.
2. **Efficient Cache Utilization**: The design allows for better organization of collision data to maximize cache coherence and improve performance.
3. **Encapsulation**: It provides a clear separation between game logic and physics logic, making the codebase more understandable, maintainable, testable, and reusable.

This separation is crucial as it ensures that only potentially colliding objects are considered in each frame, leading to faster computations.

??x
---

#### Physics World Integration

The physics world integrates closely with the collision system, sharing common data structures for efficiency.

:p How does the integration of a rigid body dynamics system with the collision system typically manifest?
??x
The integration between a rigid body dynamics system and the collision system is often seamless because:
- They share common "world" data structures.
- Each rigid body in simulation corresponds to a single collidable object in the collision system.

This design ensures frequent and detailed collision queries can be handled efficiently. Typically, the physics engine drives the collision system, running tests multiple times per frame to ensure accurate and up-to-date collision information.

:p How does this integration affect performance?
??x
The tight coupling between rigid body dynamics and collision systems improves performance by:
- Reducing redundant checks.
- Ensuring that the most current state of objects is used for collision detection.

Here’s a simple pseudocode example illustrating how this might work in a physics engine:

```pseudo
// Pseudocode Example
function updatePhysics(timestep) {
    // Update rigid body positions and velocities
    for each rigidBody in simulation {
        updateRigidBodyState(rigidBody, timestep)
    }
    
    // Run collision tests based on updated states
    runCollisionTests()
}

function updateRigidBodyState(rigidBody, timestep) {
    // Update state (position, velocity) using physics equations
}
```

This example shows how the rigid body dynamics and collision systems work in tandem to ensure accurate and efficient simulation.

??x
---

#### Havok and PhysX Rigid Body Concepts

Havok and PhysX handle collidables and rigid bodies differently but share a similar goal of integrating these concepts tightly within their physics engines.

:p How do Havok and PhysX manage the relationship between rigid bodies and collidables?
??x
In both Havok and PhysX, there is an integration between rigid bodies and collidables:
- **Havok**: A rigid body (`hkpRigidBody`) has a pointer to exactly one `hkpCollidable`.
- **PhysX**: The `NxActor` class serves as both the collidable and rigid body (though physical properties are stored separately in an instance of `NxBodyDesc`).

This design ensures that each dynamic rigid body is associated with only one collidable, optimizing performance by minimizing redundant checks.

:p Provide a simple example to illustrate this concept.
??x
Here’s how you might represent this relationship in code for PhysX:

```cpp
// Example C++ Code for PhysX
NxActor* actor = ...; // Create an Actor
NxBodyDesc bodyDesc;
bodyDesc.actor = actor; // Link the actor with its body description
actor->setBodyDescription(bodyDesc); // Apply the body description to the actor
```

This example shows how a single `NxActor` can be both a collidable and part of a rigid body simulation.

??x
---

#### Fixed Rigid Bodies

Some physics systems allow for static or fixed objects that are not part of dynamic simulations but still contribute to collision detection.

:p How do games handle fixed, non-dynamic rigid bodies?
??x
Fixed, non-dynamic rigid bodies can be configured in a game as follows:
- The system is told to ignore these objects during the dynamics simulation.
- They serve only as collidables for other dynamic objects.

Here’s an example of setting up a fixed rigid body in PhysX:

```cpp
// Example C++ Code for PhysX
NxActor* actor = ...; // Create an Actor
actor->setFixedFlag(true); // Mark the actor as non-dynamic but still collidable

// The actor can now be used to check collisions without affecting dynamics simulation.
```

This code demonstrates how a fixed rigid body is set up, ensuring it does not participate in dynamic simulations while still providing collision information.

??x
---

#### Shape Concepts in 2D and 3D
In two dimensions, a shape is defined by its boundary, which can be curved or composed of straight edges forming polygons. In three dimensions, shapes have volume, with boundaries that can be surfaces (with front and back but no inside) or polyhedra.
:p What are the basic definitions of shapes in 2D and 3D?
??x
In 2D, a shape is defined by its boundary which could be curved lines or polygons. In 3D, a shape has volume and its boundary can either be a surface (with front and back) or a polyhedron composed of polygons.
x??

---
#### Surfaces in 3D Space
Surfaces are two-dimensional geometric entities with no inside or outside but have a defined front and back. Examples include planes, triangles, subdivision surfaces, and surfaces formed by groups of connected triangles or other polygons.
:p What is the definition of a surface in three-dimensional space?
??x
A surface in 3D space is a two-dimensional geometric entity that has a front and back but no inside or outside. It includes examples like planes, triangles, subdivision surfaces, and surfaces created from connected triangles or polygons.
x??

---
#### Collision SDK Support for Surfaces
Collision SDKs support both closed volumes (like polyhedra) and open surfaces. Surfaces can be given volume via an extrusion parameter to address the "bullet through paper" problem, which occurs with small fast-moving objects intersecting infinitesimally thin surfaces.
:p How do collision systems handle surfaces?
??x
Collision systems handle surfaces by extending the term "shape" to include both closed volumes (like polyhedra) and open surfaces. They allow surfaces to be given volume through an extrusion parameter, addressing issues like the "bullet through paper" problem where small fast-moving objects can intersect with very thin surfaces.
x??

---
#### Intersection in Collision Systems
Intersection refers to the set of all points that are common to two shapes. In practical terms, it means finding the overlapping region between two shapes.
:p What is intersection in collision systems?
??x
In collision systems, intersection refers to finding the common subset of points shared by two shapes. Practically, this involves identifying the overlapping area or region where both shapes intersect.
x??

---
#### Contact Information in Collision Systems
Contact information includes details about how objects are touching each other post-collision. This data helps in separating objects in a physically plausible and efficient manner. Key pieces include separating vectors to slide objects apart and information on which specific shapes were involved in the contact.
:p What does collision systems provide for contacts?
??x
Collision systems provide contact information such as separating vectors to slide objects apart efficiently and details about which shapes or features of those shapes are in contact. This helps in physically plausible separation of colliding objects.
x??

---
#### Collision Systems' Data Structures
Collision libraries often use convenient data structures, like `hkContactPoint` in Havok, for each detected contact. These structures store information such as separating vectors and details about the involved collidables and shapes.
:p How do collision systems manage contact information?
??x
Collision systems manage contact information using data structures that can be instantiated for each detected contact. For example, Havok uses `hkContactPoint` to store relevant information like separating vectors and details about the colliding objects and their shapes.
x??

---

#### Convex Shapes and Concave Shapes
Convex shapes are defined as those for which any ray originating inside the shape will pass through its surface only once. Concave (non-convex) shapes do not have this property, meaning a ray might pass through the surface multiple times.

A simple way to check if a shape is convex in 2D involves imagining shrink-wrapping it with plastic film—no air pockets should be left under the film for the shape to be considered convex. In three dimensions, similar logic applies, but now considering volumes instead of areas.
:p What is the definition of a convex shape?
??x
A convex shape is defined as one where any ray originating inside the shape will pass through its surface only once. This means that if you imagine shrink-wrapping the object with plastic film, there should be no air pockets left under the film.
x??

---

#### Collision Primitives
Collision detection systems often rely on basic shapes called collision primitives to simplify and optimize collision checks. These primitives serve as building blocks for more complex shapes.

Common types of collision primitives include spheres, capsules, and axis-aligned bounding boxes (AABBs).
:p What are collision primitives?
??x
Collision primitives are fundamental shapes used in collision detection systems to represent objects, simplifying the process and making it computationally efficient. They include basic shapes like spheres, capsules, and AABBs.
x??

---

#### Spheres as Collision Primitives
Spheres are one of the simplest three-dimensional volumes, representing a perfect round shape. They are particularly efficient for collision checks because their geometry is simple.

A sphere can be represented by a center point \((cx, cy, cz)\) and a radius \(r\). This information can be stored in a four-element floating-point vector.
:p How is a sphere defined as a collision primitive?
??x
A sphere is defined by a center point \((cx, cy, cz)\) and a radius \(r\). It can be represented using a four-element floating-point vector containing the coordinates of the center and the radius. This format works well with SIMD math libraries.
```java
class Sphere {
    float cx, cy, cz; // Center coordinates
    float r;         // Radius

    public void initialize(float x, float y, float z, float radius) {
        cx = x;
        cy = y;
        cz = z;
        r = radius;
    }
}
```
x??

---

#### Capsules as Collision Primitives
Capsules are pill-shaped volumes composed of a cylinder and two hemispherical end caps. They can be thought of as swept spheres, representing the shape traced out by a moving sphere.

A capsule is often represented using two points \((p1, p2)\) defining its endpoints and a radius \(r\).
:p What defines a capsule?
??x
A capsule is defined by two points \((p1, p2)\) which represent the endpoints of the cylinder part, and a radius \(r\) for the hemispherical end caps. It can be visualized as a swept sphere, representing the shape traced out by a moving sphere.
```java
class Capsule {
    Vector3 p1; // Start point
    Vector3 p2; // End point
    float r;    // Radius

    public void initialize(Vector3 start, Vector3 end, float radius) {
        p1 = start;
        p2 = end;
        this.r = radius;
    }
}
```
x??

---

#### Axis-Aligned Bounding Boxes (AABBs)
Axis-aligned bounding boxes (AABBs) are rectangular volumes whose faces are parallel to the axes of the coordinate system. They provide a simple and efficient way to approximate object shapes.

An AABB can be defined by two points: one containing the minimum coordinates along each axis, and another containing the maximum coordinates.
:p How is an Axis-Aligned Bounding Box (AABB) defined?
??x
An AABB is defined by two points: one containing the minimum coordinates \((minX, minY, minZ)\) and the other containing the maximum coordinates \((maxX, maxY, maxZ)\). This allows for efficient collision detection with other axis-aligned boxes.
```java
class Aabb {
    Vector3 min; // Minimum coordinates
    Vector3 max; // Maximum coordinates

    public void initialize(Vector3 minCoord, Vector3 maxCoord) {
        this.min = minCoord;
        this.max = maxCoord;
    }
}
```
x??

#### Oriented Bounding Boxes (OBB)
Background context: When an axis-aligned bounding box (AABB) is allowed to rotate relative to its coordinate system, it becomes known as an oriented bounding box (OBB). OBBs are represented using three half-dimensions and a transformation that positions the center of the box and defines its orientation. This representation allows for better fitting around arbitrarily oriented objects.
:p What is an Oriented Bounding Box (OBB)?
??x
An Oriented Bounding Box (OBB) is an axis-aligned bounding box that can rotate relative to its coordinate system. It uses three half-dimensions (half-width, half-depth, and half-height) along with a transformation matrix to position the center of the box and define its orientation.
??? 

#### Discrete Oriented Polytopes (DOP)
Background context: A discrete oriented polytope (DOP) is a more general case of an AABB and OBB. It approximates the shape of an object using a convex polytope. DOPs can be constructed by sliding planes at infinity along their normal vectors until they contact the object.
:p What is a Discrete Oriented Polytope (DOP)?
??x
A Discrete Oriented Polytope (DOP) is a convex polytope that approximates an object's shape. It is created by sliding planes parallel to the object’s surface along their normal vectors until they make contact with the object.
??? 

---

#### Arbitrary Convex Volumes
Background context: Collision engines often allow for arbitrary convex volumes, which are typically constructed using polygons (triangles or quads). These shapes can be converted into a collection of intersecting planes. If the shape fails the convexity test, it may still be represented as a polygon soup.
:p What is an Arbitrary Convex Volume?
??x
An Arbitrary Convex Volume is a complex geometric shape that can be constructed using polygons (triangles or quads). It can be represented by converting its triangles into a collection of planes and then representing these planes using plane equations or points and normal vectors.
??? 

---

#### Poly Soup
Background context: Some collision systems support totally arbitrary, non-convex shapes known as poly soups. These are typically constructed from triangles or other simple polygons. This type of shape is often used to model complex static geometry such as terrain and buildings.
:p What is a Poly Soup?
??x
A Poly Soup refers to an arbitrary, non-convex shape that can be constructed using triangles or other simple polygons. It is commonly used in collision systems for modeling complex static geometry like terrain and buildings.
??? 

---

These flashcards cover the key concepts of OBBs, DOPs, arbitrary convex volumes, and poly soups as described in the provided text. Each card provides a brief explanation and addresses a specific concept to aid understanding.

#### Poly Soup Overview
Background context explaining poly soup shapes. Unlike convex and simple shapes, a poly soup does not necessarily represent a volume—it can represent an open surface as well. This makes it difficult for collision systems to differentiate between a closed volume and an open surface when handling interpenetration issues.

:p What is a poly soup shape?
??x
A poly soup shape is a collection of polygons that do not necessarily form a closed, convex object. It is often used to model complex static surfaces such as terrain or buildings. Unlike convex shapes, the interior (if any) and exterior are not clearly defined without additional information.

```java
// Pseudocode for checking if a point is inside a poly soup
public boolean isPointInsidePolySoup(PolySoup shape, Point p) {
    // Logic to check if the point lies within the winding order of polygons in the poly soup
    return polygonContains(polygonWindingOrder(shape.getPolygons()), p);
}
```
x??

---

#### Inside and Outside for Poly Soup Shapes
Background context explaining how to define an inside and outside for a poly soup. This involves ensuring that all triangles have consistent vertex winding orders, which defines a front and back side of the shape.

:p How can you determine if a poly soup has an "inside" and "outside"?
??x
You can determine the "inside" and "outside" for a poly soup by ensuring all polygons’ vertex winding orders are consistent. If all triangles face in the same direction, the entire poly soup gains a notion of “front” and “back.” For closed shapes, you can interpret the front as the outside and the back as the inside.

```java
// Pseudocode for determining if a poly soup is open or closed
public boolean isOpenPolySoup(PolySoup shape) {
    // Check if vertex winding orders are consistent across all polygons
    return !hasConsistentWindingOrder(shape.getPolygons());
}

// Example of checking vertex winding order consistency
private boolean hasConsistentWindingOrder(List<Polygon> polygons) {
    int initialOrientation = getOrientation(polygons.get(0).getVertices());
    for (Polygon poly : polygons) {
        if (initialOrientation != getOrientation(poly.getVertices())) {
            return false;
        }
    }
    return true;
}
```
x??

---

#### Compound Shapes
Background context explaining compound shapes and their benefits. Compound shapes are collections of multiple simple shapes used to approximate complex objects, which can be more efficient than using a single poly soup for non-convex objects.

:p What is a compound shape?
??x
A compound shape is a collection of simpler geometric shapes (e.g., boxes) that together approximate a more complex object. This approach can be more efficient and accurate for modeling non-convex objects compared to using a single poly soup. Some collision systems leverage the convex bounding volumes of compound shapes to optimize collision detection.

```java
// Pseudocode for creating a compound shape from multiple simple shapes
public CompoundShape createCompoundShape(List<Shape> simpleShapes) {
    // Logic to combine multiple simple shapes into one compound shape
    return new CompoundShape(simpleShapes);
}

class CompoundShape {
    private List<Shape> subshapes;

    public CompoundShape(List<Shape> subshapes) {
        this.subshapes = subshapes;
    }

    // Method to get the convex bounding volume of a compound shape
    public BoundingVolume getConvexBoundingVolume() {
        return new BoundingVolume(subshapes.stream()
                .map(Shape::getConvexBoundingVolume)
                .reduce((bv1, bv2) -> bv1.intersect(bv2))
                .orElse(null));
    }
}
```
x??

---

#### Midphase Collision Detection
Background context explaining midphase collision detection in Havok. This involves using convex bounding volumes of compound shapes to quickly eliminate non-intersecting subshapes.

:p What is midphase collision detection?
??x
Midphase collision detection in Havok uses the convex bounding volumes of compound shapes to optimize collision tests. The system first checks if two compound shapes' convex bounding volumes intersect. If they do not, it skips testing for collisions between their subshapes, significantly reducing computational overhead.

```java
// Pseudocode for midphase collision detection
public boolean midPhaseCollisionDetection(CompoundShape shapeA, CompoundShape shapeB) {
    BoundingVolume bvA = shapeA.getConvexBoundingVolume();
    BoundingVolume bvB = shapeB.getConvexBoundingVolume();

    // Check if the convex bounding volumes intersect
    return !bvA.intersects(bvB);
}
```
x??

---

#### Collision Testing and Analytical Geometry
Background context explaining how collision systems use analytical geometry to detect intersections between shapes. This involves mathematical descriptions of three-dimensional volumes and surfaces.

:p What role does analytical geometry play in collision detection?
??x
Analytical geometry is crucial for collision detection as it provides a way to mathematically describe three-dimensional volumes and surfaces. Collision systems use these descriptions to compute intersections between shapes, enabling accurate and efficient collision detection.

```java
// Pseudocode for checking intersection using analytical geometry
public boolean checkIntersection(Sphere sphere1, Sphere sphere2) {
    // Calculate the distance between centers of the two spheres
    double distance = Math.sqrt(Math.pow(sphere1.center.x - sphere2.center.x, 2)
                                 + Math.pow(sphere1.center.y - sphere2.center.y, 2)
                                 + Math.pow(sphere1.center.z - sphere2.center.z, 2));

    // Check if the spheres intersect based on their radii
    return distance <= (sphere1.radius + sphere2.radius);
}
```
x??

#### Point versus Sphere

Background context: Determining whether a point lies within a sphere involves calculating the distance between the point and the center of the sphere. If this distance is less than or equal to the radius, the point is inside; otherwise, it's outside.

Relevant formula:
\[ s = c - p \]
\[ \text{if } |s| \leq r, \text{ then the point lies inside the sphere} \]

:p How do you determine if a point lies within a sphere?
??x
To determine if a point \( p \) lies within a sphere with center \( c \) and radius \( r \), calculate the vector from the center to the point:
\[ s = c - p \]
Then, check if the length of this vector is less than or equal to the radius:
\[ |s| \leq r \]
If true, the point lies inside the sphere; otherwise, it's outside.
x??

---

#### Sphere versus Sphere

Background context: Determining intersection between two spheres involves checking if the distance between their centers is smaller than or equal to the sum of their radii.

Relevant formula:
\[ s = c1 - c2 \]
\[ \text{if } |s| \leq (r1 + r2), \text{ then the spheres intersect} \]

:p How do you determine if two spheres intersect?
??x
To determine if two spheres with centers \( c1 \) and \( c2 \) and radii \( r1 \) and \( r2 \) intersect, calculate the vector between their centers:
\[ s = c1 - c2 \]
Then, check if the length of this vector is less than or equal to the sum of the spheres' radii:
\[ |s| \leq (r1 + r2) \]
If true, the spheres intersect; otherwise, they do not.
x??

---

#### Separating Axis Theorem

Background context: This theorem states that if a line can be found such that one object is entirely on one side of the line and another object is entirely on the other side, then the objects do not overlap. If no such line exists, and both shapes are convex, they must intersect.

Relevant formula:
For projection intervals \([c1min, c1max]\) and \([c2min, c2max]\):
\[ \text{if } [c1min, c1max] \cap [c2min, c2max] = \emptyset, \text{ then objects do not overlap} \]

:p How does the separating axis theorem work?
??x
The separating axis theorem states that if a line can be found such that one object is entirely on one side of this line and another object is entirely on the other side, they do not overlap. For convex shapes in 2D, we check projections along axes perpendicular to possible separating lines (or planes in 3D).

For example:
- In 2D: Check all edges of both objects as potential separating axes.
- In 3D: Check faces and edges of both objects.

If for any axis the projection intervals do not overlap (\([c1min, c1max] \cap [c2min, c2max] = \emptyset\)), no intersection exists. If all projections overlap, the shapes intersect.

Here's a simplified pseudocode example:
```java
public boolean separatingAxisTheorem(ConvexShape shape1, ConvexShape shape2) {
    for (Vector axis : getSeparatingAxes(shape1, shape2)) {
        Interval proj1 = projectOn(axis, shape1);
        Interval proj2 = projectOn(axis, shape2);
        if (!proj1.overlaps(proj2)) return false;
    }
    return true; // shapes intersect
}
```
x??

---
These flashcards cover the key concepts in the provided text. Each card is designed to help with understanding and application rather than pure memorization.

#### Separating Axis Theorem Overview
Background context: The separating axis theorem (SAT) is a method used to determine whether two convex shapes intersect. It involves projecting both shapes onto potential separating axes and checking if the projections overlap.

If any of the projection intervals are disjoint, then no separating axis exists, implying that the shapes do not intersect. If an axis separates them, it means they do not overlap along this direction.

Formula: The intervals \([cA_{\text{min}}, cA_{\text{max}}]\) and \([cB_{\text{min}}, cB_{\text{max}}]\) are disjoint if:
- \(cA_{\text{max}} < cB_{\text{min}}\) or 
- \(cB_{\text{max}} < cA_{\text{min}}\)

:p What is the separating axis theorem used for?
??x
The separating axis theorem (SAT) is used to determine whether two convex shapes intersect. It works by projecting both shapes onto potential separating axes and checking if their projections overlap.

If there exists a separating axis, the shapes do not intersect along that direction. If no such axis can be found, then the shapes are determined to be in collision.

```java
public class SATChecker {
    public boolean checkCollision(ShapeA shapeA, ShapeB shapeB) {
        // Get potential separating axes
        List<Vector2D> axes = getPotentialSeparatingAxes(shapeA, shapeB);
        
        for (Vector2D axis : axes) {
            double projectionAmin = project(shapeA, axis);
            double projectionAmax = project(shapeA, -axis);
            double projectionBmin = project(shapeB, axis);
            double projectionBmax = project(shapeB, -axis);
            
            if (!overlaps(projectionAmin, projectionAmax, projectionBmin, projectionBmax)) {
                return false; // No separating axis found
            }
        }
        
        return true; // All axes overlap, shapes intersect
    }

    private List<Vector2D> getPotentialSeparatingAxes(ShapeA shapeA, ShapeB shapeB) {
        // Implement logic to find potential separating axes
        // This could be vertices normals or other relevant vectors
    }

    private double project(Shape shape, Vector2D axis) {
        // Project the shape onto a given axis and return min and max values
    }

    private boolean overlaps(double a1, double a2, double b1, double b2) {
        // Check if intervals [a1, a2] and [b1, b2] overlap
        return !(a2 < b1 || b2 < a1);
    }
}
```
x??

---

#### Sphere-Sphere Collision Test
Background context: The sphere-sphere collision test is a specific application of the separating axis theorem. It determines whether two spheres intersect by checking if any axis (parallel to the line connecting their centers) separates them.

Formula: If no such separating axis exists, the spheres are considered intersecting.

:p How do you determine if two spheres collide using the separating axis theorem?
??x
To determine if two spheres collide using the separating axis theorem, check if the axis parallel to the vector connecting their center points is a separating axis. This means that the distance between the centers of the spheres should be less than or equal to the sum of their radii.

If this condition holds true for any such axis (there will typically only be one in 3D space), then the spheres are intersecting.

```java
public class SphereSphereCollision {
    public boolean checkCollision(Sphere sphere1, Sphere sphere2) {
        Vector3D centerDiff = subtract(sphere2.getCenter(), sphere1.getCenter());
        double distanceSquared = centerDiff.lengthSquared();
        
        double sumOfRadiiSquared = Math.pow(sphere1.getRadius() + sphere2.getRadius(), 2);
        
        // If the squared distance between centers is less than or equal to the sum of radii squared, they intersect
        return distanceSquared <= sumOfRadiiSquared;
    }

    private Vector3D subtract(Vector3D v1, Vector3D v2) {
        // Subtract vector v2 from v1 and return the result
    }
}
```
x??

---

#### Axis-Aligned Bounding Box (AABB) Collision Test
Background context: The axis-aligned bounding box (AABB) collision test uses the separating axis theorem to determine if two AABBs intersect by projecting both boxes onto each of three coordinate axes.

Formula: If projections overlap along all three axes, then the AABBs are intersecting. Otherwise, they do not.

:p How do you determine if two AABBs collide using the separating axis theorem?
??x
To determine if two AABBs collide using the separating axis theorem, project both boxes onto each of the x, y, and z coordinate axes independently. If projections overlap along all three axes, then the AABBs intersect; otherwise, they do not.

The logic involves checking intervals formed by the minimum and maximum coordinates of each box along each axis.

```java
public class AABBCollision {
    public boolean checkCollision(AABB aabb1, AABB aabb2) {
        // Check overlap on x-axis
        if (!overlap(aabb1.getXMin(), aabb1.getXMax(), aabb2.getXMin(), aabb2.getXMax())) return false;
        
        // Check overlap on y-axis
        if (!overlap(aabb1.getYMin(), aabb1.getYMax(), aabb2.getYMin(), aabb2.getYMax())) return false;
        
        // Check overlap on z-axis
        if (!overlap(aabb1.getZMin(), aabb1.getZMax(), aabb2.getZMin(), aabb2.getZMax())) return false;
        
        return true; // Overlap on all axes, AABBs intersect
    }

    private boolean overlap(double min1, double max1, double min2, double max2) {
        // Check if intervals [min1, max1] and [min2, max2] overlap
        return !(max1 < min2 || max2 < min1);
    }
}
```
x??

---

#### GJK Algorithm Overview
Background context: The GJK (Gilbert–Johnson–Keerthi) algorithm is an efficient method for detecting intersections between arbitrary convex polytopes in 2D or 3D space. It works by iteratively reducing the search volume until it finds either a separating hyperplane or determines that no such plane exists.

:p What is the GJK algorithm used for?
??x
The GJK algorithm is used to detect intersections between arbitrary convex polytopes, which can be two-dimensional polygons or three-dimensional polyhedra. It iteratively narrows down the search space by constructing a simplex (a point, line segment, triangle, tetrahedron) and determining if it contains the origin.

If the algorithm reaches the origin inside the Minkowski sum of the shapes, they are intersecting; otherwise, no intersection is found.

```java
public class GJKAlgorithm {
    public boolean checkCollision(ConvexPolyhedron poly1, ConvexPolyhedron poly2) {
        // Initialize a kernel with the origin point
        Kernel kernel = new Kernel(new Point3D(0, 0, 0));
        
        while (!kernel.isEmpty()) {
            Simplex simplex = kernel.getSimplex();
            
            if (simplex instanceof Point3D && containsOrigin((Point3D) simplex)) {
                return true; // Origin is inside the Minkowski sum
            }
            
            HalfSpace halfSpace = getSupportingHalfSpace(poly1, poly2, -simplex.getNormal());
            kernel = kernel.intersection(halfSpace);
        }
        
        return false; // No separating hyperplane found
    }

    private boolean containsOrigin(Point3D point) {
        // Check if the given point is within a small threshold of the origin
        double threshold = 1e-6;
        return Math.abs(point.getX()) < threshold && 
               Math.abs(point.getY()) < threshold &&
               Math.abs(point.getZ()) < threshold;
    }

    private HalfSpace getSupportingHalfSpace(ConvexPolyhedron poly1, ConvexPolyhedron poly2, Vector3D normal) {
        // Find the supporting hyperplane and return a half-space definition
    }
}
```
x??

---

#### Minkowski Difference
Background context explaining the concept. The Minkowski difference between two shapes A and B is defined as the set of all possible differences \(A_i - B_j\), where \(A_i\) is a point in shape A and \(B_j\) is a point in shape B.
:p What is the Minkowski difference?
??x
The Minkowski difference between two shapes A and B is the set of all points obtained by subtracting every point in B from every point in A. This operation results in a new set of points that, when applied to convex shapes, will contain the origin if and only if A and B intersect.
If we have two spheres A and B:
```java
public class Point {
    double x, y, z;
    
    // constructor, getters, setters, etc.
}

public class Sphere {
    Point center;
    double radius;

    public Set<Point> minkowskiDifference(Sphere other) {
        Set<Point> difference = new HashSet<>();
        for (double dx = -radius; dx <= radius; dx += 0.1) {
            for (double dy = -radius; dy <= radius; dy += 0.1) {
                for (double dz = -radius; dz <= radius; dz += 0.1) {
                    Point subPoint = new Point(center.x + dx, center.y + dy, center.z + dz);
                    difference.add(subtractPoints(subPoint, other.center));
                }
            }
        }
        return difference;
    }

    private Point subtractPoints(Point a, Point b) {
        return new Point(a.x - b.x, a.y - b.y, a.z - b.z);
    }
}
```
x??

---

#### GJK Algorithm Overview
Background context explaining the concept. The GJK algorithm uses the Minkowski difference to determine if two convex shapes intersect by constructing a simplex (a point, line segment, triangle, or tetrahedron) that encloses the origin within its convex hull.
:p What is the GJK algorithm?
??x
The GJK algorithm is an iterative method for determining whether two convex shapes intersect. It uses the Minkowski difference of the shapes and constructs a simplex (a point, line segment, triangle, or tetrahedron) to enclose the origin within its convex hull.
```java
public class Simplex {
    Point[] points; // Array of points that form the simplex

    public boolean containsOrigin() {
        // Logic to check if the origin is inside the simplex's convex hull
        return true; // Placeholder for actual implementation
    }
}

public class GJK {
    public boolean doShapesIntersect(Sphere A, Sphere B) {
        Simplex currentSimplex = new Simplex(); // Start with a single point
        
        while (!currentSimplex.containsOrigin()) {
            Point closestPoint = getClosestPointToOrigin(MinkowskiDifference(A, B)); // Find the closest point to origin in Minkowski difference
            currentSimplex.add(closestPoint); // Add this point to the simplex
            if (currentSimplex.getDimension() == 4) { // If we reach a tetrahedron
                return false; // No intersection
            }
        }
        return true; // Intersection found
    }

    private Point getClosestPointToOrigin(Set<Point> minkowskiDifference) {
        // Logic to find the closest point to origin in Minkowski difference
        return null; // Placeholder for actual implementation
    }
}
```
x??

---

#### Simplex and GJK Iteration
Background context explaining the concept. A simplex is a generalization of points into higher dimensions—points, line segments, triangles, and tetrahedrons are all simplices. The GJK algorithm iteratively builds a simplex starting from a single point until it either encloses the origin or determines that no intersection exists.
:p What is a simplex in the context of GJK?
??x
In the context of the GJK algorithm, a simplex is a geometric object used to approximate and eventually determine if two convex shapes intersect. It starts as a single point and can grow into line segments (1-simplex), triangles (2-simplices), or tetrahedrons (3-simplices) depending on the dimensionality.
```java
public class Simplex {
    Point[] points; // Array of points that form the simplex

    public int getDimension() {
        return points.length - 1; // Dimension is one less than number of points
    }

    public boolean containsOrigin() {
        // Logic to check if the origin is inside the simplex's convex hull
        return true; // Placeholder for actual implementation
    }

    public void add(Point point) {
        points = Arrays.copyOf(points, points.length + 1);
        points[points.length - 1] = point;
    }
}
```
x??

---

#### Intersection Determination with GJK
Background context explaining the concept. Once a simplex is constructed in GJK, it needs to check whether the origin lies within its convex hull. If it does, then the shapes intersect; otherwise, they do not.
:p How does GJK determine if two shapes intersect?
??x
GJK determines if two shapes intersect by constructing a simplex starting from the Minkowski difference of the shapes and checking if the origin is inside the simplex's convex hull. If the simplex encloses the origin at any point during the iteration, it indicates an intersection; otherwise, no intersection exists.
```java
public boolean doShapesIntersect(Sphere A, Sphere B) {
    Simplex currentSimplex = new Simplex(); // Start with a single point
    
    while (!currentSimplex.containsOrigin()) { // Iterate until we enclose the origin or determine no intersection
        Point closestPoint = getClosestPointToOrigin(MinkowskiDifference(A, B)); // Find the closest point to origin in Minkowski difference
        currentSimplex.add(closestPoint); // Add this point to the simplex
        
        if (currentSimplex.getDimension() == 4) { // If we reach a tetrahedron and still haven't enclosed origin
            return false; // No intersection
        }
    }
    return true; // Intersection found
}
```
x??

---

#### GJK Algorithm Overview
Background context: The GJK (Gilbert–Johnson–Keerthi) algorithm is a method used for collision detection between convex shapes. It works by iteratively constructing higher-order simplexes to determine if an origin point lies within the Minkowski difference of two shapes.

:p What is the purpose of the GJK algorithm?
??x
The GJK algorithm aims to determine whether two convex shapes intersect without explicitly computing their intersection points or surfaces, making it efficient for real-time applications such as video games and simulations.
x??

---

#### Constructing Simplexes in GJK
Background context: In each iteration of the GJK algorithm, a simplex is constructed that might potentially contain the origin. The simplex is expanded by adding new vertices until it surrounds the origin or until no closer vertex can be found.

:p What happens during each iteration of the GJK algorithm?
??x
During each iteration, the current simplex (a line segment, triangle, tetrahedron) is checked to see if it contains the origin. A supporting vertex in the direction opposite to the origin is then found and added to the simplex, forming a higher-order simplex. If adding this new point causes the simplex to surround the origin, the shapes are determined to intersect; otherwise, the algorithm terminates as no intersection exists.
x??

---

#### Finding Supporting Vertices
Background context: A supporting vertex is the closest point on the Minkowski difference in the direction opposite to the origin. This step is crucial for determining whether a higher-order simplex can surround the origin.

:p How does the GJK algorithm find a supporting vertex?
??x
The algorithm determines the direction relative to the current simplex and finds the supporting vertex, which is the closest point on the convex hull of the Minkowski difference in that direction. This involves checking the vertices of the current simplex and selecting the one that provides the shortest distance along the chosen search direction.
x??

---

#### Determining Intersection
Background context: Once a higher-order simplex surrounds the origin or no closer vertex can be found, the algorithm can determine whether the two shapes intersect.

:p How does GJK decide if the shapes intersect?
??x
If adding a new point to the current simplex causes it to surround the origin, then the two convex shapes are determined to intersect. Conversely, if no supporting vertex brings the simplex any closer to the origin, the shapes do not intersect.
x??

---

#### Simplexes and Their Orders
Background context: The GJK algorithm works with simplexes of different orders (points, line segments, triangles, tetrahedrons) as it iterates through the process.

:p What are simplexes in the context of GJK?
??x
Simplexes in the GJK algorithm refer to geometric shapes that represent vertices, edges, faces, and volumes in a hierarchical manner. They start with points and evolve into higher-order simplexes (line segments, triangles, tetrahedrons) as the algorithm progresses.
x??

---

#### Code Example for GJK Algorithm
Background context: Here is a simplified pseudocode to illustrate the logic of the GJK algorithm.

:p Provide an example of pseudocode for the GJK algorithm.
??x
```pseudocode
function gjkAlgorithm(shape1, shape2):
    simplex = [a vertex from shape1 - a vertex from shape2]
    
    while true:
        direction = -origin
        supportingVertex = findSupportingVertex(simplex, direction)
        
        if not supportingVertex:  // no closer vertex found
            return false
        
        simplex.add(supportingVertex)
        
        if isOriginInSimplex(simplex):  // surrounds the origin?
            return true

function findSupportingVertex(simplex, direction):
    // logic to find and return a closest vertex in 'direction'
    
function isOriginInSimplex(simplex):
    // logic to check if the simplex surrounds the origin
```
x??

---

#### Practical Considerations for GJK Algorithm
Background context: The complexity of collision detection increases with the number of shape types. The GJK algorithm simplifies this by handling all convex shapes in one step.

:p Why is the GJK algorithm preferred over other methods?
??x
The GJK algorithm is preferred because it can handle any pair of convex shapes efficiently, reducing the number of intersection cases a collision detector must handle. This makes it more practical for real-time applications where performance and simplicity are crucial.
x??

---

---
#### Double Dispatch in Collision Engines
Double dispatch is a design pattern used to handle operations between two objects by dispatching based on the types of both. Unlike single dispatch, where only one type (usually the receiver) determines the method to be called, double dispatch uses the types of both sender and receiver.

In the context of collision engines like Havok, this means that when checking for collisions between different agents (e.g., spheres, capsules), the engine needs to dynamically determine which specific intersection test should be used based on the types of the two collidables involved. This is crucial for efficiency and correctness in handling various shapes.

:p How does double dispatch work in collision engines?
??x
Double dispatch works by using a virtual function call mechanism that determines the appropriate method based on the types of both objects. Typically, this involves setting up a lookup table or arranging calls such that the type of one object (often called the receiver) and the other object (the sender) together determine which concrete implementation of the intersection test should be executed.

For example, in C++, you might define virtual functions for each agent type:
```cpp
class hkpCollisionAgent {
public:
    virtual void intersect(const hkpCollisionAgent& other) = 0;
};

class hkpSphereAgent : public hkpCollisionAgent {
public:
    void intersect(const hkpCollisionAgent& other) override {
        // Specific implementation for sphere-sphere intersection
    }
};
```
The `intersect` function is overridden in each concrete agent class, and the call to this function dispatches based on both the type of the calling object and the type of the receiver.

x??
---
#### Bullet Through Paper Problem (Tunneling)
In collision detection for moving objects, especially in games where movement is simulated discretely over time steps, small, fast-moving objects can cause issues. The problem arises because positions are considered stationary at each time step, leading to potential collisions being missed if an object moves faster than its own size per frame.

This issue is known as the "bullet through paper" or "tunneling" problem. It occurs when a moving object covers more space than its own size between two consecutive snapshots of the collision world, and another object lies in this "gap."

:p What is the bullet through paper problem?
??x
The bullet through paper problem (also known as tunneling) refers to a scenario where a fast-moving object does not detect collisions with other objects due to the discretization of movement. In simulation steps, if an object moves too far between frames relative to its size, it can pass through another object without detecting or resolving the collision.

For example, in a game frame, if an object is represented by a sphere and it moves more than its own radius during one frame, the static intersection tests at each snapshot may fail to detect that there was a collision with another object that entered the gap created by this fast movement.

x??
---
#### Swept Shapes for Collision Detection
To address the issue of tunneling in collision detection, swept shapes are used. A swept shape is a new shape formed by the motion of an existing shape from one point to another over time. This allows the engine to consider the path that a moving object takes rather than just its instantaneous position.

For instance, if an object moves along a straight line, its swept shape would be a capsule extending in both directions along this trajectory. Swept shapes enable more accurate collision detection by considering the entire motion of the object instead of static positions at individual time steps.

:p What are swept shapes used for?
??x
Swept shapes are used to improve collision detection accuracy, particularly when dealing with fast-moving objects that might otherwise pass through each other due to the discretization of movement over time. By creating a shape that represents an object's entire motion path (sweep) between two points in time, swept shapes help ensure that collisions are detected even if the object moves quickly.

For example, for a moving sphere, its swept shape could be represented as a capsule extending along its trajectory from the previous frame to the current frame. This allows collision detection algorithms to check for overlaps not just at discrete positions but also during the motion of the object.

x??
---

#### Swept Shapes and Collision Testing

Background context: Instead of testing static snapshots for intersections, we can test swept shapes formed by moving collidables from their previous to current positions and orientations. This approach approximates motion with linear interpolation, but it may not be accurate for fast-moving or rotating objects.

:p What is the advantage of using swept shapes in collision detection?
??x
The advantage lies in ensuring that collisions are detected even when they occur between static snapshots, thereby avoiding tunneling errors.
x??

---

#### Linear Interpolation and Curved Paths

Background context: Linearly interpolating motion may not accurately represent the true path followed by fast-moving or rotating objects. For a curved path, linear interpolation can lead to non-convex swept shapes, making collision tests complex and computationally intensive.

:p Why might linear interpolation be problematic for convex shapes?
??x
Linear interpolation of a convex shape along a curve results in a non-convex shape, which complicates collision detection algorithms.
x??

---

#### Linear Extrapolation vs. True Motion

Background context: To approximate the true motion, we can linearly extrapolate the extremal features from previous and current snapshots. However, this approximation may not accurately represent the actual movement during a time step.

:p What is the main limitation of using linear extrapolation for rotating objects?
??x
Linear extrapolation does not accurately capture the curved path followed by rotating objects, leading to potentially inaccurate collision detection.
x??

---

#### Continuous Collision Detection (CCD)

Background context: CCD aims to find the earliest time of impact between moving objects over a given time interval. This technique is particularly useful for avoiding tunneling errors in collision detection.

:p What does continuous collision detection (CCD) aim to achieve?
??x
CCD aims to identify the first moment when two collidables intersect, providing a more accurate representation of actual physical interactions.
x??

---

#### CCD Algorithm Overview

Background context: CCD algorithms typically involve maintaining both the previous and current positions and orientations of each collidable. These are used for linear interpolation of position and rotation.

:p What information is maintained for each collidable in a CCD algorithm?
??x
For each collidable, its position and orientation at the previous time step and its current position and orientation are maintained.
x??

---

#### Searching for Earliest TOI

Background context: After interpolating positions and rotations, an algorithm searches for the earliest time of impact (TOI) along the motion path. Various search algorithms can be used, such as conservative advancement or ray casting on Minkowski sums.

:p What is the goal of searching for the earliest TOI in a CCD algorithm?
??x
The goal is to find the first moment when two collidables intersect, ensuring accurate and timely detection of collisions.
x??

---

#### Conservative Advancement Method

Background context: One common method used in CCD algorithms is the conservative advancement method. It iteratively checks for potential intersections along the motion path.

:p Explain the basic idea behind the conservative advancement method.
??x
The conservative advancement method involves moving collidables gradually over time and checking for collisions at each step to find the earliest TOI without overshooting the actual intersection point.
x??

---

#### Minkowski Sum Method

Background context: Another approach in CCD is using ray casting on the Minkowski sum of the shapes. The Minkowski sum helps determine if two objects are close enough to potentially collide.

:p What does the Minkowski sum do in the context of CCD?
??x
The Minkowski sum combines two shapes by adding their vectors, helping to detect potential collisions between them by casting rays or using other geometric methods.
x??

---

#### Feature Pair Minimum TOI

Background context: Considering individual feature pairs (e.g., vertices and edges) can also be used to find the minimum TOI. This approach focuses on detecting interactions at specific points of contact.

:p How does considering individual feature pairs help in CCD?
??x
Considering individual feature pairs allows for a more granular search for collisions, potentially improving accuracy by focusing on critical points where actual impacts might occur.
x??

---

---
#### Temporal Coherency
When collidables are moving at reasonable speeds, their positions and orientations are usually quite similar from time step to time step. This allows us to avoid recalculating certain kinds of information every frame by caching results across multiple time steps.

Temporal coherency is an optimization technique used in collision detection where previous calculations can be reused if the motion of collidables has not changed significantly since the last update.
:p What is temporal coherency and how does it help in optimizing collision detection?
??x
Temporal coherency refers to the similarity in positions, orientations, or other properties of collidable objects between consecutive time steps. By leveraging this property, we can avoid redundant calculations that would otherwise be necessary for each frame update.

For instance, if a game object's position and orientation did not change significantly from one frame to another, certain aspects such as collision agents (hkpCollisionAgent in Havok) may remain valid across frames, reducing the computational overhead of recalculation.
```java
// Example pseudocode for checking temporal coherency
if (object.getNewPosition().isCloseTo(object.getLastPosition()) && 
    object.getNewOrientation().isCloseTo(object.getLastOrientation())) {
  // Reuse previous calculations
} else {
  // Recalculate necessary information
}
```
x?
---

#### Spatial Partitioning
Spatial partitioning is a technique used to reduce the number of intersection tests required by dividing space into smaller regions. By determining inexpensive ways to find that pairs of collidables do not occupy the same region, we can avoid more detailed intersection tests on those objects.

Hierarchical schemes like octrees, binary space partitioning trees (BSPs), kd-trees, or sphere trees are commonly used for spatial partitioning.
:p What is spatial partitioning and how does it work?
??x
Spatial partitioning involves dividing the game world into smaller regions to reduce the number of intersection tests needed. This is achieved by recursively subdividing space starting from a coarse level and refining it further.

For example, an octree divides 3D space into eight sub-regions at each level, allowing efficient spatial organization:
```java
// Pseudocode for creating an octree node
public class OctreeNode {
    private OctreeNode[] children; // Array of 8 child nodes

    public void addObject(Object obj) {
        if (isLeaf()) { // If this is a leaf node
            addObjectToNode(obj);
        } else { // Otherwise, recurse into the correct child
            int index = getCorrectChildIndex(obj.getPosition());
            children[index].addObject(obj);
        }
    }

    private void addObjectToNode(Object obj) {
        // Logic to add object to this node if it fits within its boundaries
    }

    private int getCorrectChildIndex(Vector3 pos) {
        // Determine which child node the position belongs in
        return ...
    }
}
```
x?
---

#### Broad Phase, Midphase and Narrow Phase
Havok uses a three-tiered approach to filter down the set of collidables that need to be tested for collisions during each time step. This approach is divided into broad phase, midphase, and narrow phase.

- **Broad Phase**: Filters out large sets of non-colliding objects using simple distance checks or bounding volume hierarchies.
- **Midphase**: Further reduces the set by performing more detailed intersection tests on pairs that passed the broad phase.
- **Narrow Phase**: Performs precise collision detection between remaining object pairs.
:p Describe Havok's three-tiered approach to collision filtering?
??x
Havok's three-tiered approach to collision filtering is designed to efficiently narrow down the number of potential collisions by dividing the process into three phases:

1. **Broad Phase**: Initially, it filters out large sets of non-colliding objects using simple distance checks or bounding volume hierarchies.
2. **Midphase**: Further reduces the set by performing more detailed intersection tests on pairs that passed the broad phase.
3. **Narrow Phase**: Finally, performs precise collision detection between remaining object pairs.

This hierarchical filtering helps to significantly reduce the computational load of detecting collisions in complex environments.
```java
// Pseudocode for Havok's three-tiered approach
public void detectCollisions() {
    // Broad Phase: Initial filtering using bounding volumes
    List<Collider> potentialColliders = broadPhaseFilter();

    // Midphase: More detailed checks on remaining candidates
    List<CollisionPair> midphaseCandidates = midPhaseFilter(potentialColliders);

    // Narrow Phase: Precise collision detection for final pairs
    List<CollisionResult> narrowPhaseResults = narrowPhaseDetection(midphaseCandidates);
}
```
x?
---

#### Broadphase Collision Detection
Background context: In collision detection, broadphase is used to reduce the number of narrowphase checks by identifying potentially intersecting collidables. It sorts and tests coarse bounding volumes (AABBs) to determine which objects need further detailed examination.

:p What is the role of broadphase in collision detection?
??x
Broadphase aims to quickly identify pairs of collidables that might be intersecting, thereby reducing the number of narrowphase checks required. This step uses techniques like sorting and testing bounding volumes.
x??

---

#### Midphase Collision Detection
Background context: After identifying potentially intersecting collidables in broadphase, midphase collision detection further narrows down these pairs by checking compound shapes and their bounding volume hierarchies.

:p What is the purpose of midphase collision detection?
??x
Midphase refines the list of potentially intersecting collidables from the broadphase stage. It traverses the bounding volume hierarchy to test sub-shapes, making it more efficient than checking all pairs.
x??

---

#### Narrowphase Collision Detection
Background context: Narrowphase checks the actual primitives (individual shapes) of collidables that have passed through both broadphase and midphase stages.

:p What is narrowphase collision detection?
??x
Narrowphase performs detailed collision tests on individual primitives to determine if they are intersecting. This step follows the broader filtering done in the previous phases.
x??

---

#### Sweep and Prune Algorithm
Background context: Sweep and prune is a broadphase algorithm used to efficiently check for overlapping AABBs by sorting them along axes and then testing for overlaps.

:p How does the sweep and prune algorithm work?
??x
The sweep and prune algorithm sorts collidables' bounding volumes (AABBs) along each axis, then checks for overlaps in the sorted lists. Frame coherency can reduce overhead when objects move slightly between frames.
```java
public void sweepAndPrune(ArrayList<AABB> collidables) {
    // Sort AABBs by their min and max bounds along each axis
    collidables.sort((a, b) -> Double.compare(a.getMinX(), b.getMinX()));
    
    for (AABB a : collidables) {
        for (AABB b : collidables) {
            if (a.intersects(b)) {
                // Check if AABBs overlap
                System.out.println("Overlap detected: " + a.getId() + " and " + b.getId());
            }
        }
    }
}
```
x??

---

#### Collision Queries
Background context: Collision queries are hypothetical questions answered by the collision detection system. They help in scenarios like determining possible targets for projectiles or checking paths for vehicles.

:p What is a collision query?
??x
A collision query uses collision detection to answer hypothetical questions, such as finding the first object hit by a ray or whether an object can move from one point to another without colliding.
x??

---

#### Ray Casting
Background context: A common type of collision query involves casting rays to determine if and where they intersect with collidables. This is used for tasks like firing projectiles.

:p What is a raycast in the context of collision queries?
??x
A raycast determines the first point (or points) at which a directed line segment intersects any collidable object, answering hypothetical questions about paths or trajectories.
```java
public Point castRay(Point start, Vector direction) {
    for (Collidable c : world.getCollidables()) {
        if (c.intersects(start, direction)) {
            return start.add(direction);
        }
    }
    return null;
}
```
x??

---

#### Parametric Equation of a Line Segment
Background context: Ray casting uses the parametric equation to find any point on the line segment defined by its start and end points.

:p What is the parametric equation for a ray?
??x
The parametric equation p(t) = p0 + td, where t ranges from 0 to 1, describes a point on a line segment starting at p0 and ending at p1. This allows finding any point along the ray.
```java
public Point getPointOnRay(double t) {
    return start.add(delta.multiply(t));
}
```
x??

#### Ray Cast Contact Data Structure
Background context: The provided text discusses ray casting, a technique used to determine contact points between rays and surfaces. The ray cast returns information about these contacts, including the t-value (the parameter value along the ray), an identifier for the collidable entity hit, and additional details like surface normals.
:p What is the structure of the `RayCastContact` data type?
??x
The `RayCastContact` struct contains fields to store the t-value, a unique identifier for the collidable entity hit, and possibly other relevant information.

```c++
struct RayCastContact {
    F32 m_t; // The t value for this contact.
    U32 m_collidableId; // Which collidable did we hit?
    Vector m_normal; // Surface normal at contact pt.
    // Other information...
};
```

x??

---

#### Applications of Ray Casting
Background context: Ray casting is widely used in various applications, including games and simulations. It helps determine visibility, collision detection, weapon hits, player mechanics, AI systems, and vehicle interactions.

:p How can ray casting be used to check if character A has a direct line of sight to character B?
??x
By casting a directed line segment from the eyes of character A to the chest of character B. If the ray hits character B, we know that A can see B. Otherwise, another object blocks the view.

```c++
bool isLineOfSightClear(const Point& eyeA, const Point& targetB) {
    Ray ray = createRay(eyeA, targetB);
    RayCastResult result = castRay(ray);
    return (result.m_t >= 0 && result.m_t <= distanceBetweenPoints(eyeA, targetB));
}
```

x??

---

#### Shape Casting
Background context: Another common query involves determining how far an imaginary convex shape would travel along a directed line segment before hitting something solid. This is known as a shape cast.

:p What is the purpose of a shape cast?
??x
The purpose of a shape cast is to determine the distance a given convex shape can travel in a specified direction without colliding with another object.

```c++
float shapeCastDistance(const Point& p0, const Vector& d, const Shape& shape) {
    Ray ray = createRay(p0, d);
    RayCastResult result = castRay(ray);
    
    if (result.m_t >= 0 && result.m_t <= distanceBetweenPoints(p0, p0 + d)) {
        return result.m_t;
    } else {
        return -1; // No collision
    }
}
```

x??

---

#### Interpenetrating vs. Non-interpenetrating Cast Shapes
Background context: When casting a convex shape, two scenarios can occur. The first is when the cast shape is already interpenetrating or contacting at least one other collidable, preventing it from moving away from its starting location. The second scenario involves the cast shape not intersecting with anything else at its starting location.

:p How does the collision system handle the case where a convex shape is initially interpenetrating?
??x
In this scenario, the collision system typically reports the contact(s) between the cast shape and all of the collidables with which it is initially interpenetrating. This helps in understanding the current state of the objects involved.

```c++
void handleInterpenetration(const Shape& shape, const std::vector<Collidable*>& collidables) {
    for (auto& collidable : collidables) {
        if (isShapeIntersecting(shape, *collidable)) {
            // Report contact with collidable
        }
    }
}
```

x??

---

#### Cast Shape Contacts Inside or On Surface

Background context: The text discusses how a cast shape can generate contact points inside or on its surface during collision detection. It explains that depending on the starting position and movement of the cast shape, these contacts might be located either within the shape or on its surface.

:p What are the possible locations for contact points generated by a cast shape?
??x
Contact points can lie both inside or on the surface of the cast shape.
x??

---

#### Movement of Cast Shape Before Hitting an Object

Background context: The text explains that a cast shape might move a nonzero distance along its line segment before it strikes something. If it hits, typically only one collidable is hit, but under certain conditions (like hitting a non-convex poly soup), multiple contact points may occur.

:p What is the movement behavior of a cast shape before it hits an object?
??x
The cast shape can move a nonzero distance along its line segment before striking something. If it hits another object, usually only one collidable is hit, but this can vary if the trajectory and shape are such that multiple contacts might occur.
x??

---

#### Multiple Contact Points with Cast Shape

Background context: The text states that no matter what kind of convex shape is cast, it is possible for the cast to generate multiple contact points. These contacts will always be on the surface of the cast shape because it was not interpenetrating anything when starting its journey.

:p How can a single convex shape cast result in multiple contact points?
??x
A single convex shape cast can result in multiple contact points if the trajectory is such that it touches more than one part of a non-convex poly soup simultaneously. However, contacts will always be on the surface of the cast shape.
x??

---

#### Reporting Multiple Contacts with Shape Casting

Background context: The text explains how some shape casting APIs report all the contacts experienced by the cast shape during its journey, while others might only return the earliest contact. Regardless, most APIs return both a tvalue and the actual contact point along with other relevant information.

:p How do shape casting APIs handle multiple contact points?
??x
Shape casting APIs typically return an array or list of contact point data structures, each containing the tvalue (location of center point), collidable ID, contact point location, and surface normal at the contact point. This allows reporting all contacts experienced during the cast's journey.
x??

---

#### Example Contact Data Structure

Background context: The text describes a possible structure for contact information returned by a shape casting API.

:p What is an example of a contact data structure in a shape casting system?
??x
An example of a contact data structure might look like this:

```cpp
struct ShapeCastContact {
    F32 m_t; // the t value for this contact
    U32 m_collidableId; // which collidable did we hit?
    Point m_contactPoint; // location of actual contact
    Vector m_normal; // surface normal at contact pt.
};
```
x??

---

#### Shape Casting vs. Ray Casting

Background context: The text contrasts ray casting and shape casting APIs, noting that while some may return only the earliest contact, others might report all contacts along the path.

:p How do shape casting APIs differ from ray casting APIs in terms of reporting contacts?
??x
Shape casting APIs must always be capable of reporting multiple contacts because even if they only report the contact with the earliest tvalue, the shape may have touched multiple distinct collidables or a single non-convex collidable at more than one point. In contrast, some ray casting APIs might return only the earliest contact.
x??

---

#### Distinguishing Groups of Contact Points by tvalue
Background context: In collision systems, contact points are often returned unsorted. To ensure that you always get the earliest contact point (i.e., the one with the minimum `t` value), it is essential to sort the results manually.

This helps in scenarios where multiple contacts occur at different times and you need to handle them sequentially. Sorting by `t` ensures that when you process the first contact, it will be among the earliest along the shape's path.
:p How can we ensure that we always get the earliest contact point from a list of contact points?
??x
To ensure that we get the earliest contact point, we need to sort the list of contact points by their `t` values. The first contact in the sorted list will be the one with the minimum `t`, which is the earliest contact.

Here's an example in Java:
```java
public class ContactPoint {
    double t; // time of impact

    public ContactPoint(double t) {
        this.t = t;
    }

    @Override
    public int compareTo(ContactPoint other) {
        return Double.compare(this.t, other.t);
    }
}

List<ContactPoint> contacts = ... // list of contact points
Collections.sort(contacts); // Sort by t value

// The first element in the sorted list is the earliest contact point.
ContactPoint earliestContact = contacts.get(0);
```
x??

#### Using Sphere and Capsule Casts for Character Movement
Background context: In game development, sphere or capsule casts are commonly used to implement character movement. These casts help determine where a character can move on uneven terrain.

For instance, you can cast a sphere or capsule between the character's feet in the direction of motion. Adjusting this cast vertically ensures it remains in contact with the ground, which helps in handling obstacles like curbs and walls.
:p How can we use a sphere or capsule cast to determine where a character should move on uneven terrain?
??x
To determine where a character should move on uneven terrain using a sphere or capsule cast, follow these steps:

1. **Cast a Sphere/Capsule**: Cast the shape between the character's feet in the direction of movement.
2. **Adjust Vertically**: Adjust the vertical position to ensure the shape remains in contact with the ground.
3. **Handle Obstacles**:
   - If the sphere hits a short vertical obstruction (like a curb), adjust it vertically until it no longer collides, effectively "popping up" over the obstacle.
   - If the sphere or capsule collides with a tall vertical obstruction (like a wall), slide it horizontally along the wall to find a new resting position.

The final resting place of the cast shape will be the character's new location for the next frame.

```java
public class CharacterMovement {
    private Vector3 startFeetPosition;
    private Vector3 direction;

    public void move(Character character) {
        // Cast a sphere or capsule between the feet and the direction of movement
        SphereCastResult result = castSphereOrCapsule(startFeetPosition, direction);

        if (result.collidesWithShortObstacle()) {
            adjustVertically(result);
        } else if (result.collidesWithTallObstacle()) {
            slideHorizontallyAlongWall(result);
        }

        // Update character's position based on the new resting place
        Vector3 newPosition = result.getRestingPosition();
        character.setPosition(newPosition);
    }

    private void adjustVertically(SphereCastResult result) {
        // Adjust vertical until no longer collides with a short obstacle
        while (result.collidesWithShortObstacle()) {
            startFeetPosition.y += 0.1f; // Increase y to move up slightly
            result = castSphereOrCapsule(startFeetPosition, direction);
        }
    }

    private void slideHorizontallyAlongWall(SphereCastResult result) {
        // Slide horizontally along the wall until no longer collides
        while (result.collidesWithTallObstacle()) {
            startFeetPosition.x += 0.1f; // Increase x to move right slightly
            result = castSphereOrCapsule(startFeetPosition, direction);
        }
    }

    private SphereCastResult castSphereOrCapsule(Vector3 startFeetPosition, Vector3 direction) {
        // Implementation of casting a sphere or capsule and checking for collisions
        return new SphereCastResult();
    }
}

class SphereCastResult {
    boolean collidesWithShortObstacle() { ... }
    boolean collidesWithTallObstacle() { ... }
    Vector3 getRestingPosition() { ... }
}
```
x??

#### Phantoms in Collision Systems
Background context: A phantom is a special kind of collidable object that acts like a shape cast with a zero-distance vector. It can be used to determine which collidable objects lie within a specific volume without affecting the dynamics simulation.

Phantoms are persistent, meaning they can take advantage of temporal coherency optimizations in collision detection between real collidables.
:p What is the purpose of using phantoms in game development?
??x
The primary purpose of using phantoms in game development is to determine which collidable objects lie within a specific volume without affecting the dynamics simulation. Phantoms act like shape casts but do not participate in the dynamics and are "invisible" to other real collidables.

Here's how you can use a phantom to find enemies within a certain radius of the player character:

1. **Create a Phantom**: Set up a phantom at the position of interest.
2. **Query Contacts**: Query the phantom for contacts with other collidables in the world.
3. **Process Results**: Process the contact results as needed.

```java
public class GameWorld {
    public List<Enemy> getEnemiesWithinRadius(Player player, float radius) {
        // Create a phantom at the player's position
        Phantom phantom = new Phantom(player.getPosition());

        // Query for contacts within the specified radius
        List<ContactPoint> contacts = phantom.getContactsWithinRadius(radius);

        // Filter out only enemy contacts
        List<Enemy> enemies = new ArrayList<>();
        for (ContactPoint contact : contacts) {
            if (contact.getCollidable() instanceof Enemy) {
                enemies.add((Enemy) contact.getCollidable());
            }
        }

        return enemies;
    }
}

class Phantom {
    Vector3 position;

    public Phantom(Vector3 position) {
        this.position = position;
    }

    public List<ContactPoint> getContactsWithinRadius(float radius) {
        // Implement the logic to query for contacts within a given radius
        return new ArrayList<>();
    }
}
```
x??

#### Closest Point Queries in Collision Engines
Background context: Some collision engines support closest point queries, which are used to find the set of points on other collidables that are closest to a given collidable.

This is useful for various purposes like finding the nearest enemy or determining the closest approach distance between two objects.
:p What is a closest point query and when might it be useful in game development?
??x
A closest point query is a type of collision engine query used to find the set of points on one collidable object that are closest to another collidable object. This is particularly useful for scenarios like finding the nearest enemy, determining the closest approach distance between two objects, or optimizing pathfinding.

Here’s an example in Java:

```java
public class ClosestPointQuery {
    public static Vector3 findClosestPoints(CollisionObject object1, CollisionObject object2) {
        // Implement the logic to find the closest points on object2 from object1
        return new Vector3(0.5f, 0.6f, 0.7f); // Example point
    }
}

class CollisionObject {
    // Representation of a collidable object in the game world
}

public class GameWorld {
    public List<Vector3> findNearestEnemies(Player player) {
        List<Vector3> closestPoints = new ArrayList<>();

        for (Enemy enemy : enemies) {
            Vector3 closestPoint = ClosestPointQuery.findClosestPoints(player, enemy);
            closestPoints.add(closestPoint);
        }

        return closestPoints;
    }
}
```
x??

#### Collision Filtering in Game Development
Background context: Game developers often want to enable or disable collisions between certain types of objects. For example, most objects might allow collision with the ground but not with each other.

This is achieved using collision filtering, which allows setting up rules for which collidable objects can interact.
:p How does collision filtering work in game development?
??x
Collision filtering works by allowing developers to define rules or masks that control which collidable objects are allowed to collide. This helps in scenarios where you want certain types of objects to ignore each other while still being able to detect collisions with specific types.

Here's a simple example in Java:

```java
public class CollisionFiltering {
    public static final int GROUND_MASK = 0x1;
    public static final int ENEMY_MASK = 0x2;
    public static final int PLAYER_MASK = 0x4;

    // Set the mask for an object to allow or disallow collisions with specific types
    public void setCollisionMask(CollisionObject obj, int mask) {
        obj.setCollisionMask(mask);
    }

    // Example of setting collision masks in a game world
    public GameWorld setupCollisionRules() {
        GameWorld world = new GameWorld();
        Player player = new Player();
        Ground ground = new Ground();

        // Allow player to collide with the ground but not other players or enemies
        setCollisionMask(player, GROUND_MASK);
        setCollisionMask(ground, GROUND_MASK);

        return world;
    }
}

class CollisionObject {
    int collisionMask;

    public void setCollisionMask(int mask) {
        this.collisionMask = mask;
    }

    public boolean shouldCollide(CollisionObject other) {
        return (this.collisionMask & other.collisionMask) != 0;
    }
}
```
x??
---

#### Collision Masking and Layers
Background context: In game development, collision masking and layers are common techniques used for managing how objects interact with each other. This is particularly useful when dealing with complex scenes where certain types of interactions should be allowed while others should be blocked.

Havok uses a specific implementation of this concept, where collidables can belong to one (and only one) collision layer. Each layer has a 32-bit mask that determines which layers can collide with it. This is done through the use of `hkpGroupFilter`, an instance that maintains these masks.

:p How does Havok handle categorization and filtering of collisions using layers?
??x
Havok categorizes objects into different collision layers, each represented by a 32-bit mask. The system checks the masks to determine if two collidables can interact based on their assigned layers. For instance, water might be in one layer while wood is in another; these layers would have specific bits set or cleared in their respective masks.
```java
// Pseudocode for setting up collision layers and masks
hkpGroupFilter filter;
filter.setLayerMask(0x1); // Set a bit for the first layer (e.g., water)
```
x??

---

#### Collision Callbacks
Background context: Another approach to managing collisions is through callbacks. These are functions that get invoked whenever a potential collision is detected, allowing developers to make real-time decisions based on specific criteria.

In Havok, there are two main callback methods: `contactPointAdded()` and `contactPointConfirmed()`. The former gets called when contact points are first added, while the latter confirms if a contact point is valid after further checks. This allows for dynamic decision-making during gameplay.

:p What are collision callbacks in game engines like Havok?
??x
Collision callbacks allow developers to inspect details of potential collisions and decide whether to accept or reject them based on custom criteria. In Havok, these include `contactPointAdded()` which is called when contact points are first added, and `contactPointConfirmed()` which validates if a point should be kept.

Example pseudocode for implementing a callback in C++:
```cpp
void myContactCallback(const hkpWorld*, hkpContactPoint* contact)
{
    // Logic to inspect and decide on collision based on specific criteria
    if (/* some condition */)
        contact->setValid(true);
    else
        contact->setValid(false);
}
```
x??

---

#### Game-Specific Collision Materials
Background context: To manage complex interactions between different types of surfaces, game developers often use a categorization mechanism for collision materials. This allows not only controlling how objects collide but also other secondary effects such as sound and particle effects.

Conceptually similar to material systems used in rendering engines, this categorization associates physical properties (like coefficient of restitution) with each collidable surface.

:p How do game-specific collision materials work?
??x
Game-specific collision materials allow developers to define behaviors for different types of surfaces. This includes defining how objects interact physically and generating secondary effects like sounds or particle systems upon impact.

Example in C++:
```cpp
// Pseudocode for setting a material with physical properties
hkpMaterial* mat = new hkpMaterial();
mat->setCoefficientOfRestitution(0.5); // Set the coefficient of restitution
```
x??

---


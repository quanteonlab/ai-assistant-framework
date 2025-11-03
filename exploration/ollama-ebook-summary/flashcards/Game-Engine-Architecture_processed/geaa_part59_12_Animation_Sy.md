# Flashcards: Game-Engine-Architecture_processed (Part 59)

**Starting Chapter:** 12 Animation Systems. 12.1 Types of Character Animation

---

#### Cel Animation and Sprite Animation
Background context explaining that cel animation is a precursor to modern game animation techniques, originally used for traditional hand-drawn cartoons. In 3D rendering, it's analogous to displaying a sequence of still images to create an illusion of motion.

Sprite animation is explained as the electronic equivalent where small bitmaps (sprites) are overlaid on top of background images without disrupting them, often utilized in 2D games.
:p What is cel animation?
??x
Cel animation refers to traditional hand-drawn animation where a sequence of still pictures known as frames are displayed in rapid succession to create an illusion of motion. This technique uses transparent sheets called cels on which the images can be painted or drawn, and these cels are layered over a fixed background without needing to redraw it repeatedly.

```java
// Pseudocode for cel animation logic
public class CelAnimation {
    private List<Frame> frames;

    public void addFrame(Frame frame) {
        frames.add(frame);
    }

    public void playAnimation() {
        // Display each frame in rapid succession
        for (Frame frame : frames) {
            display(frame.getImage());
            try {
                Thread.sleep(50);  // Simulate frame delay
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void display(Image image) {
        // Code to render the image on screen
    }
}
```
x??

---

#### Sprite Animation in Games
Background context explaining that sprite animation is a key technique for 2D games, allowing small bitmaps (sprites) to be overlaid on top of full-screen backgrounds without disrupting them. This was especially common during the 2D game era and continues to be used.

:p What are sprites?
??x
Sprites are small bitmaps used in 2D games that can be overlaid onto a background image, creating an illusion of motion or animation. They do not disrupt the static nature of the background, allowing for efficient reuse of graphics without constant redrawing.
```java
// Pseudocode for sprite overlay logic
public class Sprite {
    private Image bitmap;

    public Sprite(Image bitmap) {
        this.bitmap = bitmap;
    }

    public void drawOnBackground(Graphics2D g2d, int x, int y) {
        // Draw the sprite at specified position on the background image
        g2d.drawImage(bitmap, x, y, null);
    }
}

public class GameScreen {
    private Graphics2D graphics;

    public GameScreen(Graphics2D graphics) {
        this.graphics = graphics;
    }

    public void updateSpritePosition(Sprite sprite, int newX, int newY) {
        // Update the position of the sprite
        sprite.drawOnBackground(graphics, newX, newY);
    }
}
```
x??

---

#### Run Cycle Animation in Characters
Background context explaining that run cycle animations make characters appear to be running by looping a sequence of frames. This type of animation is common in many games where characters move fluidly.

:p What is a run cycle?
??x
A run cycle is an animation loop used to make characters appear as if they are running or moving fluidly. It consists of a series of frames that, when played in sequence, give the illusion of continuous motion. Run cycles are particularly useful for creating realistic and dynamic character movements.
```java
// Pseudocode for run cycle animation logic
public class RunCycle {
    private List<Image> frames;

    public void addFrame(Image frame) {
        frames.add(frame);
    }

    public void playRunCycle() {
        int index = 0;
        while (true) { // Loop indefinitely
            display(frames.get(index));
            try {
                Thread.sleep(50);  // Simulate delay between frames
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            index++;
            if (index >= frames.size()) {
                index = 0; // Reset to start for looping animation
            }
        }
    }

    private void display(Image image) {
        // Code to render the image on screen
    }
}
```
x??

---

#### Types of Character Animation in Games
Background context explaining that character animations have evolved significantly since early games like Donkey Kong. Today, game designers use various advanced techniques such as skeletal animation and blend shapes to create fluid and natural-looking movements.

:p What are the three most common types of character animation used today?
??x
The three most common types of character animation used in modern game engines are:
1. **Skeletal Animation**: This technique uses a骨骼结构来驱动角色的各个部分，使它们能够进行复杂和自然的动作。
2. **Blend Shapes**: 通过调整特定控制点（通常是面部），来改变角色模型的整体形状或姿态，实现更加细腻的表情和身体动作。
3. **Inverse Kinematics (IK)**: 使用逆运动学技术来解决关节约束问题，使角色的肢体能够自然地跟随目标移动。

这些方法通常结合使用以达到最佳效果。例如，在行走循环中，可能会用到骨骼动画，而在面部表情中可能采用blend shapes和IK。
x??

---

#### Looping Animation
Background context explaining that looping animations are used to create a continuous motion illusion by repeatedly playing the same sequence of frames. They are essential for animations like idle, walk, and run cycles.

:p What is a loop in animation?
??x
A loop in animation refers to the technique of repeating a sequence of frames or poses continuously to create an illusion of natural, ongoing movement. This is particularly useful for creating animations where characters remain static (idle loops), move at a steady pace (walk loops), or run (run cycles).

Example: A walk cycle loop ensures that the character’s movement appears smooth and continuous even when played repeatedly.
```java
// Pseudocode for looping animation logic
public class LoopAnimation {
    private List<Image> frames;

    public void addFrame(Image frame) {
        frames.add(frame);
    }

    public void playLoop() {
        int index = 0;
        while (true) { // Infinite loop to keep playing the animation
            display(frames.get(index));
            try {
                Thread.sleep(50);  // Simulate delay between frames
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            index++;
            if (index >= frames.size()) {
                index = 0; // Reset to start for looping animation
            }
        }
    }

    private void display(Image image) {
        // Code to render the image on screen
    }
}
```
x??

---
#### Rigid Hierarchical Animation
Background context: Early 3D games used a sprite-like system for character animation, but as graphics improved and more complex characters were needed, new techniques emerged. One of these is rigid hierarchical animation, where a character is modeled as a collection of rigid pieces (like parts of the body) that are connected in a hierarchy.

The hierarchical structure allows natural movement, such as when moving an arm, the hand moves with it due to the constraints. The pelvis serves as the root, with other body parts like torso, arms, and legs as its children.

:p What is rigid hierarchical animation?
??x
Rigid hierarchical animation models a character using rigid pieces connected in a hierarchy, allowing natural movement through constraints between parts of the body.
x??

---
#### Cracking at Joints
Background context: Rigid hierarchical animation can look unnatural due to "cracking" or abrupt changes in joint behavior. This is because each piece moves independently, leading to unrealistic motion.

:p What issue does rigid hierarchical animation face with joints?
??x
Rigid hierarchical animation faces the issue of "cracking," where joint behavior appears unnatural and can result in abrupt movements that do not mimic real-world flexibility.
x??

---
#### Per-Vertex Animation
Background context: To overcome the limitations of rigid hierarchical animation, per-vertex animation was introduced. This method allows individual vertices to be animated, enabling more natural deformation through runtime motion data.

:p What is per-vertex animation?
??x
Per-vertex animation involves animating each vertex individually and providing runtime motion data to the game engine to achieve more natural-looking deformations.
x??

---
#### Morph Target Animation
Background context: Per-vertex animation can be data-intensive, making it impractical for real-time applications. A variation called morph target animation addresses this by creating a set of fixed poses (motions) that are blended at runtime to produce animations.

:p What is morph target animation?
??x
Morph target animation involves defining a limited number of extreme poses for a mesh and blending them at runtime to create smooth animations.
x??

---
#### Application in Real-Time Games
Background context: Morph target animation strikes a balance between realism and performance by providing a manageable set of fixed poses that can be blended dynamically. This technique is often used in real-time games where high-quality, natural-looking motion is desired but data efficiency is crucial.

:p How does morph target animation work in real-time games?
??x
Morph target animation works by defining several extreme poses for a mesh and blending them at runtime to create smooth animations, balancing realism with performance.
x??

---

#### Morph Target Animation for Facial Expressions

Background context: In computer animation, especially for facial expressions, morph target animation is a technique that allows animators to manipulate each vertex of a 3D model's mesh independently. This method is particularly useful for the human face, which has around 50 muscles and requires precise control over every detail.

:p What is morph target animation used for in facial animation?
??x
Morph target animation is used to animate facial expressions by moving individual vertices on the surface of a 3D model. Each vertex can be moved to a position defined in an "extreme" pose, allowing for subtle or extreme movements that closely approximate real facial muscle movements.
x??

---

#### Jointed Facial Rigs as an Alternative

Background context: As computational power increases, some studios are using jointed rigs with hundreds of joints as an alternative to morph target animation. These rig-based systems allow more complex and detailed animations but can be computationally expensive.

:p How do some studios use jointed facial rigs?
??x
Some studios use jointed facial rigs containing many joints (hundreds) to achieve highly detailed and intricate facial movements, which cannot always be achieved with morph targets alone.
x??

---

#### Combining Jointed Rigs and Morph Targets

Background context: Other studios combine the techniques of using a jointed rig for primary poses and applying small tweaks via morph targets. This hybrid approach leverages the strengths of both methods.

:p How do some studios integrate jointed rigs and morph targets?
??x
Studios often use jointed facial rigs to establish the main pose or animation, then apply subtle adjustments through morph target animation. This combination provides a balance between detailed primary poses and small, precise tweaks.
x??

---

#### Skinned Animation Overview

Background context: Skinned animation is an advanced technique that offers many benefits of per-vertex and morph target animations while maintaining the efficiency of rigid hierarchical animation. It simulates the movement of skin and clothing by binding a mesh to bones.

:p What is skinned animation?
??x
Skinned animation is an advanced method in 3D animation where a smooth, continuous triangle mesh (the "skin") is bound to the joints of a skeleton. The vertices of this skin move according to the movements of the skeletal joints, allowing for realistic animations of skin and clothing.
x??

---

#### Construction of Skinned Animation

Background context: In skinned animation, a skeleton is built with rigid bones, similar to hierarchical rigging. However, unlike in traditional rigging where bones are rendered on-screen, they remain hidden while the skin mesh tracks their movements.

:p How does skinned animation differ from traditional hierarchical rigging?
??x
In skinned animation, the bone structure remains hidden and only affects the movement of a smooth continuous triangle mesh (the "skin"). In contrast, traditional hierarchical rigging involves rendering the bones on-screen to control the mesh directly.
x??

---

#### Weighting Vertices in Skinned Animation

Background context: Each vertex in a skin can be weighted to multiple joints. This allows for natural stretching and deforming of the skin as the joints move.

:p How do vertices in skinned animation work?
??x
Vertices in skinned animation are weighted to multiple joints, enabling the skin to deform naturally as the bones move. The weight distribution influences how each vertex moves relative to the influence of its associated joints.
x??

---

#### Trade-offs and Compression Methods

Background context: Skinned animation can be seen as a form of data compression where the motion of many vertices is constrained by a relatively small number of skeletal joints, reducing the amount of data needed for animation.

:p How does skinned animation relate to data compression in animation?
??x
Skinned animation serves as a method of data compression in animation. By constraining the movement of numerous vertices to follow the motions of a smaller set of skeletal joints, it reduces the complexity and memory usage while still achieving realistic animations.
x??

---

#### Example Character with Skinned Animation

Background context: The text provides an example of Crank the Weasel, a game character designed for Midway Home Entertainment in 2001. It has both visible skin mesh and hidden skeletal structure.

:p What does the internal structure of Crank the Weasel reveal?
??x
The internal structure of Crank the Weasel reveals that while its outer skin is made up of a mesh of triangles, there are also hidden rigid bones and joints responsible for animating the character.
x??

---

#### Joint and Bone Misnomer
Background context: In game development, terms "joint" and "bone" are often used interchangeably. However, technically speaking, joints are the objects manipulated directly by the animator, while bones are simply the empty spaces between these joints.

:p What is a joint in the context of skeletal animation?
??x
A joint is an object that can be directly manipulated by the animator. It acts as a pivot point or a node in the skeleton hierarchy.
x??

---

#### Skeleton Hierarchy
Background context: A skeleton forms a hierarchical structure where one joint serves as the root, and all other joints are its children, grandchildren, etc. This hierarchy is crucial for animating characters.

:p What defines the root joint in a skeletal hierarchy?
??x
The root joint has no parent; therefore, it is the starting point of the hierarchy. Its parent index is usually set to an invalid value like -1.
x??

---

#### Skeleton Data Structure
Background context: A skeleton is typically represented as a data structure that contains an array of joint data structures. Each joint stores information such as its name, parent index, and inverse bind pose.

:p What information does each joint data structure typically contain?
??x
Each joint data structure usually contains:
- The name of the joint (as a string or hashed 32-bit string id).
- The index of the joint’s parent within the skeleton.
- The inverse bind pose transform of the joint. This transformation records the position, orientation, and scale of the joint when it was bound to the vertices of the skin mesh.

Example code snippet:
```cpp
struct Joint {
    std::string name; // Name of the joint
    int parentIndex;  // Index of the parent within the skeleton
    Matrix4x3 m_invBindPose; // Inverse bind pose transform
};
```
x??

---

#### Joint Indices for Efficiency
Background context: Joint indices are used to reference joints in animation data structures, making it more efficient than referring by name. Indices can be as small as 8 bits wide, allowing up to 256 joints per skeleton.

:p How do joint indices improve efficiency in skeletal animation?
??x
Joint indices enhance efficiency because they allow for quick lookup of referenced joints. By using an index, we can jump directly to the desired joint in the array without needing to search through a list by name. This is particularly useful when dealing with large numbers of joints.
x??

---

#### Hierarchical Joint Example
Background context: The pelvis joint in the Crank the Weasel character model serves as an example of how one joint can appear to have multiple bones. The pelvis joint connects to four other joints (tail, spine, and two legs), but it is only a single joint.

:p How does the pelvis joint in Crank the Weasel illustrate the concept of "bones"?
??x
The pelvis joint in Crank the Weasel appears to have four bones sticking out of it because it connects to four other joints (tail, spine, and two legs). Despite being a single joint, it behaves as if there are multiple bones due to its connections. This is an example of how skeletal structures can present complex behaviors with relatively few actual joints.
x??

---

#### Game Engines vs Animators
Background context: In the game industry, "bones" are often misused terms by animators, but technically, they refer to the spaces between joints manipulated directly by the animator.

:p How do game engines view bones versus joints?
??x
Game engines care only about joints; "bones" are merely a term used by animators. In practice, when someone uses the term "bone," it usually refers to a joint.
x??

---

#### Summary of Key Concepts
Background context: Understanding the difference between joints and bones is crucial for efficient skeletal animation. Knowing how skeletons form hierarchies, and using joint indices effectively can significantly enhance performance.

:p What are the main points about joints and bones in skeletal animation?
??x
- Joints are directly manipulated by animators.
- Bones are empty spaces between joints.
- A skeleton forms a hierarchical structure with one root joint.
- Joint indices improve efficiency in referencing joints.
x??

---

---
#### Bind Pose Definition
Background context: In skeletal animation, a bind pose is a special pose of the skeleton that represents the initial or reference state before any transformation occurs. It's crucial for defining how vertices of a 3D mesh are attached to bones during the skinning process.

:p What is the bind pose in skeletal animation?
??x
The bind pose in skeletal animation is the initial position and orientation of the skeleton before any animations take place, which defines the attachment points for the mesh vertices. It's also sometimes referred to as the reference or rest pose.
x??

---
#### Local Poses Explanation
Background context: Local poses describe how each joint can be moved relative to its parent joint. These local changes allow for natural movement and are typically stored in SRT (Scale, Rotation, Translation) format.

:p What is a local pose?
??x
A local pose represents the transformation of a joint relative to its parent joint. It includes rotation, translation, and scale, but only as changes from the parent's state. This allows for natural movement within a hierarchical structure.
x??

---
#### Representing Joint Poses in Memory
Background context: Joint poses are commonly stored using SRT (Scale, Rotation, Translation) format to efficiently manage transformations.

:p How is a joint pose typically represented in memory?
??x
A joint pose is typically represented in memory as a data structure like this:
```c++
struct JointPose {
    Quaternion m_rot; // R
    Vector3 m_trans;  // T
    F32 m_scale;      // S (uniform scale only)
};
```
If non-uniform scaling is allowed, it might be represented as:
```c++
struct JointPose {
    Quaternion m_rot;   // R
    Vector4 m_trans;    // T
    Vector4 m_scale;    // S
};
```
x??

---
#### Global Poses Explanation
Background context: Global poses represent the absolute position and orientation of a joint in model space or world space, derived from walking up the skeletal hierarchy.

:p What is a global pose?
??x
A global pose represents the transformation of a joint in terms of its absolute position and orientation within the entire scene (model space or world space). It's calculated by combining local poses along the hierarchical path to the root.
x??

---
#### Calculating Global Poses Mathematically
Background context: The global pose of any joint can be mathematically derived by multiplying local poses from the current joint to the root.

:p How is a global pose calculated for a given joint?
??x
The global pose (j.M) of a joint j in model space or world space can be calculated as:
```c++
Pj.M = Pj.p(j) * Pp(j).0 * ... * P1.0 * p(0).M;
```
Where `Pj.p(j)` is the local pose of joint j, and `Pk.0` represents the global pose of its parent, with `p(0).M` being model space.
x??

---
#### Joint Pose as a Change of Basis
Background context: A joint pose can be viewed as transforming points from the child joint's coordinate system to the parent joint's.

:p How does a joint pose relate to change of basis?
??x
A joint pose transforms points and vectors from the local (child) space to the global (parent) space. The transformation can be expressed mathematically as:
```c++
Pj = (PC.P)j
```
Where `Pj` is the pose that takes a point or vector in child space (C) to parent space (P).
x??

---
#### Inverse Transformation for Global Poses
Background context: The inverse of a local joint pose can be used to transform points and vectors from the parent's coordinate system back into the child's.

:p What is the inverse transformation for global poses?
??x
The inverse transformation for a local joint pose is given by:
```c++
Pp(j).j = (Pj.p(j))^-1
```
This represents the transformation that converts points and vectors from parent space to child space.
x??

---


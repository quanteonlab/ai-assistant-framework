# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 27)

**Starting Chapter:** Inaccurate Comparisons

---

#### No Silver Bullet for Microservices
Background context: The passage discusses microservices and their limitations, emphasizing that they are not a one-size-fits-all solution. It highlights that while microservices offer flexibility and choice, they also bring increased complexity due to distributed systems challenges.

:p What is the main point about microservices mentioned in this text?
??x
Microservices are no silver bullet or free lunch; they require significant effort in deployment, testing, monitoring, scaling, and ensuring resilience. Additionally, they introduce complexities such as managing distributed transactions and understanding the CAP theorem.
x??

---

#### Architectural Evolution in Microservices
Background context: The passage discusses how microservices change the role of architects from a static, design-oriented position to a more dynamic, evolutionary one that focuses on guiding system evolution through rapid changes.

:p How does the role of an architect evolve with microservices?
??x
With microservices, the traditional static, design-focused architectural role transforms into an evolutionary architecture where architects must navigate and guide rapid changes in technology stacks, programming idioms, and service boundaries.
x??

---

#### Architect Role Redefined
Background context: The text critiques the term "architect" borrowed from other professions like engineering or medicine, pointing out that it often leads to misunderstandings about the architect's role and responsibilities. It highlights issues with certification and the expectations placed on architects.

:p Why is the term "architect" problematic in the context of software development?
??x
The term "architect" can be problematic because it implies a level of expertise and responsibility that may not accurately reflect the current state of the software industry. Architects are expected to have a deep understanding of complex systems, but the rapid changes and unique challenges of software development mean that these expectations are often unrealistic.
x??

---

#### Architect Responsibilities
Background context: The passage emphasizes the architect's role in ensuring a cohesive technical vision for the system while managing multiple teams and coordinating efforts across different geographical locations.

:p What is the primary responsibility of an architect according to this text?
??x
The primary responsibility of an architect is to ensure a joined-up technical vision that helps deliver the system required by customers, which involves working with or coordinating multiple teams and managing the overall architecture.
x??

---

#### Challenges in Defining Architect Role
Background context: The text discusses the difficulties in precisely defining the architect's role due to its varied responsibilities across different organizational levels. It also mentions societal misunderstandings about the profession.

:p Why is it challenging to define the architect's role accurately?
??x
Defining the architect's role accurately is challenging because it varies widely depending on the level of responsibility—whether working with one team or coordinating multiple teams globally. Additionally, society often borrows terms like "architect" and "engineer" without fully understanding their application in software development.
x??

---

#### Architect vs. Other Professions
Background context: The passage contrasts the architectural profession with its counterpart in engineering, highlighting differences such as the physical constraints faced by real architects versus the flexibility of software architectures.

:p How do the challenges faced by software architects compare to those faced by traditional architects?
??x
Software architects face fundamentally different challenges compared to traditional architects due to the dynamic and flexible nature of software systems. Unlike traditional architecture, which is constrained by physical rules like materials and gravity, software can adapt and evolve with changing user requirements.
x??

---

#### Conclusion on Microservices and Architect Role
Background context: The passage concludes that while microservices offer many benefits, their adoption requires careful consideration based on the specific needs of the organization. It emphasizes the importance of understanding potential pitfalls to navigate a steady path forward.

:p What is the key takeaway regarding microservices and architects according to this text?
??x
The key takeaway is that microservices are not a silver bullet and require significant architectural thought, including considering deployment, testing, monitoring, scaling, and resilience. Architects need to reassess their role in guiding system evolution with a focus on adaptability and rapid change.
x??

---

#### Evolutionary Thinking for Architects

Background context: The text discusses how architects for software systems need to shift their focus from creating perfect end products to designing frameworks that can evolve over time. This is compared to town planners, who do not specify every building but instead zone areas and plan for the flow of people and utilities.

:p What analogy does the author use to explain how architects should think about evolving software systems?
??x
The author uses the analogy of a town planner in SimCity. Just as a town planner zones different parts of a city (industrial, residential) without specifying every building, software architects should design frameworks that allow for flexibility and growth.
x??

---
#### Architect Role as Town Planner

Background context: The role of an IT architect is compared to that of a town planner who optimizes the layout of a city based on current and anticipated future needs. The focus is on creating zones rather than specifying every detail.

:p How does the role of an IT architect differ from traditional architecture?
??x
The role of an IT architect differs significantly because they do not create fixed, perfect end products but instead design frameworks that can evolve over time. They set broad goals and constraints (like zoning in a city) but allow developers to implement specific solutions within these guidelines.
x??

---
#### Zoning vs. Detailed Planning

Background context: The text emphasizes the importance of zoning areas for different purposes rather than detailed planning, allowing the system to adapt as needed.

:p Why does the author recommend focusing on zoning rather than detailed planning?
??x
The author recommends focusing on zoning because it allows the system to evolve naturally based on real-world usage. Detailed planning is often impractical given the unpredictability of future needs and changes in technology.
x??

---
#### Habitability for Developers

Background context: Architects must ensure that their systems are not only user-friendly but also developer-friendly, providing a good working environment for those who will maintain and extend the system.

:p What does the author mean by ensuring the system is "habitable" for developers?
??x
Ensuring the system is habitable for developers means creating an environment where they can work efficiently and effectively. This includes having clear structure, ease of maintenance, and tools that support development processes.
x??

---
#### Flexibility in Implementation

Background context: Architects should set broad goals but allow detailed implementation to be flexible based on user needs and evolving technology.

:p How should architects balance between setting direction and allowing flexibility?
??x
Architects should set broad strokes for the overall system architecture and constraints (like zoning in a city). However, they should avoid over-specifying details so that developers can adapt solutions as needed. This balance ensures both flexibility and coherence.
x??

---
#### Anticipation of Change

Background context: The text highlights the need to anticipate changes but not try to control every aspect of the system's evolution.

:p Why is it important for architects to plan for change?
??x
It is crucial because software systems must adapt to changing user needs, technologies, and business environments. Architects who can anticipate these changes are better positioned to guide the system's evolution without being overly rigid.
x??

---
#### User and Developer Happiness

Background context: The architect’s goal should be to create a system that satisfies both users and developers.

:p How do architects ensure satisfaction for both users and developers?
??x
Architects can achieve this by designing systems that are intuitive for users while providing the necessary flexibility for developers. This balance ensures that everyone involved finds the system valuable and easy to work with.
x??

---

#### Service Boundaries and Zones
In the context of architecting systems, we often think of our services as zones or coarse-grained groups. These boundaries help us manage complexity by separating concerns into manageable pieces. The focus for architects should be on how these zones communicate with each other rather than on internal details.
:p What are service boundaries in an architectural context?
??x
Service boundaries, or zones, refer to the separation of services into distinct, coarse-grained groups that allow for better management and scaling. Architects need to ensure proper communication protocols between these zones while giving teams autonomy within their zone.
x??

---

#### Microservices and Team Autonomy
Many organizations have adopted microservices to maximize team autonomy. This approach allows each team to make local decisions about technology stacks and data stores, promoting flexibility and innovation.
:p How do microservices support team autonomy?
??x
Microservices enable teams to independently choose their technology stack and data store based on the specific needs of their service. This autonomy can lead to better alignment with business goals and faster delivery times.
x??

---

#### Standardizing Technology Stacks
While allowing for diversity in zones, it's important to consider standardization across services. Large organizations like Netflix have standardized on a particular technology (e.g., Cassandra) to reduce operational complexity and training costs.
:p Why might an organization standardize its technology stack?
??x
Standardizing the technology stack can simplify operations by reducing the number of platforms that need to be managed, which in turn lowers operational overhead. It also helps in scaling teams more easily since they do not need to specialize in multiple technologies.
x??

---

#### Integration and Communication Protocols
When different services use different communication protocols (e.g., REST over HTTP vs. Protocol Buffers), integration becomes complex. A consistent approach can help streamline interactions between services.
:p What are the challenges of using diverse communication protocols between services?
??x
Using various communication protocols can lead to increased complexity in integrating services, making it harder for consuming services to handle multiple styles of data interchange. This can result in a tangled and hard-to-maintain system architecture.
x??

---

#### Pair Programming and Architectural Awareness
Architects should spend time coding with developers to ensure their decisions are practical and align with real-world development challenges. Pair programming is an effective method for architects to gain hands-on experience.
:p How can architects participate in pair programming?
??x
Architects can join teams as a partner during pair programming sessions, working on normal stories alongside team members. This direct interaction helps architects understand the day-to-day realities of development and improve communication with the team.
x??

---

#### Regular Team Engagement
Regularly spending time with teams is crucial for architects to stay informed about ongoing projects and challenges faced by developers. Frequent interactions can enhance collaboration and ensure that architectural decisions are well-considered.
:p How often should an architect spend time with a team?
??x
The frequency of engagement depends on the size of the team(s). For instance, spending half a day with each team every four weeks can help maintain awareness and improve communication. The key is to make such interactions routine.
x??

---


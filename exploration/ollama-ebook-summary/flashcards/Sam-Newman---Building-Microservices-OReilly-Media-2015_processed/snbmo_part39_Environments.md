# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 39)

**Starting Chapter:** Environments

---

#### Virtual Machine Images as Artifacts
Background context: The text discusses using virtual machine (VM) images to encapsulate dependencies and services, which speeds up deployment. Netflix has adopted this model for AWS AMIs. By baking service dependencies into the image, it abstracts away differences in technology stacks.
:p What is the advantage of creating VM images that contain all necessary dependencies?
??x
Creating VM images with all dependencies baked in ensures faster spin-up times and simplifies the process by abstracting away the underlying technology stack differences. This allows developers to focus on automating image creation and deployment, making the environment more consistent.
x??

---
#### Immutable Servers
Background context: The concept of immutable servers involves storing all configuration in source control and ensuring that no changes can be made directly to a running server. Changes must go through a build pipeline to create new machines, preventing configuration drift.
:p How does an immutable server approach help prevent configuration drift?
??x
Immutable servers help prevent configuration drift by enforcing that any change to the system goes through a controlled process (e.g., code changes in source control and a deployment pipeline). This means no manual changes can be made directly on running servers, ensuring that the production environment always reflects the state defined in version-controlled configurations.
x??

---
#### Environments and Deployment Pipelines
Background context: As software moves through different stages of the Continuous Delivery (CD) pipeline, it is deployed into various environments such as test, UAT, performance, and production. Each environment serves a specific purpose and can have distinct configurations and host setups.
:p Why do we need to consider multiple environments when deploying microservices?
??x
Multiple environments are necessary because each one serves a different purpose in the deployment lifecycle. For example, a test environment might be simpler with fewer hosts, while a production environment could require more complex setups like load balancing across data centers for durability. This variety helps catch issues early and ensures that the service behaves correctly under various conditions.
x??

---
#### Session State Replication Issue
Background context: The text describes an issue encountered when deploying Java web services into a clustered WebLogic application container, where session state replication failed due to changes in configuration between environments (test vs. production).
:p What lesson can be learned from the described issue with WebLogic session state?
??x
The key lesson is that environment differences can introduce subtle but critical issues, such as session state replication failure. To avoid these, it's essential to closely mirror the target environment during testing and ensure that all relevant configurations are consistent across different stages of deployment.
x??

---
#### Managing Environments for Microservices
Background context: The text highlights the complexity of managing multiple environments (e.g., test, UAT, performance, production) for microservices. Each environment can have distinct characteristics, requiring careful configuration management and balancing between fast feedback and production-like environments.
:p What is a key challenge in managing environments for microservices?
??x
A key challenge is balancing the need for environments that closely mimic production to catch issues early with the requirement for faster deployment cycles and shorter feedback loops. This balance can be complex, especially as the number of environments increases, requiring careful planning and automation.
x??

---

#### Service Configuration Management

Background context explaining the concept of managing service configurations for different environments. The key issue is to ensure that the configuration does not change fundamental service behavior and should be kept minimal. There are challenges with building artifacts per environment, as it can lead to deployment issues where the correct artifact may not always be used.

:p How should we handle configuration differences between environments to avoid problems?
??x
To manage configuration effectively, create a single artifact for all environments and keep configurations separate from the codebase. This could involve using properties files or parameterized install processes. Avoid building artifacts with configurations baked in because it can lead to deployment discrepancies where the wrong version of the software might be deployed.

```bash
# Example properties file (config.properties)
database.username=testuser
database.password=secret
```
x??

---

#### Service-to-Host Mapping

Background context explaining the concept of deploying services on hosts. The term "host" is used to refer to a unit of isolation, which could be a physical machine or a virtualized environment. The goal is to determine how many microservices should reside on a single host.

:p How do you decide how many microservices to deploy per host?
??x
The decision depends on various factors such as ease of management, cost efficiency, and the ability to scale independently. Deploying multiple services per host can simplify infrastructure management but may complicate monitoring, resource allocation, and deployment processes. Each service should ideally be deployed independently to maintain autonomy and avoid side effects.

```java
public class ServiceDeployer {
    public void deployServicesOnHost(String hostName, List<Service> services) {
        for (Service service : services) {
            service.deploy(hostName);
        }
    }
}
```
x??

---

#### Multiple Services Per Host

Background context explaining the benefits and challenges of deploying multiple microservices on a single host. This model is attractive due to simplicity in management and cost savings but can introduce complexities related to monitoring, resource allocation, and deployment.

:p What are the benefits of having multiple services per host?
??x
Benefits include:
1. **Simplified Host Management**: Reduces the number of hosts that need to be managed.
2. **Cost Efficiency**: Minimizes the overhead associated with virtualization.
3. **Familiarity for Developers**: Simplifies local development and deployment practices.

However, challenges include:
- **Monitoring Complexity**: Difficulty in isolating resource usage per service.
- **Resource Contention**: Load on one service can affect others running on the same host.
- **Impact Analysis**: Host failures can have broader ripple effects.
- **Deployment Complexity**: Ensuring one deployment does not affect another.

```java
public class HostManager {
    public void manageHostResources(String hostName) {
        // Logic to monitor and allocate resources for multiple services on a single host
    }
}
```
x??

---

#### Application Containers Overview
Background context explaining the concept of application containers. Multiple distinct services or applications can sit inside a single container, providing benefits such as clustering support and monitoring tools. However, this setup also has downsides including technology stack constraints and potential for slower spin-up times.
:p What are the key advantages of using application containers for deploying services?
??x
The key advantages include improved manageability through features like clustering and monitoring tools, and reduced overhead by sharing a single runtime environment across multiple applications. For example, running five Java services in one container means only one JVM is needed instead of five.
```java
// Example of setting up a simple embedded servlet container
public class EmbeddedServletContainer {
    public void start() {
        JettyServer jetty = new JettyServer();
        jetty.start();
        // More setup code here
    }
}
```
x??

---
#### Downsides of Application Containers
The text mentions several downsides to using application containers, such as technology stack constraints, limited automation and management options, slow spin-up times for applications, and added resource overhead.
:p What are some of the main drawbacks associated with application containers?
??x
Some key drawbacks include constrained technology choices, reduced flexibility in automation and management solutions, slower startup times for applications, and additional resource overhead due to their commercial nature or inherent complexity. For instance, embedded Jetty has faster spin-up times compared to full JVMs but still adds some overhead.
```java
// Example of a simple Jetty server start method
public class SimpleJettyServer {
    public void start() {
        // Initialize and start the Jetty server
        Server jetty = new Server();
        jetty.start();
    }
}
```
x??

---
#### Self-Contained Deployable Microservices
The text suggests considering self-contained deployable microservices as artifacts. Examples include using Nancy for .NET and embedded Jetty in Java, which can provide lightweight solutions with better scalability and faster startup times.
:p Why might self-contained microservices be preferred over application containers?
??x
Self-contained microservices offer more flexibility in technology choices and automation options, potentially faster spin-up times, and reduced resource overhead. They allow services to run independently without the constraints of a shared process, making them easier to manage and scale. For example, using embedded Jetty for serving static content demonstrates how lightweight solutions can operate at scale.
```java
// Example of embedding Jetty in a Java application
public class EmbeddedJettyApplication {
    public void start() {
        // Initialize and configure Jetty server
        Server jetty = new Server();
        HttpConfiguration httpConfig = new HttpConfiguration();
        // Start the server
        jetty.start();
    }
}
```
x??

---


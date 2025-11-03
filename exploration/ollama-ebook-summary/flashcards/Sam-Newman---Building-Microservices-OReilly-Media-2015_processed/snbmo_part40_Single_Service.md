# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 40)

**Starting Chapter:** Single Service Per Host

---

#### Single Service Per Host Model
Background context: The single service per host model aims to simplify monitoring, remediation, and scaling while reducing single points of failure. This model places one microservice on each host, ensuring that an outage in one host impacts only one service.
:p What is the primary benefit of using a single-service-per-host model?
??x
The primary benefits include simpler monitoring and remediation, reduced complexity in dealing with failures, easier scaling of individual services, and enhanced security management. Each host runs only one microservice, which simplifies troubleshooting and resource allocation.
x??

---
#### Reduced Complexity Through Single Service Per Host
Background context: Implementing a single-service-per-host model helps reduce the overall system's complexity by isolating each service on its own host. This isolation makes it easier to manage failures, scale services independently, and handle security concerns.
:p How does implementing a single-service-per-host help in reducing system complexity?
??x
Implementing a single-service-per-host reduces complexity because each microservice runs independently on its own host. This setup simplifies fault management as outages are isolated to individual services rather than affecting multiple services hosted on the same machine. It also eases scaling, as you can scale each service independently without impacting others.
x??

---
#### Deployment Techniques with Single Service Per Host
Background context: The single-service-per-host model opens up opportunities for using alternative deployment techniques such as image-based deployments or the immutable server pattern. These methods can further enhance flexibility and reliability in microservice architectures.
:p What are some advantages of using image-based deployments or the immutable server pattern with a single service per host?
??x
Using image-based deployments or the immutable server pattern with a single service per host allows for consistent, repeatable deployments that minimize configuration drift. This approach ensures that each host runs exactly the same version of the service, reducing the risk of bugs and inconsistencies across different environments.
```java
// Example of an Immutable Server Pattern in Pseudo-code
public class DeploymentManager {
    public void updateService(String serviceName) throws Exception {
        // Create a new image with updated code
        Image newImage = createNewImage(serviceName);
        
        // Stop the old service instance on the host
        stopServiceInstance(newImage.serviceName);
        
        // Start the new service instance using the updated image
        startServiceInstance(newImage.serviceName, newImage.imageId);
    }
}
```
x??

---
#### Increased Number of Hosts and Management Challenges
Background context: While the single-service-per-host model offers numerous benefits, it also introduces challenges such as increased management overhead and potential cost implications. More hosts mean more servers to manage, which can increase operational complexity.
:p What are some downsides of having a high number of hosts in a microservice architecture?
??x
The downsides include higher server management costs, more complex monitoring and maintenance, and increased risk associated with managing multiple systems. Each additional host adds to the overall operational overhead, requiring more resources for setup, maintenance, and security.
x??

---
#### Preferred Model for Microservices
Background context: The single-service-per-host model is favored because it simplifies the architecture by isolating services, making troubleshooting easier and reducing the risk of inter-service conflicts. However, it's acknowledged that this approach may not be feasible in all cases due to resource constraints or existing infrastructure.
:p Why is the single service per host model preferred for microservices?
??x
The single service per host model is preferred because it isolates each service, making it easier to manage and scale independently. This setup also simplifies fault isolation and enhances security by focusing on a single service at a time. However, its implementation may depend on the availability of suitable infrastructure and resources.
x??

---

#### Platform as a Service (PaaS) Overview
PaaS allows users to work at a higher-level abstraction, focusing on deploying applications rather than managing infrastructure. Most PaaS solutions rely on taking technology-specific artifacts and automatically provisioning them for you. Some platforms handle scaling transparently, but more often they provide control over the number of nodes while handling other aspects.
:p What is Platform as a Service (PaaS)?
??x
PaaS provides a platform to deploy applications without managing infrastructure. It abstracts away many details, allowing developers to focus on their application logic rather than deployment and scaling issues. Examples include Heroku for Java WAR files or Ruby gems.
```java
// Example of deploying an artifact using PaaS (pseudo-code)
public class DeployApplication {
    public void deployArtifact(String artifactPath) {
        // Code to upload and run the artifact
        System.out.println("Deploying application: " + artifactPath);
    }
}
```
x??

---

#### Hosted vs. Self-Hosted PaaS Solutions
Hosted PaaS solutions are fully managed by a third party, offering services like automatic scaling and database management. However, they can be less flexible when your application requirements deviate from the standard use cases. Self-hosted solutions exist but are generally less mature.
:p What are the main differences between hosted and self-hosted PaaS?
??x
Hosted PaaS is fully managed by a third party, providing scalable services out of the box. Self-hosted PaaS requires more manual configuration and maintenance. Hosted options like Heroku offer easy setup but may not fit nonstandard applications as well.
```java
// Example of choosing between hosted and self-hosted (pseudo-code)
public class PaaSChoice {
    public String choosePaaS(String applicationType) {
        if ("nonStandard".equals(applicationType)) {
            return "Self-Hosted";
        } else {
            return "Heroku";
        }
    }
}
```
x??

---

#### Automation in DevOps
Automation is crucial for managing and deploying applications as the number of services increases. Manual management becomes impractical with multiple servers and services, but automation can keep overhead constant regardless of the number of hosts.
:p Why is automation important in modern software development?
??x Automation is essential because it reduces manual intervention, which minimizes human error and allows scaling without increasing work proportionally. With more moving parts (services), automation ensures that deployment, monitoring, and management tasks are handled efficiently.
```java
// Example of an automated deployment script (pseudo-code)
public class AutomatedDeployment {
    public void deployServices(List<Service> services) {
        for (Service service : services) {
            System.out.println("Deploying " + service.getName());
            // Code to handle actual deployment
        }
    }
}
```
x??

---

#### Managing Services with PaaS
PaaS solutions like Heroku can simplify the process of deploying applications, including database management. However, they may not always fit every application's needs due to their one-size-fits-all approach.
:p How do PaaS solutions typically handle services and databases?
??x PaaS platforms often provide pre-configured environments for running your application, such as Heroku which supports Java WAR files or Ruby gems directly. They also offer managed database services. However, these solutions may not be ideal if your application has unique requirements that deviate from the standard use cases.
```java
// Example of deploying a service on Heroku (pseudo-code)
public class DeployOnHeroku {
    public void deployApplication(String appName) {
        System.out.println("Deploying " + appName + " to Heroku");
        // Code for actual deployment
    }
}
```
x??

---

#### Overhead in Multi-Service Environments
In environments with many services, managing each service individually can become cumbersome. Automation tools help manage the overhead, making it possible to scale and maintain a large number of services without increasing manual effort.
:p What is the main challenge in managing multiple services?
??x The primary challenge in managing multiple services is the increased administrative burden as more services are added. Manual management can lead to inefficiencies and errors, but automation tools help keep this overhead constant by handling tasks like deployment, scaling, and monitoring.
```java
// Example of managing a list of services with automation (pseudo-code)
public class ServiceManager {
    public void manageServices(List<Service> services) {
        for (Service service : services) {
            System.out.println("Managing " + service.getName());
            // Code to handle management tasks like deployment and scaling
        }
    }
}
```
x??

#### Multiple Deployments and Services Management
Background context explaining how managing multiple deployments, services, and logs can become complex without automation. Automation helps ensure developers remain productive by allowing self-service provisioning of individual or groups of services.

:p How does automation contribute to keeping the complexities of microservice architectures in check?
??x
Automation contributes by enabling tasks such as launching virtual machines, deploying code, and managing database changes automatically. This reduces manual intervention, freeing up developer time for other tasks. For example, using tools like Docker and Kubernetes can help automate container orchestration and service deployment.

```bash
# Example of starting a VM with Bash script
#!/bin/bash
echo "Starting VM..."
# Logic to start the virtual machine here
```
x??

---

#### Developer Self-Service Provisioning
Context explaining the importance of developers having access to the same tool chain used in production, ensuring early detection of issues. This involves giving developers tools to provision and manage their services independently.

:p What is the key benefit of allowing developers to self-provision services using the same toolchain as production?
??x
The key benefit is that it ensures consistency between development and production environments, reducing discrepancies and potential bugs. Developers can quickly set up and tear down environments as needed, accelerating development cycles and improving productivity.

```bash
# Example Bash script for provisioning a service
#!/bin/bash
echo "Provisioning Service..."
# Logic to provision the necessary services here, e.g., deploying code, setting up databases, etc.
```
x??

---

#### Case Study: RealEstate.com.au (REA)
Context describing how RealEstate.com.au moved towards microservices and experienced significant productivity gains through automation. It took them 18 months to move from two services to over 60-70 services.

:p How did RealEstate.com.au benefit from implementing an automated toolchain?
??x
RealEstate.com.au benefited by reducing the initial setup time for new microservices, allowing a rapid increase in service count. By automating deployment, monitoring, and support processes, they were able to scale their development efforts more efficiently.

In the first 18 months, RealEstate moved from 2 services to over 60-70 services, with developers responsible for the entire lifecycle of these services. Automation tools helped in provisioning machines, deploying code, and monitoring services seamlessly.

```bash
# Example Bash script for service deployment at REA
#!/bin/bash
echo "Deploying Service..."
# Logic to deploy new microservices here
```
x??

---

#### Case Study: Gilt Online Fashion Retailer
Context explaining how Gilt transitioned from a monolithic architecture to a microservice-based system, leveraging automation extensively. By 2014, Gilt had over 450 services.

:p What role did automation play in the success of Gilt's move to microservices?
??x
Automation played a crucial role by enabling frequent and reliable deployments without manual intervention. This allowed Gilt to scale their service count significantly while maintaining quality and reliability. As the number of microservices grew, automation tools helped manage complexity, ensuring that developers could focus on code rather than infrastructure.

By 2014, Gilt had over 450 services, with a ratio of about three services per developer. Automation tools such as CI/CD pipelines and orchestration platforms were key in supporting this growth.

```bash
# Example CI/CD script for Gilt's microservices deployment
#!/bin/bash
echo "Starting CI/CD Pipeline..."
# Logic to trigger continuous integration and deployment here
```
x??

---

#### Traditional Virtualization and Host Management
Context discussing the benefits of traditional virtualization tools like VMware or AWS in reducing host management overhead. This helps in efficiently managing a large number of hosts.

:p How do virtualization tools help manage multiple physical machines?
??x
Virtualization tools, such as those provided by VMware or AWS, enable chunking up existing physical machines into smaller parts, thereby reducing the overhead associated with managing each machine individually. This approach enhances resource utilization and simplifies host management tasks like provisioning, scaling, and monitoring.

For example:
```bash
# Example Bash script to create a virtual machine using AWS CLI
#!/bin/bash
echo "Creating VM on AWS..."
aws ec2 run-instances --image-id ami-0abcdef1234567890 --count 1 --instance-type t2.micro
```
x??

---

#### Physical Host Limitations
Background context explaining why having lots of physical hosts can be expensive. The overhead comes from needing a separate host for each service, making it challenging to scale efficiently.
:p Why is using multiple physical hosts per service expensive?
??x
Using multiple physical hosts per service increases costs because each service requires its own dedicated hardware. This leads to underutilized resources and higher expenses due to the need for more physical infrastructure.
x??

---

#### Virtualization Overview
Background context explaining how virtualization allows slicing a single physical server into multiple virtual machines (VMs), but at the cost of overhead.
:p What is the main idea behind using virtualization in microservice architectures?
??x
Virtualization enables running multiple services on a single physical host by creating separate VMs. However, this comes with additional overhead due to the hypervisor managing these VMs, which can impact performance and resource utilization.
x??

---

#### Type 2 Virtualization vs Lightweight Containers
Background context comparing traditional type 2 virtualization (e.g., AWS, VMware) with lightweight container technologies like Docker in terms of overhead and efficiency.
:p What are the key differences between type 2 virtualization and lightweight containers?
??x
Type 2 virtualization involves running a hypervisor on top of an existing host OS, which introduces significant overhead as the hypervisor manages resources. In contrast, lightweight containers like Docker run directly on the host OS without an additional layer of abstraction, reducing overhead.
x??

---

#### Hypervisor Overhead in Type 2 Virtualization
Background context explaining that the hypervisor in type 2 virtualization sets aside resources for its own operations, thereby reducing available resources for VMs. This overhead becomes a constraint when trying to slice physical infrastructure into smaller parts.
:p How does the hypervisor's role in managing VMs affect resource allocation?
??x
The hypervisor's job of mapping resources from VMs to the host and controlling those VMs takes up valuable CPU, I/O, and memory resources. As more VMs are created, this overhead increases, limiting the total number of VMs that can be run efficiently.
x??

---

#### Physical Drawer Analogy
Background context using a physical drawer analogy to explain why adding dividers (virtualization layers) reduces overall storage capacity (resource efficiency).
:p Why is it not possible to store more socks by adding more dividers in your drawer?
??x
Adding more dividers in the drawer increases organizational complexity but decreases the total storage space available for socks. Similarly, adding more VMs or virtualization layers on a physical host consumes resources that could otherwise be used for running services.
x??

---

#### Resource Allocation in Virtualization
Background context illustrating how resource allocation is managed between the hypervisor and VMs, leading to diminishing returns when slicing hosts into smaller parts.
:p How does resource allocation work in type 2 virtualization?
??x
In type 2 virtualization, the hypervisor runs on top of a host OS and manages resources like CPU and memory for each VM. As more VMs are created, the hypervisor takes up more resources, reducing the available capacity for running services efficiently.
x??

---

#### Vagrant Overview
Vagrant provides a platform for creating and managing virtual machines (VMs) on your local machine, using tools like VirtualBox. The configuration of VMs is defined in a text file that can be checked into version control systems and shared among team members.
:p What does Vagrant provide?
??x
Vagrant provides an environment to create and manage VMs locally, leveraging virtualization technologies such as VirtualBox. Configurations are stored in a text file that can be shared across the development team, ensuring everyone has an identical setup.
x??

---

#### Running Multiple VMs with Vagrant
With Vagrant, you can define multiple VMs in a single configuration file and manage them as a group. This allows for easier setup of complex systems on your local machine and enables testing of failure modes by shutting down individual VMs.
:p How does managing multiple VMs with Vagrant simplify development?
??x
Managing multiple VMs with Vagrant simplifies development by allowing you to define all required services in a single configuration file. This ensures consistency across the team and makes it easy to spin up or shut down specific services for testing different failure modes.
x??

---

#### Linux Containers Introduction
Linux containers are an alternative to full virtualization, offering faster startup times due to not needing a hypervisor. They share the same kernel but can run different operating system distributions as long as they use the same kernel version.
:p What is the key difference between virtual machines and Linux containers?
??x
The key difference lies in their underlying architecture: VMs require a hypervisor to segment hardware resources, while containers leverage process space within the host OS to isolate processes. This means containers share the same kernel but can run different distributions of operating systems.
x??

---

#### Advantages of Linux Containers
Linux containers offer faster startup times and better resource utilization compared to full virtual machines. They are more lightweight, allowing for many more instances running on the same hardware, and provide finer-grained control over resources.
:p What advantages do Linux containers have over full VMs?
??x
Linux containers are advantageous because they start much faster (seconds vs. minutes), utilize fewer system resources, and allow for more instances to run on the same hardware due to their lightweight nature. They also offer better resource allocation control compared to traditional virtualization.
x??

---

#### Isolation in Linux Containers
While Linux containers provide process isolation, there are known vulnerabilities that can allow processes from one container to interact with others or the host system. This requires careful consideration of security and code reliability when using containers.
:p What potential issue should be considered when using Linux containers?
??x
A potential issue is that despite providing some level of isolation, Linux containers can still have vulnerabilities allowing processes in one container to interact with others or the underlying host. This necessitates thorough testing and reliable coding practices for secure use.
x??

---


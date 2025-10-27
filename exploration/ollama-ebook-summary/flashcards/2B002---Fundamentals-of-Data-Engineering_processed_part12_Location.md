# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 12)

**Starting Chapter:** Location

---

#### Hive's Legacy and Evolution
Hive was initially successful but had shortcomings that led to the development of other technologies like Presto. Over time, Hive is now primarily used in legacy deployments due to these advancements.

:p What technology did engineers develop to improve upon Hive's limitations?
??x
Engineers developed Presto to address the shortcomings of Hive.
x??

---
#### Technology Lifecycles and Decline
Technologies often follow a lifecycle where they decline over time. Hive is an example of this, being replaced by newer technologies like Presto.

:p What happens to most data technologies over their lifecycle?
??x
Most data technologies decline over time and are eventually replaced by newer or improved versions.
x??

---
#### Evaluating Data Technologies
It's recommended to evaluate data technologies every two years due to the rapid pace of changes in tools and best practices. This helps ensure that you're using the most up-to-date solutions.

:p How often should one reevaluate their technology choices according to this advice?
??x
One should reevaluate technology choices every two years.
x??

---
#### Immutable Technologies
In data engineering, it's advisable to identify immutable technologies as a base and build transitory tools around them. This provides stability in the long run.

:p What type of technologies are recommended as the base in data engineering lifecycle?
??x
Immutable technologies should be used as the base in the data engineering lifecycle.
x??

---
#### Transitioning Technologies
Consider how easy it is to transition from a chosen technology, especially given the high probability of failure for many data technologies. Evaluate the barriers to leaving the current solution.

:p What factors should one consider when choosing and using data technologies?
??x
Factors include evaluating the ease of transitioning and considering the barriers to switching technologies due to their potential failure.
x??

---
#### Opportunity Cost in Technology Choices
Avoid making technology choices based solely on immediate benefits, as this could lead to higher opportunity costs over time.

:p What concept is emphasized regarding technology choices?
??x
The emphasis is on avoiding technology choices that may have high opportunity costs in the long term.
x??

---

---
#### On Premises Systems
Background context explaining on-premises systems, including hardware ownership and management responsibilities. Companies are operationally responsible for their hardware and software, needing to manage upgrades and ensure sufficient capacity during peak times.

If applicable, add code examples with explanations of how data engineers handle resource allocation.
:p What is the key responsibility of companies running technology stacks on premises?
??x
Companies on premises must be operationally responsible for their hardware and software. They need to manage hardware failures by repairing or replacing them, handle upgrade cycles every few years as new hardware is released, and ensure enough capacity during peak times such as Black Friday.

For data engineers, this means buying large-enough systems to allow good performance without overbuying and overspending.
??x
The key responsibilities include managing hardware failure, maintaining upgrades, ensuring sufficient capacity for peak loads, and balancing resource allocation to avoid underutilization or overprovisioning.
```
public class OnPremisesDataEngineer {
    public void allocateResources(int peakLoad) {
        int requiredCapacity = calculatePeakLoad(peakLoad);
        ensureSufficientHardware(requiredCapacity);
    }

    private int calculatePeakLoad(int peakLoad) {
        // Logic to determine the necessary capacity based on peak load
        return (int)(peakLoad * 1.5); // Example calculation
    }

    private void ensureSufficientHardware(int requiredCapacity) {
        // Code to purchase or upgrade hardware as needed
    }
}
```
x?
---

#### Cloud Hosting
Background context explaining the cloud hosting model, including the shift from on-premises systems and the benefits of renting managed services from providers like AWS, Azure, or Google Cloud. Companies can scale rapidly without upfront investment in hardware.

:p What is a key benefit of using cloud hosting over traditional on-premises systems?
??x
A key benefit of using cloud hosting is the ability to rent hardware and managed services from cloud providers such as AWS, Azure, or GCP. This allows companies to scale rapidly without the need for upfront investment in hardware, providing flexibility and cost savings.

Companies can also take advantage of cloud-managed services and DevOps practices like containers, Kubernetes, microservices, and continuous deployment.
??x
The key benefit is rapid scalability, reduced initial costs, and access to managed services. This allows companies to focus on their core business rather than hardware management.
```
public class CloudUser {
    public void scaleResources(int requiredCapacity) {
        cloudProvider.allocateResources(requiredCapacity);
    }

    private ICloudProvider cloudProvider = new AWSProvider(); // Example provider
}
```
x?
---

#### Hybrid Cloud and Multicloud Strategies
Background context explaining the hybrid cloud (combining on-premises with cloud resources) and multicloud (using multiple cloud providers). These strategies allow companies to leverage the benefits of both models while managing costs and risks.

:p What is a benefit of using a hybrid cloud strategy?
??x
A benefit of using a hybrid cloud strategy is that it allows companies to leverage the best aspects of both on-premises systems and cloud resources. This can include using local infrastructure for sensitive data or compliance requirements, while utilizing the scalability and cost-effectiveness of the public cloud for other workloads.

It also provides greater flexibility in managing costs and risks by spreading them across different environments.
??x
The key benefit is balancing security, compliance, and cost by using on-premises resources for critical needs and leveraging the public cloud for more flexible and scalable workloads. This strategy helps manage costs and mitigate risks associated with over-provisioning or underutilization of resources.
```java
public class HybridCloudStrategy {
    private OnPremisesDataCenter dataCenter;
    private CloudProvider cloudProvider;

    public void balanceWorkloads() {
        if (dataCenter.isCriticalWorkload()) {
            useOnPremises(dataCenter.getRequiredResources());
        } else {
            useCloud(cloudProvider.allocateResources(getRequiredCapacity()));
        }
    }

    private void useOnPremises(int requiredResources) {
        dataCenter.provisionResources(requiredResources);
    }

    private void useCloud(int requiredResources) {
        cloudProvider.allocateResources(requiredResources);
    }
}
```
x?
---

#### Competitive Pressures and Cloud Migration
Background context explaining the competitive pressures faced by companies to migrate to the cloud or adopt new technologies. Established companies must keep their existing systems running while making decisions about future technology stacks, balancing the risks of technological failure with the benefits of agility.

:p What is a key challenge for established companies in the face of competitive pressures?
??x
A key challenge for established companies in the face of competitive pressures is deciding whether to migrate their technology stacks entirely to the cloud or keep on-premises systems while adopting new technologies. They must balance the risks of technological failure, high costs associated with poor planning, and the threat of being left behind by more agile competition.

Companies need to evaluate their existing operational practices, consider the benefits of cloud-managed services, and decide how to scale rapidly without disrupting current operations.
??x
The key challenge is balancing existing infrastructure management with the need for technological advancement and agility. Companies must weigh risks such as high costs from poor planning against the potential for greater efficiency and cost savings through cloud adoption.

```java
public class CTODecisionMaker {
    private int currentTechRisk;
    private double projectedCostSavings;

    public void evaluateMigration() {
        if (currentTechRisk > threshold && projectedCostSavings >= minimumSavings) {
            migrateToCloud();
        } else {
            maintainOnPremises();
        }
    }

    private void migrateToCloud() {
        // Code to plan and execute cloud migration
    }

    private void maintainOnPremises() {
        // Code to continue running on-premises systems while adopting new technologies
    }
}
```
x?
---

#### VMs and Short-Term Resource Reservation
VMs can be quickly spun up, typically within a minute. Subsequent usage is billed per second. This allows for dynamic scaling of resources that were previously impractical with on-premises servers.

:p How does cloud computing enable quick resource provisioning?
??x
Cloud providers offer virtual machines (VMs) that can be launched almost instantaneously. These VMs are essentially rented slices of hardware, providing users the ability to scale up or down based on demand without worrying about upfront hardware investments and setup times. The billing model is typically per-second, making it cost-effective for short-term projects.

```java
// Pseudocode for launching a VM in a cloud environment
public void launchVM(String vmType) {
    CloudProvider provider = new AWS();
    VM vm = provider.createVM(vmType);
    vm.start();
}
```
x??

---

#### Dynamic Scaling and Cloud Computing
Dynamic scaling allows businesses to handle increased loads during peak times, such as Black Friday for retail. This is particularly valuable in industries experiencing seasonal or unpredictable traffic.

:p How does dynamic scaling benefit businesses?
??x
Dynamic scaling helps businesses manage variability in load more efficiently by automatically adjusting the number of active resources based on demand. For instance, a retailer can scale up their server capacity during peak shopping times (like Black Friday) and scale down when the load decreases to reduce operational costs.

```java
// Pseudocode for dynamic scaling using cloud services
public void adjustResources(int requiredInstances) {
    CloudProvider provider = new AWS();
    List<Instance> instances = provider.listInstances();
    
    if (requiredInstances > instances.size()) {
        int newInstancesNeeded = requiredInstances - instances.size();
        for (int i = 0; i < newInstancesNeeded; i++) {
            VM vm = provider.createVM("type");
            vm.start();
        }
    } else if (requiredInstances < instances.size()) {
        List<Instance> toShutdown = instances.subList(requiredInstances, instances.size());
        for (Instance instance : toShutdown) {
            instance.stop();
        }
    }
}
```
x??

---

#### Shift from IaaS to PaaS
The evolution of cloud services has moved from infrastructure as a service (IaaS) offerings like VMs and virtual disks to platform as a service (PaaS). PaaS adds managed services for application support, such as databases and Kubernetes.

:p What is the difference between IaaS and PaaS?
??x
Infrastructure as a Service (IaaS) provides users with virtualized computing resources over the internet. The user has control over the operating system, storage, and deployment of applications. Platform as a Service (PaaS), on the other hand, extends beyond just providing hardware. It includes managed services that enable developers to build, deploy, and manage their applications without worrying about the underlying infrastructure.

```java
// Pseudocode for deploying an application on a PaaS platform
public void deployApplication(String appName) {
    CloudProvider provider = new AWS();
    Application app = new Application(appName);
    
    // Assuming there are managed services like RDS or Kubernetes available
    ManagedService dbService = provider.getManagedService("RDS");
    ManagedService k8sService = provider.getManagedService("Kubernetes");
    
    dbService.deployDatabase(app);
    k8sService.deployApplication(app);
}
```
x??

---

#### SaaS Offerings and Examples
Software as a Service (SaaS) offers fully functioning enterprise software platforms with minimal operational management. Examples include Salesforce, Google Workspace, Microsoft 365, and Zoom.

:p What are some examples of SaaS products?
??x
Some common examples of Software as a Service (SaaS) products include:

- **Salesforce**: A CRM system for managing customer relationships.
- **Google Workspace**: An office suite including Gmail, Google Drive, and Google Meet.
- **Microsoft 365**: Office applications like Word, Excel, PowerPoint, along with cloud storage and collaboration tools.
- **Zoom**: Video conferencing software.

These services are fully managed by the provider, allowing users to focus on their core business activities without worrying about server management or updates.

```java
// Pseudocode for integrating a SaaS product into an application
public void integrateWithSaaS(String saasProductName) {
    CloudProvider provider = new AWS();
    
    if (saasProductName.equals("Salesforce")) {
        SalesforceClient client = provider.getSalesforceClient();
        client.authenticateUser("username", "password");
        client.createLead("customerName", "emailAddress");
    } else if (saasProductName.equals("Google Workspace")) {
        GoogleWorkspaceClient client = provider.getGoogleWorkspaceClient();
        client.sendEmail("from@example.com", "to@example.com", "subject", "body");
    }
}
```
x??

---

#### Serverless Computing
Serverless computing allows for automated scaling and pay-as-you-go billing. It abstracts away the underlying server details, enabling engineers to focus on writing code without worrying about infrastructure management.

:p What is serverless computing?
??x
Serverless computing refers to a type of cloud computing where developers are not required to provision or manage servers. The cloud provider automatically allocates resources based on demand and bills users only for the time that their functions run. This model enables quick deployment, maintenance-free operation, and cost-effective execution.

```java
// Pseudocode for serverless function invocation
public void invokeServerlessFunction(String functionName) {
    CloudProvider provider = new AWS();
    
    // Assuming there is a managed service like AWS Lambda available
    ManagedService lambdaService = provider.getManagedService("Lambda");
    
    FunctionInvocationResponse response = lambdaService.invoke(functionName, "eventData");
    System.out.println(response.getResult());
}
```
x??

---

#### FinOps and Cloud Pricing Models
Migration to the cloud often requires a shift in financial practices. Enterprises need to adapt their budgeting and cost management strategies to align with the pay-as-you-go model of cloud services.

:p What is FinOps?
??x
FinOps, or Financial Operations, refers to the practice of applying financial principles and disciplines to cloud computing environments. It involves optimizing cloud spending by aligning budgeting practices with actual usage patterns and costs. This requires close monitoring of resource utilization, cost control, and strategic planning to maximize efficiency.

```java
// Pseudocode for basic FinOps implementation in a cloud environment
public void manageCloudBudget(double budget) {
    CloudProvider provider = new AWS();
    
    // Monitor resource usage
    List<ResourceUsage> usages = provider.monitorResourceUsages();
    
    double totalCost = 0;
    for (ResourceUsage usage : usages) {
        totalCost += usage.getCost();
    }
    
    if (totalCost > budget) {
        System.out.println("Exceeded budget: " + (totalCost - budget));
    } else {
        System.out.println("Within budget");
    }
}
```
x??

#### Credit Default Swaps and Cloud Services Analogy
Cloud services can be compared to financial derivatives, where cloud providers slice hardware assets into smaller pieces with varying technical characteristics and risks. This analogy helps understand how cloud providers monetize different aspects of their services.

:p How are cloud services similar to credit default swaps?
??x
In the context of cloud services, cloud providers offer different tiers of risk by slicing hardware resources such as storage and computing power into various configurations. These slices can be sold with varying performance characteristics (like IOPs, bandwidth) and reliability levels, analogous to how financial derivatives transfer different levels of risk.

For example, consider a scenario where a provider sells "gold" class storage for critical data at higher prices but also offers cheaper "silver" or "bronze" classes that are less performant but still reliable. This differentiation allows the provider to charge more for premium services while offering cost-effective options.
??x
This analogy helps in understanding how providers can optimize their pricing and resource allocation by selling different risk profiles, similar to financial derivatives.

---
#### Archival Storage Pricing and Optimization
Archival storage is an example where cloud providers sell cheaper storage space with lower performance characteristics. Providers often sell empty disk space as a cost-effective solution for long-term data storage needs.

:p Why might archival storage be significantly cheaper than standard storage?
??x
Archival storage can be cheaper because it typically offers reduced performance features such as slower IOPs and bandwidth, but the underlying hardware remains the same as that used for standard storage. Providers can maximize their utilization by offering lower-cost options that still serve the needs of users who do not require high-performance access.

For example:
```java
class StorageTier {
    private String name;
    private double pricePerGB; // in $/month per GB
    private int iops;
    private int bandwidthMbps;

    public StorageTier(String name, double pricePerGB, int iops, int bandwidthMbps) {
        this.name = name;
        this.pricePerGB = pricePerGB;
        this.iops = iops;
        this.bandwidthMbps = bandwidthMbps;
    }

    // getters and setters
}

// Example of creating tiers
StorageTier archival = new StorageTier("Archival", 0.005, 10, 5);
StorageTier standard = new StorageTier("Standard", 0.0833333, 100, 200);

System.out.println(archival.pricePerGB / standard.pricePerGB); // prints approximately 0.06
```
??x
This example illustrates how providers can offer cheaper storage options by reducing the performance metrics such as IOPs and bandwidth without significantly increasing hardware costs.

---
#### Cloud ≠ On-Premises Servers
Moving on-premises servers to the cloud does not simply replicate existing infrastructure; there are significant differences in cost models, resource management, and utilization that users must understand. These differences often lead to higher bills if not managed correctly during a migration.

:p Why can moving from on-premises to the cloud be more expensive initially?
??x
Moving servers from an on-premises environment to the cloud does not simply replicate existing infrastructure; it involves changes in cost models and resource management that users must understand. On-premises hardware is often seen as a commodity, but cloud services are priced based on various factors like durability, reliability, longevity, and predictability.

For example:
```java
class OnPremiseServer {
    private double cpuCores;
    private int memoryGB;

    public OnPremiseServer(double cpuCores, int memoryGB) {
        this.cpuCores = cpuCores;
        this.memoryGB = memoryGB;
    }

    // getters and setters
}

class CloudInstance {
    private double cpuCores;
    private int memoryGB;
    private double durabilityFactor; // e.g., 1.5 for higher reliability, 0.8 for lower predictability

    public CloudInstance(double cpuCores, int memoryGB, double durabilityFactor) {
        this.cpuCores = cpuCores;
        this.memoryGB = memoryGB;
        this.durabilityFactor = durabilityFactor;
    }

    // method to calculate cost
    public double calculateCost() {
        return (cpuCores * 0.1 + memoryGB * 0.05) * durabilityFactor; // simplified pricing model
    }
}

// Example of usage
OnPremiseServer onPremise = new OnPremiseServer(2, 8);
CloudInstance cloudInstance = new CloudInstance(2, 8, 1.2);

System.out.println("On-Premise Cost: " + onPremise.calculateCost());
System.out.println("Cloud Instance Cost: " + cloudInstance.calculateCost());
```
??x
This example demonstrates how the cost of maintaining an on-premises server and a cloud instance can differ significantly due to factors like reliability, predictability, and resource utilization. Cloud instances may be more expensive initially due to additional overhead in terms of service uptime and support.

---
#### Simple Lift-and-Shift Migration Strategy
A simple lift-and-shift migration strategy involves moving on-premises servers one by one to virtual machines (VMs) in the cloud. While this approach can be appropriate for initial phases, it may not fully leverage the benefits of cloud architecture.

:p What is a simple lift-and-shift migration strategy?
??x
A simple lift-and-shift migration strategy involves copying existing on-premises servers and applications directly into VMs in the cloud without making any significant changes. This approach is straightforward but may not optimize resource usage, performance, or cost in the cloud environment.

For example:
```java
class ServerMigrationStrategy {
    public void migrateServer(OnPremiseServer server) {
        CloudInstance cloudInstance = new CloudInstance(
            server.getCpuCores(), 
            server.getMemoryGB(), 
            1.0 // default durability factor for simplicity
        );
        System.out.println("Migrating " + server.cpuCores + " CPU cores and " + server.memoryGB + " GB of RAM to cloud instance.");
    }
}

// Example of migration process
ServerMigrationStrategy strategy = new ServerMigrationStrategy();
OnPremiseServer onPremise = new OnPremiseServer(2, 8);
strategy.migrateServer(onPremise);
```
??x
This example shows a basic implementation where an existing on-premises server is directly migrated to a cloud instance. However, this approach may not fully utilize the benefits of cloud services such as scalability and cost optimization.

---
#### Cloud vs On-Premises Cost Models
Cloud providers monetize characteristics like durability, reliability, longevity, and predictability in their pricing models, which can differ significantly from on-premises costs. Understanding these differences is crucial for efficient cloud usage.

:p How do cloud providers charge for services?
??x
Cloud providers charge based on various factors such as the durability of data storage, reliability of compute resources, and predictability of service uptime. Unlike traditional on-premises hardware, which is often seen as a commodity with fixed costs, cloud services are priced dynamically to reflect these additional service guarantees.

For example:
```java
class CloudPricing {
    public double calculatePrice(double durabilityFactor, int iops, int bandwidthMbps) {
        return (durabilityFactor * 0.1 + iops * 0.02 + bandwidthMbps * 0.03); // simplified pricing model
    }
}

// Example of using the pricing model
CloudPricing pricing = new CloudPricing();
double price = pricing.calculatePrice(1.5, 100, 200);
System.out.println("Calculated Price: " + price);
```
??x
This example demonstrates how cloud providers can adjust their pricing based on factors like durability and performance characteristics, reflecting the additional service guarantees provided by cloud infrastructure.

#### Autoscaling and Serverless Functions
Cloud FinOps emphasizes the importance of using autoscaling to manage workloads efficiently. This approach allows servers to scale down when loads are light, reducing costs, and scale up during peak times to handle increased demand.

:p How does autoscaling help in managing cloud resources?
??x
Autoscaling helps by dynamically adjusting the number of active servers based on the current load. When there is low traffic, fewer servers can be used, saving money; when traffic increases, additional servers are automatically provisioned to handle the load, ensuring performance without over-provisioning.

```java
// Pseudocode for a simple autoscaling logic in Java
public class Autoscaler {
    private int minServers;
    private int maxServers;

    public void adjustServerCount(int currentLoad) {
        if (currentLoad < 50) { // Example: If load is less than 50%
            decreaseServerCount();
        } else if (currentLoad > 80) { // Example: If load is more than 80%
            increaseServerCount();
        }
    }

    private void decreaseServerCount() {
        // Code to reduce the number of active servers
    }

    private void increaseServerCount() {
        // Code to increase the number of active servers
    }
}
```
x??

---

#### Reserved or Spot Instances
Using reserved or spot instances is another strategy in cloud FinOps. These instances can offer significant cost savings, allowing workloads to be run more affordably by leveraging unused capacity.

:p How do reserved and spot instances contribute to cost optimization?
??x
Reserved instances provide a commitment for a period of time (1 or 3 years) at a discounted rate, while spot instances allow the use of unused capacity at lower prices but with the risk that they might be interrupted if demand rises. Both can help reduce costs significantly.

```java
// Pseudocode for using reserved and spot instances in Java
public class InstanceManager {
    private boolean isUsingReservedInstance = false;
    private boolean isUsingSpotInstance = true;

    public void selectInstanceType(double estimatedLoad) {
        if (estimatedLoad > 50 && !isUsingReservedInstance) { // Example threshold
            switchToReservedInstance();
        } else if (estimatedLoad < 30 && isUsingSpotInstance) { // Another example threshold
            switchToSpotInstance();
        }
    }

    private void switchToReservedInstance() {
        // Code to purchase and use reserved instances
    }

    private void switchToSpotInstance() {
        // Code to use spot instances
    }
}
```
x??

---

#### Serverless Functions
Serverless functions, also known as functions as a service (FaaS), are a part of cloud FinOps where the infrastructure is managed by the provider. They allow for highly scalable and cost-efficient execution of code in response to events.

:p What advantages do serverless functions offer?
??x
Serverless functions reduce operational overhead significantly because they only execute when triggered, pay-per-use models minimize costs, and automatic scaling ensures that applications can handle sudden surges without manual intervention.

```java
// Pseudocode for a simple serverless function in Java using AWS Lambda
public class MyLambdaFunction {
    public String handler(String input) {
        // Process the input and return a response
        if (input.equals("start")) {
            return "Starting process...";
        } else {
            return "Unknown command: " + input;
        }
    }
}
```
x??

---

#### Data Gravity
Data gravity refers to the phenomenon where data that lands in a cloud platform is difficult and expensive to extract, often due to dependencies on services built around it.

:p Why is data gravity significant for cloud users?
??x
Data gravity makes it harder to move data out of a cloud provider's ecosystem. Once large volumes of data are stored within a cloud service, the costs associated with extracting or moving that data can be substantial, discouraging users from switching providers.

```java
// Pseudocode for handling data egress in Java
public class DataEgressManager {
    private double dataMovementCostPerGB;

    public void calculateEgressCost(int dataSizeInGB) {
        // Calculate the cost of moving data out based on the data size and rate
        double cost = dataSizeInGB * dataMovementCostPerGB;
        System.out.println("Data egress cost: $" + cost);
    }
}
```
x??

---

#### Hybrid Cloud Model
The hybrid cloud model allows businesses to maintain some workloads in their own infrastructure while others are hosted on the cloud, providing flexibility and reducing risks.

:p What are the key benefits of a hybrid cloud approach?
??x
A hybrid cloud model offers several advantages including operational excellence in certain areas (like local hardware), the ability to quickly scale resources using the cloud, cost savings from running less critical workloads on-premises, and better control over data sovereignty.

```java
// Pseudocode for managing a hybrid cloud environment in Java
public class HybridCloudManager {
    private CloudInstanceManager cloudInstanceManager;
    private OnPremiseInstanceManager onPremiseInstanceManager;

    public void migrateWorkloadToCloud(String workloadName) {
        // Logic to move the specified workload to the cloud
        cloudInstanceManager.provisionInstance(workloadName);
        onPremiseInstanceManager.decommissionInstance(workloadName);
    }
}
```
x??

---

#### Multicloud Approach
Multicloud refers to deploying workloads across multiple public clouds. This strategy can leverage the best services from different providers, reduce network latency issues, and provide a more robust environment.

:p What are the primary motivations for using a multicloud approach?
??x
The primary motivations for a multicloud approach include proximity to existing customer cloud workloads (for SaaS platforms), handling data-intensive applications that require low-latency and high-bandwidth connections, and taking advantage of different service offerings from various providers to optimize cost and performance.

```java
// Pseudocode for managing a multicloud setup in Java
public class MulticloudManager {
    private CloudProvider gcp;
    private CloudProvider aws;
    private CloudProvider azure;

    public void migrateDataToCloud(String dataName, CloudProvider provider) {
        // Logic to migrate the specified data to the chosen cloud provider
        if (provider == CloudProvider.GCP) {
            gcp.uploadData(dataName);
        } else if (provider == CloudProvider.AWS) {
            aws.uploadData(dataName);
        } else if (provider == CloudProvider.AZURE) {
            azure.uploadData(dataName);
        }
    }
}
```
x??

---


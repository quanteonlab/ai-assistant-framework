# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 41)

**Starting Chapter:** Docker

---

#### Docker Overview
Docker is a platform built on top of lightweight containers that automates the deployment, scaling, and management of applications. It simplifies the creation and deployment of apps, which are referred to as images in this context. Docker manages container provisioning, networking, and provides its own registry for storing and versioning Docker applications.
:p What does Docker automate in application development?
??x
Docker automates the deployment, scaling, and management of applications by using lightweight containers. This allows developers to focus on building their services rather than managing underlying infrastructure.
x??

---

#### Docker App Abstraction
Just as with VM images, the underlying technology used to implement a service is hidden from us when using Docker apps. Builds for services are created into Docker applications, and these are stored in the Docker registry for easy deployment and versioning.
:p How do Docker apps differ from traditional VM images?
??x
Docker apps provide an abstraction layer where the specific implementation details of the underlying operating system are not visible to the developer. They can be seen as containers that encapsulate an application's code, libraries, and dependencies, making them portable across different environments.
x??

---

#### Docker for Development and Testing
Docker can reduce the complexity of running multiple services locally by using a single VM with Vagrant to host a Docker instance. This allows for faster provisioning of individual services compared to managing multiple independent VMs.
:p How does Docker simplify local development and testing?
??x
Using Docker, developers can set up and tear down a Docker platform on a single VM in Vagrant quickly. Each service can be run as an individual container, which speeds up the development and testing process significantly compared to running multiple independent virtual machines.
```java
// Example of setting up a Docker container for a Java application
public class DockerSetup {
    public void setupDocker() {
        // Pseudocode to create a Dockerfile with necessary configurations
        String dockerFile = "FROM openjdk:8\n" +
                            "COPY target/myapp.jar /app/\n" +
                            "EXPOSE 8080\n" +
                            "CMD java -jar /app/myapp.jar";
        
        // Code to build and run the Docker container
        String command = "docker build -t myapp . && docker run -p 8080:8080 myapp";
    }
}
```
x??

---

#### CoreOS and Lightweight OS for Containers
CoreOS is a Linux-based operating system designed with Docker in mind, providing only essential services to allow containers to run efficiently. This makes it resource-efficient, suitable for running multiple containers.
:p What makes CoreOS unique compared to other operating systems?
??x
CoreOS is unique because it is a stripped-down Linux OS that focuses on minimalism and efficiency. It includes only the necessary components required to run Docker containers, making it lighter and more efficient than traditional OSes like Ubuntu or CentOS.
```bash
# Example of installing software using CoreOS's package manager
sudo yum install -y docker-container-tool
```
x??

---

#### Kubernetes and Container Scheduling
Kubernetes is an open-source platform that helps manage services across multiple Docker instances on different machines. It acts as a scheduling layer to find available Docker containers for running new requests.
:p What does Kubernetes provide in terms of container management?
??x
Kubernetes provides a mechanism for managing Docker containers across multiple nodes or machines. Its primary function is to schedule, deploy, and manage the lifecycle of containers based on defined rules and policies, ensuring that applications run smoothly even when containers are restarted or fail.
```java
// Pseudocode for creating a Kubernetes deployment
public class KubernetesDeployment {
    public void createDeployment() {
        // Define the desired state of the application
        String deploymentYaml = "apiVersion: apps/v1\n" +
                                "kind: Deployment\n" +
                                "metadata:\n" +
                                "  name: myapp-deployment\n" +
                                "spec:\n" +
                                "  replicas: 3\n" +
                                "  selector:\n" +
                                "    matchLabels:\n" +
                                "      app: myapp\n" +
                                "  template:\n" +
                                "    metadata:\n" +
                                "      labels:\n" +
                                "        app: myapp\n" +
                                "    spec:\n" +
                                "      containers:\n" +
                                "      - name: myapp-container\n" +
                                "        image: myapp:latest";
        
        // Code to apply the deployment configuration
        String command = "kubectl apply -f deploymentYaml.yaml";
    }
}
```
x??

---

#### Deis and PaaS on Docker
Deis is a tool that provides a Heroku-like Platform as a Service (PaaS) on top of Docker. It aims to simplify container-based deployments, making it easier for developers to manage applications.
:p How does Deis compare to traditional PaaS solutions?
??x
Deis offers a more lightweight and flexible alternative to traditional PaaS providers like Heroku by leveraging Docker containers. It simplifies the deployment process and allows for greater control over the underlying infrastructure while still providing many of the benefits of managed services.
```java
// Example of deploying an application on Deis
public class DeisDeployment {
    public void deployApplication() {
        // Pseudocode to push a new app to Deis
        String command = "deis create myapp && deis buildpack:add heroku/java && git push deis master";
    }
}
```
x??

---

#### Single Command-Line Deployment Interface

Background context explaining the importance of having a uniform interface for deploying microservices. Highlight that this approach simplifies deployment across different environments and reduces errors.

:p What is the single command-line call used to trigger deployments, and what parameters does it take?
??x
The `deploy` command-line script is used to trigger any deployment. It takes three parameters: 
- `artifact`: The name of the microservice.
- `environment`: The target environment for the deployment (e.g., local, CI, integrated_qa).
- `version`: The version of the artifact to deploy.

For example:
```sh
$ deploy artifact=catalog environment=local version=local
```
This command can be used by developers locally or by CI tools in a pipeline.
x??

---

#### Versioning Strategy

Background context on how different environments might require different versions of microservices, with specific examples provided for local development, CI testing, and QA testing.

:p How does the deployment script handle versioning during different stages (local, CI, QA)?
??x
The deployment script handles versioning based on the environment:
- **Local Development**: The current local version is used.
- **CI Testing**: The latest "green" build, which could be the most recent blessed artifact in the repository, is used. This is typically identified by a specific build number.
- **QA Testing**: The latest version available for testing and diagnosing issues.

For example:
```sh
$ deploy artifact=catalog environment=ci version=b456
```
Here, `b456` could be the CI build number from the recent pipeline run. 
x??

---

#### Deployment Environments

Background context on how microservice topologies might differ between environments but the deployment script abstracts away these differences.

:p What is the role of the `environment` parameter in the deployment command?
??x
The `environment` parameter specifies the target environment for deploying the microservice. This abstraction ensures that the same deployment logic can be used across different environments, hiding the underlying topology and infrastructure details from developers or operators.

For example:
```sh
$ deploy artifact=catalog environment=local version=local
```
This command deploys the `catalog` service into a local environment using the current local version.
x??

---

#### Sample Deployment Command

Background context on common deployment scenarios, including examples of commands for different environments and purposes.

:p Provide an example of how to use the `deploy` script in a CI pipeline.
??x
In a CI pipeline, the `deploy` command is used after a build artifact has been created. For instance:
```sh
$ deploy artifact=catalog environment=ci version=b456
```
Here, `b456` would be the latest build number generated by the CI system.

This ensures that the correct artifact (in this case, build `b456`) is deployed to the CI environment for further testing.
x??

---

#### Script Implementation

Background context on the tools and libraries used to implement such a deployment script. Mention Fabric and Boto as examples for Python, and Capistrano or PowerShell for other environments.

:p What tools are commonly used to implement the `deploy` script?
??x
The `deploy` script can be implemented using various tools and libraries depending on the environment:
- **Python**: Use Fabric with a library like Boto for AWS interactions.
  ```python
  from fabric.api import run, env

  def deploy():
      # Example function to SSH into an instance
      env.hosts = ['user@remote.example.com']
      run('echo Deploying artifact...')
  ```

- **Ruby**: Use Capistrano for deployment tasks.
  ```ruby
  namespace :deploy do
    desc 'Deploy the application'
    task :default do
      # Example Capistrano task to deploy the application
    end
  end
  ```

- **Windows**: Utilize PowerShell scripts.
  ```powershell
  function Deploy {
      param (
          [string]$Artifact,
          [string]$Environment,
          [string]$Version
      )
      Write-Output "Deploying $Artifact to $Environment with version $Version"
  }
  ```

These tools provide a flexible and powerful way to manage deployments, abstracting away the complexity of different environments.
x??

#### Environment Definition
Background context explaining how environment definitions are used to specify resources and services for different environments. YAML files were used as an example, storing details such as nodes, services, credentials, and regions.

:p What is an environment definition?
??x
An environment definition is a mapping from microservices to compute, network, and storage resources in specific environments like development or production. It includes details such as the AMI ID, instance size, number of instances, credentials, and services.
??x
For example:
```yaml
development:
  nodes:
    - ami_id: ami-e1e1234
      size: t1.micro
      credentials_name: eu-west-ssh
      services: [catalog-service]
      region: eu-west-1

production:
  nodes:
    - ami_id: ami-e1e1234
      size: m3.xlarge
      credentials_name: prod-credentials
      services: [catalog-service]
      number: 5
```
x??

---

#### Service Definition
Background context explaining how service definitions store information that remains constant across different environments. Puppet manifests are mentioned as an example of such information.

:p What is a service definition?
??x
A service definition stores information about microservices, which remains the same regardless of the environment. For instance, it might include details like the Puppet manifest file to run.
??x
For example:
```yaml
catalog-service:
  puppet_manifest: catalog.pp
```
x??

---

#### Environment-Specific Resources
Background context on how resources and configurations differ between environments, such as varying node sizes for cost-effectiveness.

:p How do environment definitions handle resource variation?
??x
Environment definitions allow specifying different resources based on the environment. For example, smaller instances can be used in development environments to save costs while larger instances are used in production environments.
??x
For instance:
```yaml
development:
  nodes:
    - ami_id: ami-e1e1234
      size: t1.micro

production:
  nodes:
    - ami_id: ami-e1e1234
      size: m3.xlarge
```
x??

---

#### Credentials Management
Background context on managing different credentials for sensitive environments, stored separately and accessed by specific personnel.

:p How are credentials managed in environment definitions?
??x
Credentials are managed differently based on the environment. Sensitive environments use separate credential stores accessible only to selected individuals, while non-sensitive ones might have more open access.
??x
For example:
```yaml
development:
  nodes:
    - ami_id: ami-e1e1234
      size: t1.micro
      credentials_name: eu-west-ssh

production:
  nodes:
    - ami_id: ami-e1e1234
      size: m3.xlarge
      credentials_name: prod-credentials
```
x??

---

#### Load Balancing
Background context on automatically creating load balancers for services with multiple instances.

:p How are load balancers managed in environment definitions?
??x
Load balancers are automatically created if a service has more than one instance. This is done to distribute traffic efficiently across the nodes.
??x
For example:
```yaml
development:
  nodes:
    - ami_id: ami-e1e1234
      size: t1.micro
      services: [catalog-service]

production:
  nodes:
    - ami_id: ami-e1e1234
      size: m3.xlarge
      number: 5
```
x??

---

#### Port and Connectivity Configuration
Background context on normalizing port usage for services.

:p How are ports configured in environment definitions?
??x
Ports are normalized across environments to ensure consistent service connectivity. Load balancers are automatically set up if more than one instance is present.
??x
For example:
```yaml
catalog-service:
  puppet_manifest: catalog.pp
  connectivity:
    - protocol: tcp
      ports: [8080, 8081]
      allowed: [WORLD]
```
x??

---

#### Tooling and Future Directions
Background context on tools like Terraform that can help manage environment definitions.

:p What is Terraform?
??x
Terraform is a new tool from HashiCorp designed to handle environment definitions, resource provisioning, and configuration management. It aims to create an open-source solution in this space.
??x
For example:
```yaml
provider "aws" {
  region = "eu-west-1"
}

resource "aws_instance" "example" {
  ami           = "ami-e1e1234"
  instance_type = "t1.micro"

  tags = {
    Name = "development-node"
  }
}
```
x??

---


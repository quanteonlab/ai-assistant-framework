# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 22)

**Starting Chapter:** A Crash Course on Kubernetes

---

#### Docker Port Mapping
Background context: When running a Docker container, by default, ports inside the container are not exposed to the host operating system. This means that accessing services running inside a container from the host OS requires specific configuration.

Explanation: If you try to access a service running inside a container on `localhost` (the host machine), it won't work because the port is only accessible within the container itself and not mapped to an external IP or port.

:p What happens when you run a Docker container without mapping ports?
??x
When you run a Docker container without mapping ports, the service running inside the container will be accessible only from within the container. Attempting to access it via `localhost` on the host machine will result in a "Connection refused" error.
```bash
$ docker run -it training/webapp python app.py
```

You then try:

```bash
$ curl localhost:5000
curl: (7) Failed to connect to localhost port 5000: Connection refused
```
x??

---

#### Stopping a Docker Container Gracefully
Background context: When you run a `docker run` command and the container is running, it can be stopped using the appropriate Docker commands. The way to stop the process depends on whether you're stopping an interactive shell session or just killing the container.

Explanation: For stopping a container that's running as part of an interactive shell session (like a Jupyter notebook), use `Ctrl+C`, which sends a SIGINT signal to terminate the current execution.

:p How do you stop a Docker container gracefully?
??x
To stop a Docker container gracefully, press `Ctrl+C` in the terminal where it is running. This will send a SIGINT signal to the process inside the container.
```bash
$ docker run -it training/webapp python app.py  # Run an interactive shell session

# Press Ctrl+C here to terminate the execution and stop the container
```
x??

---

#### Exposing Ports in Docker Containers
Background context: By default, ports within a Docker container are not exposed to the host OS. To make services inside a container accessible from the host machine, you need to map these ports using the `-p` flag.

Explanation: The command `docker run -p 5000:5000 training/webapp` tells Docker to expose port 5000 of the container on port 5000 of the host OS. This makes the service running inside the container accessible from `localhost`.

:p How do you map a port from a Docker container to the host machine?
??x
To map a port from a Docker container to the host machine, use the `-p` flag followed by the host and container ports in the format `docker run -p <host_port>:<container_port> <image_name>`. For example:

```bash
$ docker run -p 5000:5000 training/webapp
```

This command exposes port 5000 of the container on port 5000 of the host machine.
x??

---

#### Cleaning Up Docker Containers
Background context: Every time you run `docker run` and exit, Docker leaves behind unused containers. These can take up disk space and clutter your system.

Explanation: You can clean these up using the `docker rm <CONTAINER_ID>` command or by including the `--rm` flag in your `docker run` command to automatically remove the container when it exits.

:p How do you clean up Docker containers that are no longer needed?
??x
To clean up unused Docker containers, you can use the `docker rm <CONTAINER_ID>` command where `<CONTAINER_ID>` is the ID of the container from the `docker ps` output. Alternatively, you can include the `--rm` flag in your `docker run` command to automatically remove the container when it exits.

Example:
```bash
$ docker rm <CONTAINER_ID>
```

Or:

```bash
$ docker run --rm training/webapp
```
x??

---

#### Kubernetes Overview and Basics
Background context: Kubernetes is an orchestration tool for Docker that helps manage Docker containers across multiple servers. It handles tasks such as scheduling, auto-healing, auto-scaling, load balancing, etc.

Explanation: The main components of a Kubernetes cluster are the control plane and worker nodes. The control plane manages the state of the cluster and schedules containers, while worker nodes run the actual containers based on instructions from the control plane.

:p What is Kubernetes used for?
??x
Kubernetes is used to manage Docker containers across multiple servers. It automates tasks such as scheduling (choosing which server should run a container), auto-healing (automatically redeploying failed containers), auto-scaling (scaling the number of containers based on load), and load balancing (distributing traffic across containers).

Kubernetes is particularly useful for running applications in production environments, especially those with complex requirements like scaling and failover.
x??

---

#### Running Kubernetes Locally
Background context: If you have a recent version of Docker Desktop installed, you can easily set up a local Kubernetes cluster. This allows you to test and develop your applications without needing access to remote servers.

Explanation: In Docker Desktop preferences, there is an option for Kubernetes that enables the creation of a local Kubernetes cluster with just a few clicks.

:p How do you run a Kubernetes cluster locally?
??x
To run a Kubernetes cluster locally using Docker Desktop, open the Docker Desktop preferences and navigate to the Kubernetes section. From here, you can start or stop the local Kubernetes cluster by toggling the switch.

Alternatively, if you have an older version of Docker Desktop that doesn't support this feature directly, you can install Minikube or another tool to create a local Kubernetes environment.
x??

---

#### Enabling Kubernetes on Docker Desktop

Background context: To enable Kubernetes for local development, you need to have Docker Desktop installed and configured. Once enabled, it provides a convenient way to set up a local cluster that can be used to run and test applications using Kubernetes.

:p How do you enable Kubernetes in Docker Desktop?
??x
To enable Kubernetes on Docker Desktop, check the "Enable Kubernetes" checkbox if not already enabled, then click "Apply & Restart." After a few minutes of setup, follow the instructions from the Kubernetes website to install `kubectl`, the command-line tool for interacting with Kubernetes.

```bash
# Install kubectl and configure it.
$ curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/stable-server/binaries.list | grep /linux/amd64/kubectl)
$ chmod +x ./kubectl
$ sudo mv ./kubectl /usr/local/bin/
```

You will need to update the configuration file located in `$HOME/.kube/config` so that `kubectl` knows which Kubernetes cluster to connect to. When you enable Kubernetes through Docker Desktop, it updates this config file for you by adding a `docker-desktop` entry.

??x
:p How do you switch to using the `docker-desktop` context with `kubectl`?
??x
To use the `docker-desktop` context with `kubectl`, run:

```bash
$ kubectl config use-context docker-desktop
Switched to context "docker-desktop".
```

This command tells `kubectl` which cluster configuration to use. After running this, you can check if your Kubernetes cluster is working by using the following command:

```bash
$ kubectl get nodes
```

If everything is set up correctly, you should see information about the node(s) in your local cluster.

??x
:p What does the `kubectl get nodes` command output indicate?
??x
The `kubectl get nodes` command shows details of all nodes in your Kubernetes cluster. For a locally running setup like Docker Desktop, this usually means only one node (your computer), which is both a control plane and a worker node.

Example output:

```
NAME             STATUS   ROLES                  AGE   VERSION
docker-desktop   Ready    control-plane,master   95m   v1.22.5
```

This indicates that the `docker-desktop` cluster has one ready node with roles as both a control plane and master, running Kubernetes version `v1.22.5`.

??x
:p What are Kubernetes Deployments?
??x
Kubernetes Deployments provide a declarative way to manage application replicas in a Kubernetes cluster. You define what Docker images you want to run, how many copies of them (replicas) should be active, resource requirements like CPU and memory limits, port numbers, environment variables, and update strategies.

Example Deployment YAML:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app-container
        image: my-app-image:latest
        ports:
        - containerPort: 8080
```

With this configuration, you specify that `my-app` should run three replicas with the `my-app-image` Docker image. The Deployment ensures that there are always three running instances of the application.

??x
:p What is a Kubernetes Service?
??x
Kubernetes Services provide network load balancing to expose applications running in the cluster. They act as network endpoints, distributing traffic across multiple pods or replicas based on defined rules (like port numbers).

Example Service YAML:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

This service configuration ensures that traffic directed to `my-app-service` gets distributed among the pods labeled with `app: my-app`, using port `8080`.

??x
:p How do you apply Kubernetes objects created from YAML files?
??x
To deploy resources like Deployments and Services, you use the `kubectl apply` command. This command submits the described objects to your cluster.

Example usage:

```bash
$ kubectl apply -f deployment.yaml
$ kubectl apply -f service.yaml
```

Here, `-f` specifies the file containing the YAML configuration for the Deployment or Service you want to create.

??x

#### Creating a Terraform Module for Kubernetes Applications
Background context: This concept explains how to create a Terraform module that deploys an application using Kubernetes. The module will use `kubernetes_deployment` and `kubernetes_service` resources to manage the app's lifecycle, configuration, and networking.

:p What is the purpose of creating a Terraform module for deploying applications in Kubernetes?
??x
The purpose is to encapsulate the necessary configurations for deploying an application using Kubernetes within a reusable module. This helps maintain consistency across different environments and makes it easier to scale or update the deployment.

```hcl
module "k8s-app" {
  source = "./modules/services/k8s-app"
}
```

x??

---

#### Defining Input Variables in Terraform
Background context: Input variables define the parameters that a Terraform module requires. These inputs are used to customize the behavior of the module during deployment.

:p What input variables are defined for the `k8s-app` module?
??x
The following input variables are defined:
- name: The name to use for all resources created by this module.
- image: The Docker image to run.
- container_port: The port the Docker image listens on.
- replicas: How many replicas to run.
- environment_variables: Environment variables to set for the app.

```hcl
variable "name" {
  description = "The name to use for all resources created by this module"
  type        = string
}
```

x??

---

#### Using the `kubernetes_deployment` Resource in Terraform
Background context: The `kubernetes_deployment` resource is used to manage the deployment of a set of Pods. It ensures that a specified number of replica containers are running at any time.

:p How do you configure the `metadata` block within the `kubernetes_deployment` resource?
??x
You configure the `metadata` block by setting the `name` attribute to the name input variable:

```hcl
resource "kubernetes_deployment" "app" {
  metadata {
    name = var.name
  }
}
```

This ensures that the Deployment's name is consistent with the module's configuration.

x??

---

#### Specifying Replica Count in the `spec` Block
Background context: The number of replicas is a crucial parameter for maintaining the desired state of the application deployment. It defines how many instances of the application should be running at any time.

:p How do you specify the number of replicas in the `kubernetes_deployment` resource?
??x
You specify the number of replicas by setting the `replicas` attribute within the `spec` block:

```hcl
resource "kubernetes_deployment" "app" {
  spec {
    replicas = var.replicas
  }
}
```

This ensures that the deployment will maintain the desired number of application instances.

x??

---

#### Defining Pod Template in Kubernetes Deployment
Background context: The `template` block inside the `spec` section defines how the Pods should be created. It includes details like container specifications, labels, and environment variables.

:p How do you define the template for the Pod inside the `kubernetes_deployment` resource?
??x
You define the Pod template by creating a `template` block within the `spec` section:

```hcl
resource "kubernetes_deployment" "app" {
  spec {
    replicas = var.replicas

    template {
      metadata {
        labels = local.pod_labels
      }

      spec {
        container {
          name   = var.name
          image  = var.image
          port {
            container_port = var.container_port
          }
          dynamic "env" {
            for_each = var.environment_variables
            content {
              name  = env.key
              value = env.value
            }
          }
        }
      }
    }
  }
}
```

This block sets the container specifications, including its image and environment variables.

x??

---

#### Creating Local Variables in Terraform
Background context: Local variables are used to store values that can be reused throughout a module. They help maintain consistency and reduce code duplication.

:p How do you create and use local variables in the `kubernetes_deployment` resource?
??x
You create and use local variables by defining them at the top level of your Terraform file:

```hcl
locals {
  pod_labels = {
    app = var.name
  }
}
```

Then, reference these local variables within other blocks as needed, such as in the `metadata` block:

```hcl
resource "kubernetes_deployment" "app" {
  spec {
    replicas = var.replicas

    template {
      metadata {
        labels = local.pod_labels
      }

      spec {
        container {
          name   = var.name
          image  = var.image
          port {
            container_port = var.container_port
          }
          dynamic "env" {
            for_each = var.environment_variables
            content {
              name  = env.key
              value = env.value
            }
          }
        }
      }
    }
  }
}
```

This approach ensures that the `pod_labels` are consistently applied across different parts of your configuration.

x??

---

#### Pod Template Definition
In Kubernetes, a Pod Template defines the specifications for how the Pods should be created and configured. It includes details like container images, ports, environment variables, and labels.

Background context: When deploying applications to Kubernetes, you often need to specify multiple aspects of your application's deployment, such as which Docker image to use, what ports to expose, and which environment variables are required. The Pod Template is a crucial component that encapsulates these details.
:p What does the Pod Template in this context define?
??x
The Pod Template defines the specifications for running containers within a Kubernetes cluster. It includes the name of the container image, the port it should listen on, and any environment variables needed by the application.

For example:
```hcl
resource "kubernetes_pod" "app" {
  metadata {
    labels = var.pod_labels
  }
  spec {
    container {
      name   = var.name
      image  = var.image
      port {
        container_port = var.container_port
      }
      dynamic "env" {
        for_each = var.environment_variables
        content {
          name   = env.key
          value  = env.value
        }
      }
    }
  }
}
```
x??

---

#### Kubernetes Deployment Specification
A Kubernetes Deployment is a resource that automates the deployment and management of application containers across a cluster. It allows you to update, scale, and manage your containerized applications.

Background context: Deployments in Kubernetes are used to ensure stable rolling updates and rollbacks for your application's Pods. They provide mechanisms like rolling updates and revision history.
:p What is included in the `spec` block of a Kubernetes Deployment?
??x
The `spec` block of a Kubernetes Deployment includes details such as how many replicas should be running, which Pod Template to use, and a selector that targets specific Pods.

For example:
```hcl
resource "kubernetes_deployment" "app" {
  metadata {
    name = var.name
  }
  spec {
    replicas = var.replicas
    template {
      metadata {
        labels = local.pod_labels
      }
      spec {
        container {
          name   = var.name
          image  = var.image
          port {
            container_port = var.container_port
          }
          dynamic "env" {
            for_each = var.environment_variables
            content {
              name   = env.key
              value  = env.value
            }
          }
        }
      }
    }
    selector {
      match_labels = local.pod_labels
    }
  }
}
```
x??

---

#### Kubernetes Service Configuration
A Kubernetes Service allows you to expose an application running on a set of Pods as a network service. It defines policies for routing traffic to the services.

Background context: Services are essential for exposing internal cluster resources and making them available outside the cluster or between different clusters.
:p What does the `kubernetes_service` resource do?
??x
The `kubernetes_service` resource creates a Kubernetes Service that routes traffic to specified Pods based on labels. It can be configured as different types of load balancers depending on your cloud provider.

For example:
```hcl
resource "kubernetes_service" "app" {
  metadata {
    name = var.name
  }
  spec {
    type = "LoadBalancer"
    port {
      port         = 80
      target_port  = var.container_port
      protocol     = "TCP"
    }
    selector = local.pod_labels
  }
}
```
x??

---

#### Selector Block in Kubernetes Deployment
The `selector` block in a Kubernetes Deployment ensures that the Deployment targets specific Pods based on their labels. This is crucial for maintaining consistency between the Deployment and the Pods it manages.

Background context: The `selector` uses label keys and values to target specific sets of Pods. Without a selector, the Deployment would not know which Pods it should manage.
:p What does the `selector` block in a Kubernetes Deployment do?
??x
The `selector` block in a Kubernetes Deployment ensures that the Deployment targets specific Pods based on their labels. It uses label keys and values to match against the metadata of the Pods.

For example:
```hcl
resource "kubernetes_deployment" "app" {
  ...
  spec {
    ...
    selector {
      match_labels = local.pod_labels
    }
  }
}
```
x??

---

#### Dynamic Block for Environment Variables
The `dynamic` block in Terraform allows you to iterate over a list of items and generate multiple blocks based on the contents. In this context, it's used to set environment variables dynamically.

Background context: The dynamic block is useful when dealing with varying numbers or types of environment variables that need to be applied to containers.
:p How does the `dynamic` block work for setting environment variables?
??x
The `dynamic` block in Terraform allows you to iterate over a list of items and generate multiple blocks based on the contents. In this context, it's used to set environment variables dynamically by iterating over the `environment_variables` input variable.

For example:
```hcl
resource "kubernetes_deployment" "app" {
  ...
  spec {
    container {
      ...
      dynamic "env" {
        for_each = var.environment_variables
        content {
          name   = env.key
          value  = env.value
        }
      }
    }
  }
}
```
x??

---

#### Load Balancer Types in Kubernetes Services
Kubernetes services can be configured with different types of load balancers depending on your cloud provider, such as `LoadBalancer` for AWS EKS or GKE.

Background context: The type of load balancer determines how traffic is routed to the backend Pods. This choice affects performance and cost.
:p What are the different types of load balancers that can be used with a Kubernetes Service?
??x
The different types of load balancers that can be used with a Kubernetes Service include `LoadBalancer`, which is commonly used for external access in cloud environments like AWS EKS or GKE.

For example:
```hcl
resource "kubernetes_service" "app" {
  ...
  spec {
    type = "LoadBalancer"
    ...
  }
}
```
x??

---

---
#### Exposing Service Endpoint as an Output Variable
This section explains how to expose a Kubernetes service's load balancer hostname as an output variable in Terraform. The `kubernetes_service` resource provides the latest status of a Kubernetes service, which is stored in a local variable called `status`. For a LoadBalancer type service, this object contains nested arrays and maps.

The objective here is to extract the hostname from the deeply nested structure and present it as an output for consumption by other Terraform modules or external systems. However, if there are any changes in the structure of the `status` attribute, the extraction logic may fail, leading to errors. To handle such cases gracefully, a `try` function is used.

:p What is the purpose of using the `try` function in this context?
??x
The `try` function is used to ensure that if any part of the hostname extraction fails due to unexpected changes in the structure of the `status` attribute, the output will default to an error message rather than causing a failure.

Example code:
```terraform
locals {
  status = kubernetes_service.app.status
}

output "service_endpoint" {
  value = try(
    "http://${local.status[0][\"load_balancer\"][0][\"ingress\"][0][\"hostname\"]}",
    "(error parsing hostname from status)"
  )
  description = "The K8S Service endpoint"
}
```
x??

---
#### Module Configuration for Kubernetes Deployment
This section describes how to configure a module in Terraform to deploy a web application to a Kubernetes cluster. The `k8s-app` module is used here, which deploys an application based on Docker images.

:p How does the configuration of the `simple_webapp` module differ from other configurations?
??x
The `simple_webapp` module is configured with specific parameters such as the source (`../../modules/services/k8s-app`), name ("simple-webapp"), image ("training/webapp"), and number of replicas (2). It also specifies a container port for the application.

Example code:
```terraform
module "simple_webapp" {
  source           = "../../modules/services/k8s-app"
  name             = "simple-webapp"
  image            = "training/webapp"
  replicas         = 2
  container_port   = 5000
}
```
x??

---
#### Kubernetes Provider Configuration
This section explains how to configure the `kubernetes` provider in Terraform to connect with a local Kubernetes cluster.

:p How does the `provider kubernetes` block authenticate to the local Kubernetes cluster?
??x
The `provider kubernetes` block authenticates to the local Kubernetes cluster by using the `~/.kube/config` file and the `docker-desktop` context. This is achieved through setting the `config_path` and `config_context` attributes.

Example code:
```terraform
provider "kubernetes" {
  config_path     = "~/.kube/config"
  config_context  = "docker-desktop"
}
```
x??

---
#### Running Terraform Apply
This section describes how to deploy resources using Terraform after configuring the necessary modules and providers.

:p What command is used to see the effects of running `terraform apply`?
??x
The command used to apply the changes defined in the Terraform configuration files is:
```bash
$ terraform apply
```
After executing this command, Terraform will display a plan for the changes it intends to make and ask for your confirmation. Once confirmed, Terraform will execute the plan.

Example output:
```
Apply complete.
```
x??

---

#### Kubernetes Deployment and Pods Overview
Kubernetes is a container orchestration platform that automatically deploys, scales, and manages containerized applications. In this context, Deployments ensure that a specified number of pod replicas are running at any time, while Pods encapsulate containers with shared resources.

Deployments handle rolling updates and rolling back functionalities to maintain application availability during changes.
:p What is the difference between using Kubernetes for deploying an app versus using `docker run`?
??x
Kubernetes provides more advanced features such as automatic deployment of multiple replicas, monitoring their health, and ensuring that the desired number of containers are running. It also handles load balancing across these containers and can automatically replace failed or unhealthy containers.

In contrast, `docker run` only starts a single container without any additional management capabilities.
x??

---

#### Multiple Containers in Action
Kubernetes manages multiple instances (Pods) of an application, ensuring that the desired number of replicas are always running. This is different from running a single instance with Docker.

Each Pod can be seen as a group of containers sharing resources such as storage and network interfaces.
:p How does Kubernetes manage container health compared to `docker run`?
??x
Kubernetes actively monitors the state of Containers within Pods. If a container crashes or fails a liveness or readiness probe, Kubernetes automatically restarts it or replaces it with a new instance (Pod). This ensures that the application remains available even if individual containers fail.

In contrast, `docker run` does not have built-in mechanisms to monitor and replace failing containers.
x??

---

#### Load Balancing in Kubernetes
Kubernetes uses Services to provide load balancing across multiple Pods. A Service is an abstraction that defines a logical set of Pods running the same application and provides a single network endpoint for accessing all of them.

The `Type: LoadBalancer` service creates a load balancer that distributes traffic among the replicas.
:p What does the `kubectl get services` command show in this context?
??x
The `kubectl get services` command lists the available Services in the cluster. In this case, it shows two entries:

1. **Kubernetes Service**: A built-in service for accessing Kubernetes API.
2. **Application Service**: Named `simple-webapp`, which is a load balancer that distributes traffic to the application running in multiple Pods.

The output includes details like the type of the service (LoadBalancer), its IP address, and the ports it exposes.
x??

---

#### Automatic Rollout Updates
Kubernetes can automatically handle rolling updates for applications. This means changes are deployed gradually to ensure availability and rollback capabilities if something goes wrong.

To demonstrate this, you can set environment variables in the Kubernetes Deployment configuration.
:p How can you update an application using Environment Variables in a Kubernetes Deployment?
??x
You can update an application by modifying the `environment_variables` field in the Kubernetes Deployment configuration. For example, setting an environment variable like `PROVIDER = "Terraform"` will instruct the app to use this value instead of its default.

Here is how you would modify the `main.tf` file:

```hcl
module "simple_webapp" {
  source          = "../../modules/services/k8s-app"
  name            = "simple-webapp"
  image           = "training/webapp"
  replicas        = 2
  container_port  = 5000
  environment_variables = {
    PROVIDER = "Terraform"
  }
}
```

After making this change, you would apply the configuration to trigger the update.
x??

---

---
#### EKS Cluster Module Overview
This section introduces creating a module for an Elastic Kubernetes Service (EKS) cluster, allowing for reproducible deployment. The variables.tf file defines inputs such as the name of the cluster and the desired node configuration.

:p What are the key input variables defined in the `variables.tf` file?
??x
The key input variables defined in the `variables.tf` file are:
- `name`: The name to use for the EKS cluster.
- `min_size`: Minimum number of nodes to have in the EKS cluster.
- `max_size`: Maximum number of nodes to have in the EKS cluster.
- `desired_size`: Desired number of nodes to have in the EKS cluster.
- `instance_types`: The types of EC2 instances to run in the node group.

These variables provide flexibility and control over the deployment configuration. 
??x
---

---
#### IAM Role for EKS Control Plane
The following code snippet creates an IAM role specifically for the EKS control plane, allowing it to assume roles necessary for cluster management. The role is attached with a managed policy that provides required permissions.

:p What does this code do in terms of setting up the IAM role?
??x
This code sets up an IAM role named after the specified EKS cluster (`${var.name}-cluster-role`). It allows the EKS service to assume this role and attaches it with a managed policy `AmazonEKSClusterPolicy` that provides necessary permissions.

```pseudocode
// Create an IAM Role for EKS Control Plane
resource "aws_iam_role" "cluster" {
  name               = "${var.name}-cluster-role"
  
  // Policy Document to Allow EKS to Assume the Role
  assume_role_policy = data.aws_iam_policy_document.cluster_assume_role.json
  
  // Attach a Managed Policy for Permissions
  policy {
    arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  }
}

// Data Source for IAM Policy Document to Allow EKS to Assume the Role
data "aws_iam_policy_document" "cluster_assume_role" {
  statement {
    effect   = "Allow"
    actions  = ["sts:AssumeRole"]
    
    principals {
      type         = "Service"
      identifiers  = ["eks.amazonaws.com"]
    }
  }
}
```
??x
---

#### Fetching VPC and Subnet Information
Background context: To set up an EKS cluster, you first need to fetch information about the default VPC and its subnets. This is important because the EKS cluster will be deployed within these resources.

:p How do you fetch the default VPC using Terraform?
??x
You can use the `data "aws_vpc" "default"` data source to fetch information about the default VPC in your AWS account.
```terraform
data "aws_vpc" "default" {
  default = true
}
```
x??

---

#### Fetching Subnets for a Default VPC
Background context: After fetching the default VPC, you need to determine which subnets are available within that VPC. This is done by filtering the subnets based on their VPC ID.

:p How do you fetch subnets associated with the default VPC using Terraform?
??x
You can use the `data "aws_subnets" "default"` data source and filter it by specifying the VPC ID from the fetched default VPC.
```terraform
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
```
x??

---

#### Creating an EKS Cluster Control Plane
Background context: The control plane of the EKS cluster is where the API server, etcd, and other services reside. You need to configure it to use the default VPC and subnets.

:p How do you create an EKS cluster using Terraform?
??x
You can create an EKS cluster by using the `resource "aws_eks_cluster" "cluster"` resource. This resource requires the name, role ARN, version, and VPC configuration.
```terraform
resource "aws_eks_cluster" "cluster" {
  name     = var.name
  role_arn = aws_iam_role.cluster.arn
  version  = "1.21"
  vpc_config {
    subnet_ids  = data.aws_subnets.default.ids
  }
  depends_on = [
    aws_iam_role_policy_attachment.AmazonEKSClusterPolicy,
  ]
}
```
x??

---

#### Creating an IAM Role for EKS Node Group
Background context: To enable managed node groups, you need to create an IAM role that the EC2 instances can assume. This role should have necessary permissions.

:p How do you create an IAM role for the EKS node group using Terraform?
??x
You can create an IAM role by using the `resource "aws_iam_role" "node_group"` resource and attach multiple policies to it.
```terraform
resource "aws_iam_role" "node_group" {
  name               = "${var.name}-node-group"
  assume_role_policy = data.aws_iam_policy_document.node_assume_role.json
}

data "aws_iam_policy_document" "node_assume_role" {
  statement {
    effect   = "Allow"
    actions  = ["sts:AssumeRole"]
    principals {
      type         = "Service"
      identifiers  = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "AmazonEKSWorkerNodePolicy" {
  policy_arn  = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_group.name
}

resource "aws_iam_role_policy_attachment" "AmazonEC2ContainerRegistryReadOnly" {
  policy_arn  = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_group.name
}

resource "aws_iam_role_policy_attachment" "AmazonEKS_CNI_Policy" {
  policy_arn  = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_group.name
}
```
x??

---

#### Creating a Managed Node Group for EKS Cluster
Background context: After setting up the IAM role, you can proceed to create the managed node group that will run your worker nodes in the EKS cluster.

:p How do you create an EKS managed node group using Terraform?
??x
You can create a managed node group by using the `resource "aws_eks_node_group" "nodes"` resource. This resource requires details such as the cluster name, node group name, node role ARN, subnet IDs, and instance types.
```terraform
resource "aws_eks_node_group" "nodes" {
  cluster_name     = aws_eks_cluster.cluster.name
  node_group_name  = var.name
  node_role_arn    = aws_iam_role.node_group.arn
  subnet_ids       = data.aws_subnets.default.ids
  instance_types   = var.instance_types
  scaling_config {
    min_size      = var.min_size
    max_size      = var.max_size
    desired_size  = var.desired_size
  }
  depends_on = [
    aws_iam_role_policy_attachment.AmazonEKSClusterPolicy,
  ]
}
```
x??

#### EKS Cluster Module Configuration
This section explains how to configure and use an EKS (Elastic Kubernetes Service) cluster module within Terraform. The configuration involves setting up a managed node group, defining the required policies for IAM roles, and deploying the cluster with specific instance types.

:p What is the purpose of the `depends_on` attribute in the `eks_cluster` module?
??x
The `depends_on` attribute ensures that the Kubernetes provider can only deploy resources (like the Docker image) after the EKS cluster has been fully deployed. This dependency prevents Terraform from attempting to use an incomplete or non-functional EKS cluster, which could lead to deployment failures.

```hcl
module "simple_webapp" {
  source = "../../modules/services/k8s-app"
  name           = "simple-webapp"
  image          = "training/webapp"
  replicas       = 2
  container_port = 5000
  environment_variables = {
    PROVIDER = "Terraform"
  }
  depends_on      = [module.eks_cluster]
}
```
x??

---

#### EKS Cluster Outputs Configuration
This part describes how to define and output various details about the deployed EKS cluster, such as its name, ARN, endpoint, and certificate authority.

:p What are the key outputs defined in `outputs.tf` for the EKS cluster?
??x
The key outputs defined in `outputs.tf` provide important metadata about the EKS cluster:

```hcl
output "cluster_name" {
  value       = aws_eks_cluster.cluster.name
  description = "Name of the EKS cluster"
}

output "cluster_arn" {
  value       = aws_eks_cluster.cluster.arn
  description = "ARN of the EKS cluster"
}

output "cluster_endpoint" {
  value       = aws_eks_cluster.cluster.endpoint
  description = "Endpoint of the EKS cluster"
}

output "cluster_certificate_authority" {
  value       = aws_eks_cluster.cluster.certificate_authority
  description = "Certificate authority of the EKS cluster"
}
```
x??

---

#### Kubernetes Provider Configuration for EKS Cluster
This section explains how to configure the Kubernetes provider to authenticate and interact with an EKS cluster deployed via Terraform.

:p How does the Kubernetes provider configuration ensure secure communication with the EKS cluster?
??x
The Kubernetes provider is configured to use the correct endpoint, certificate authority, and authentication token from the EKS cluster. This setup ensures that the Kubernetes resources are securely and correctly managed by Terraform.

```hcl
provider "kubernetes" {
  host = module.eks_cluster.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks_cluster.cluster_certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}
```
x??

---

#### Service Endpoint Output for Web Application
This part demonstrates how to output the service endpoint of a deployed web application, which can be used to access it once deployed.

:p What is the purpose of the `service_endpoint` output in the context of deploying a web application?
??x
The `service_endpoint` output provides the URL that can be used to access the deployed web application running on Kubernetes. This endpoint is critical for verifying and accessing the service after deployment.

```hcl
output "service_endpoint" {
  value       = module.simple_webapp.service_endpoint
  description = "The K8S Service endpoint"
}
```
x??

---

#### IAM Policy Attachments for EKS Worker Nodes
This section details how to attach necessary policies to IAM roles for the worker nodes in an EKS cluster, ensuring they have the required permissions.

:p What are the primary policies attached to the IAM role for the EKS worker nodes?
??x
The primary policies attached to the IAM role for the EKS worker nodes include:

- `AmazonEKSWorkerNodePolicy`: Grants necessary permissions for running containerized workloads.
- `AmazonEC2ContainerRegistryReadOnly`: Allows read-only access to the Amazon ECR (Elastic Container Registry).
- `AmazonEKS_CNI_Policy`: Ensures that the CNI plugin can communicate with the Kubernetes control plane.

```hcl
depends_on = [
  aws_iam_role_policy_attachment.AmazonEKSWorkerNodePolicy,
  aws_iam_role_policy_attachment.AmazonEC2ContainerRegistryReadOnly,
  aws_iam_role_policy_attachment.AmazonEKS_CNI_Policy,
]
```
x??

---

#### Instance Type Configuration for Worker Nodes
This section specifies the instance types used by the worker nodes in the EKS cluster, ensuring they meet the required ENI (Elastic Network Interface) limitations.

:p What is the reasoning behind choosing `t3.small` as the instance type for worker nodes?
??x
`t3.small` is chosen as the smallest viable instance type for worker nodes due to the constraints of ENIs in EKS. This instance type has sufficient network capacity, whereas smaller types like `t2.micro`, which have fewer ENIs, cannot meet the requirements for running user-defined Pods.

```hcl
instance_types  = ["t3.small"]
```
x??

---

#### Running Terraform Apply for EKS Deployment
Background context: This section explains how to deploy an application using Terraform with an Amazon EKS cluster. It covers the process of deploying a Kubernetes application and testing it after deployment.

:p What is the command used to run `terraform apply` for deploying an application to an EKS cluster, and what does the output indicate?
??x
The command used to run `terraform apply` is:

```sh
$ terraform apply
```

The output indicates that the resources have been successfully added without any changes or destroyed resources. The `outputs` section shows details like the service endpoint.

```plaintext
Apply complete. Resources: 10 added, 0 changed, 0 destroyed.
Outputs:
Working with Multiple Different Providers | 269
service_endpoint = "http://774696355.us-east-2.elb.amazonaws.com"
```

This output confirms that the application has been deployed successfully and provides the service endpoint to test the deployment.

x??

---

#### Testing the Deployed Application
Background context: After deploying an application using Terraform with EKS, it's essential to verify its functionality. This involves testing the application via a URL provided by the service endpoint.

:p How can you test the application deployed in the EKS cluster after running `terraform apply`?
??x
You can test the application by using the `curl` command on the `service_endpoint` provided in the Terraform output:

```sh
$ curl http://774696355.us-east-2.elb.amazonaws.com
```

This command sends a request to the load balancer, and if everything is set up correctly, you should receive a response like "Hello Terraform." This confirms that the application is running and accessible via the provided URL.

x??

---

#### Updating Environment Variables in Kubernetes Application
Background context: The text explains how to modify environment variables for a Kubernetes application deployed on EKS. This involves updating the `environment_variables` attribute in the Terraform module to reflect changes in the application's configuration.

:p How can you update the environment variables for a Kubernetes application using Terraform?
??x
You can update the environment variables by modifying the `environment_variables` attribute within the Terraform module:

```terraform
module "simple_webapp"  {
   source          = "../../modules/services/k8s-app"
   name            = "simple-webapp"
   image           = "training/webapp"
   replicas        = 2
   container_port  = 5000
   environment_variables  = { 
     PROVIDER  = "Readers" 
   }
   # Only deploy the app after the cluster has been deployed
   depends_on      = [module.eks_cluster ] 
}
```

After making these changes, running `terraform apply` will update the Kubernetes application with the new environment variables. This process is faster due to Docker's image caching mechanisms and Kubernetes' deployment capabilities.

x??

---

#### Using kubectl to Inspect Kubernetes Cluster
Background context: After deploying an application using Terraform on EKS, you can use `kubectl` to inspect various components of your Kubernetes cluster such as nodes, deployments, pods, and services. This involves authenticating to the EKS cluster using AWS CLI commands.

:p How do you authenticate `kubectl` to interact with an EKS cluster?
??x
You can authenticate `kubectl` to interact with an EKS cluster by running:

```sh
$ aws eks update-kubeconfig --region <REGION> --name <EKS_CLUSTER_NAME>
```

Replace `<REGION>` and `<EKS_CLUSTER_NAME>` with the appropriate values. For example, if your region is `us-east-2` and your cluster name is `kubernetes-example`, you would run:

```sh
$ aws eks update-kubeconfig --region us-east-2 --name kubernetes-example
```

This command updates your local Kubernetes configuration file (`~/.kube/config`) with the necessary credentials to interact with the specified EKS cluster.

x??

---

#### Using Off-the-Shelf Production-Grade Kubernetes Modules
Background context: The use of off-the-shelf production-grade Kubernetes modules, such as those found in the Gruntwork Infrastructure as Code Library, can simplify deploying EKS clusters and Kubernetes applications. These modules help ensure consistency and reduce the risk of configuration errors.
:p How should private subnets be used for an EKS cluster instead of default VPC and public subnets?
??x
To use private subnets for your EKS cluster, you should configure the EKS cluster within a VPC that uses only private subnets. This approach enhances security by isolating the Kubernetes workloads from internet exposure.
```terraform
resource "aws_vpc" "example" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.example.id
  cidr_block        = ["10.0.${count.index + 1}.0/24"]
  availability_zone = var.availability_zones[count.index]
}
```
x??

---

#### Using Multiple Providers Sparingly
Background context: While it is possible to use multiple providers in a single Terraform module, doing so is generally discouraged due to issues related to dependency ordering and isolation. Each provider should ideally be isolated in its own module.
:p Why shouldn’t you use multiple providers in the same module?
??x
Using multiple providers in the same module can lead to issues with dependency ordering between different cloud services or Kubernetes resources, making debugging and maintaining your infrastructure more challenging. Moreover, it increases the blast radius if something goes wrong, affecting all resources managed by those providers.
```terraform
provider "aws" {
  alias  = "example"
  region = var.region1
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}
```
x??

---

#### Deploying EKS Cluster and Kubernetes Apps Separately
Background context: To avoid issues related to dependency ordering, it is recommended to deploy the EKS cluster in one module and any Kubernetes apps in separate modules. This approach helps isolate different parts of your infrastructure and limits the impact of errors.
:p How should you deploy an EKS cluster and Kubernetes apps separately?
??x
Deploy the EKS cluster in a single module using the AWS provider, then create separate modules for deploying Kubernetes apps into that cluster. This separation ensures each part is managed independently and reduces the risk of conflicts or unintended side effects.
```terraform
# Module to deploy EKS Cluster
module "eks_cluster" {
  source = "./modules/eks"
  region = var.region1
}

# Separate module to deploy Kubernetes Apps
module "k8s_apps" {
  source = "./modules/k8s-apps"
  cluster_name = module.eks_cluster.cluster_name
}
```
x??

---

#### Handling Multiple AWS Regions, Accounts, and Clouds
Background context: To support deployment across multiple AWS regions, accounts, or other clouds, you can use multiple provider blocks in your Terraform code. Each block is configured with the appropriate region, assume_role, or cloud-specific settings.
:p How should you deploy to multiple AWS regions using Terraform?
??x
To deploy resources to multiple AWS regions, you would configure separate provider blocks for each region and apply the configuration files accordingly. This ensures that each deployment is isolated and can be managed independently.
```terraform
# Provider block for Region 1
provider "aws" {
  alias  = "region1"
  region = var.region1
}

# Provider block for Region 2
provider "aws" {
  alias  = "region2"
  region = var.region2
}
```
x??

---


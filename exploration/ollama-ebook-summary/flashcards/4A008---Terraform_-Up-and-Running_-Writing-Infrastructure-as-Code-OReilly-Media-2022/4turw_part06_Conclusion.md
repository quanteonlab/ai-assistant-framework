# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 6)

**Starting Chapter:** Conclusion

---

#### Terraform Overview
Terraform is a tool used for deploying and managing infrastructure as code. It automates the process of setting up servers, networks, storage, and other resources required to run applications or services. Terraform uses configuration files written in HCL (HashiCorp Configuration Language) to define the desired state of the infrastructure.

:p What is Terraform?
??x
Terraform is a tool for managing and deploying infrastructure as code, using HCL configuration files to define the desired state of the resources.
x??

---

#### Packer Overview
Packer is a tool that creates machine images (also known as templates or AMIs) from a set of base images and customizations. It enables you to create consistent environments for your applications by building reproducible and identical instances on any infrastructure.

:p What does Packer do?
??x
Packer builds machine images (templates, AMIs) from base images with custom configurations, ensuring that the environment is consistent across all instances.
x??

---

#### Kubernetes Overview
Kubernetes manages containerized applications by orchestrating Docker containers within a cluster. It provides features such as deployment strategies, auto-healing, and auto-scaling to ensure high availability and efficient resource utilization.

:p What does Kubernetes manage?
??x
Kubernetes manages containerized applications through a cluster of nodes, offering deployment strategies, auto-healing, and auto-scaling capabilities.
x??

---

#### Docker Containers
Docker containers are lightweight, standalone executable packages that include everything needed to run an application. They encapsulate the code along with its runtime, dependencies, libraries, and configuration files into one package.

:p What is a Docker container?
??x
A Docker container is a lightweight, portable environment for running applications in isolation, ensuring that the application runs consistently across different systems.
x??

---

#### Infrastructure as Code (IaC) Tools Comparison
The table provided compares popular IaC tools such as Chef, Puppet, Ansible, Pulumi, CloudFormation, Heat, and Terraform. Each tool has its own strengths and weaknesses in terms of language support, maturity, community size, and provision type.

:p What does the table show?
??x
The table shows a comparison of popular IaC tools, highlighting their default or most common usage methods across different criteria such as language support, maturity, community size, and provision type.
x??

---

#### Chef Overview
Chef is an infrastructure automation tool that configures servers using recipes. It allows for procedural configuration management, where configurations are applied step-by-step.

:p What does Chef do?
??x
Chef automates server configuration by applying recipes to manage the desired state of the infrastructure in a procedural manner.
x??

---

#### Puppet Overview
Puppet is an open-source automation tool that uses declarative manifests to describe the desired state of systems. It supports both mutable and immutable infrastructures.

:p What does Puppet do?
??x
Puppet manages system configurations using declarative manifests, describing the desired state of resources in a flexible manner.
x??

---

#### Ansible Overview
Ansible is an open-source automation tool that uses YAML playbooks to describe tasks. It provides a simple configuration management approach with no agents needed on the managed nodes.

:p What does Ansible do?
??x
Ansible automates tasks using YAML playbooks, providing a straightforward and agentless configuration management solution.
x??

---

#### Pulumi Overview
Pulumi is an IaC tool that allows developers to use familiar programming languages like JavaScript/TypeScript for infrastructure automation. It supports cloud-native applications by leveraging modern development practices.

:p What does Pulumi do?
??x
Pulumi automates infrastructure using programming languages such as JavaScript/TypeScript, allowing for cloud-native application development with a focus on modern software engineering practices.
x??

---

#### CloudFormation Overview
CloudFormation is an AWS service that uses templates to define and provision AWS resources in an organized way. It enables the creation of complex resource configurations through YAML or JSON.

:p What does CloudFormation do?
??x
CloudFormation creates and provisions AWS resources using templates written in YAML or JSON, enabling the management of complex resource configurations.
x??

---

#### Heat Overview
Heat is an open-source orchestration engine that uses heat templates to define and provision resources. It works with OpenStack to manage cloud infrastructure.

:p What does Heat do?
??x
Heat manages cloud infrastructure by using heat templates to define and provision resources, working within the OpenStack environment.
x??

---

#### Terraform Flexibility
Terraform is flexible enough to be used in various configurations beyond its default use. For example, it can be used without a master or for immutable infrastructure.

:p How flexible is Terraform?
??x
Terraform is flexible and can be adapted for different deployment scenarios, such as using it without a master node or implementing immutable infrastructure.
x??

---

#### Table 1-4: IaC Tool Comparison
The table in the text compares popular IaC tools based on criteria like language support, maturity, community size, and provision type. This helps determine which tool might best fit specific needs.

:p What does Table 1-4 show?
??x
Table 1-4 provides a comparison of the most common ways to use popular IaC tools, highlighting their features such as language support, maturity, community size, and provision type.
x??

---

#### Gruntwork Criteria
Gruntwork selected Terraform due to its open-source nature, cloud-agnostic capabilities, large community, mature codebase, support for immutable infrastructure, declarative language, masterless and agentless architecture, and optional paid service.

:p Why did Gruntwork choose Terraform?
??x
Gruntwork chose Terraform because it is open-source, supports a wide range of clouds, has a large user community, offers a mature codebase, includes support for immutable infrastructure, uses a declarative language, features a masterless and agentless architecture, and provides an optional paid service.
x??

---

#### AWS Cloud Services Market Share
Background context: The text starts by mentioning the market share of AWS, which is the most popular cloud infrastructure provider. It has a 32 percent share in the cloud infrastructure market, more than the combined share of its next three biggest competitors (Microsoft, Google, and IBM).

:p What does the 32% market share indicate about AWS?
??x
The significant market dominance of AWS, indicating that it is the preferred provider for most companies looking to use cloud services. This large market share suggests a wide range of services and extensive customer support.
x??

---

#### Setting Up Your AWS Account
Background context: The text explains how to set up an AWS account if you don't have one, emphasizing the importance of using limited user accounts for security reasons.

:p How do you create a more-limited IAM user?
??x
To create a more-limited IAM user:
1. Go to the IAM Console.
2. Click on "Users" and then click "Add Users."
3. Enter a name for the user and ensure "Access key - Programmatic access" is selected.
4. Add permissions using an IAM Policy, such as AdministratorAccess.
5. After creating the user, save their security credentials (Access Key ID and Secret Access Key) securely.

Caveat: The root user should only be used to create limited user accounts; it’s not recommended for daily use due to its broad permissions.
x??

---

#### AWS Free Tier
Background context: The text highlights that AWS offers a free tier, which can cover the cost of running examples in the book. This is beneficial for learning without significant financial risk.

:p How much does the AWS Free Tier typically cost?
??x
The AWS Free Tier typically covers all example costs mentioned in the book, allowing you to run them at no charge or very low cost during the first year. If you've already used your free credits, running these examples should only cost a few dollars.
x??

---

#### Terraform Basics
Background context: The text introduces Terraform as an easy-to-learn tool for deploying infrastructure across various cloud providers.

:p What is Terraform?
??x
Terraform is a tool that allows you to define and provision infrastructure using configuration files. It supports provisioning infrastructure on public cloud providers like AWS, Azure, Google Cloud, DigitalOcean, and private cloud/virtualization platforms such as OpenStack and VMware. The goal of the tool in this book is to help you deploy scalable, highly available web services.
x??

---

#### Deploying a Single Server
Background context: One of the first steps in learning Terraform is to deploy a single server on AWS.

:p How do you start deploying a single server with Terraform?
??x
1. Set up your AWS account and IAM user as described earlier.
2. Install Terraform.
3. Write a configuration file using Terraform syntax, specifying the type of server (e.g., EC2 instance) to deploy.
4. Run `terraform init` followed by `terraform apply` to create the resources.

Example command:
```bash
# Initialize Terraform with AWS provider settings
terraform init

# Apply the infrastructure changes defined in your config file
terraform apply
```
x??

---

#### Deploying a Web Server
Background context: After deploying a single server, the next step is to deploy a web server that can handle HTTP requests.

:p How do you configure Terraform to deploy a web server?
??x
To deploy a web server using Terraform:
1. Create an `amazonec2_instance` resource in your configuration file.
2. Define the necessary properties like instance type, AMI, key pair, and security group rules.
3. Ensure that the security group allows HTTP traffic.

Example configuration snippet:
```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "WebServer"
  }

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo yum install httpd -y
              sudo systemctl start httpd
              sudo systemctl enable httpd
              EOF
}
```
x??

---

#### Deploying a Load Balancer
Background context: The text mentions deploying a load balancer to distribute traffic across multiple web servers.

:p How do you use Terraform to deploy a load balancer?
??x
To deploy an Elastic Load Balancer (ELB) using Terraform:
1. Create an `aws_elb` resource in your configuration file.
2. Define the name, listener ports, and backend instances.
3. Ensure that the ELB is configured with appropriate security group rules.

Example configuration snippet:
```hcl
resource "aws_elb" "web" {
  name                   = "web"
  subnets                = [for subnet in aws_subnet.web: subnet.id]
  security_groups        = [aws_security_group.web.id]

  listener {
    instance_port        = 80
    lb_port              = 80
    protocol             = "HTTP"
  }

  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 30
    interval            = 30
    target              = "HTTP:80/"
  }
}
```
x??

---

#### Cleaning Up Resources
Background context: The text suggests cleaning up resources after you are done with the examples to avoid unnecessary costs.

:p How do you clean up Terraform-managed infrastructure?
??x
To clean up resources deployed by Terraform:
1. Run `terraform destroy` to remove all resources created in your configuration.
2. Confirm the deletion process, ensuring that no errors occur during cleanup.

Example command:
```bash
# Destroy the infrastructure defined in your config file
terraform destroy
```
x??

---

#### Installing Terraform on macOS Using Homebrew
Background context: This section explains how to install Terraform using a package manager like Homebrew, which is popular for macOS users. The goal is to get Terraform set up and ready to use.

:p How do you install Terraform on macOS if you are using Homebrew?
??x
To install Terraform on macOS using Homebrew, follow these steps:

1. Tap the HashiCorp repository by running:
   ```bash
   $ brew tap hashicorp/tap
   ```
2. Install Terraform from this repository by running:
   ```bash
   $ brew install hashicorp/tap/terraform
   ```

This command will download and install the latest version of Terraform.
x??

---

#### Installing Terraform on Windows Using Chocolatey
Background context: This section explains how to install Terraform using a package manager like Chocolatey, which is popular for Windows users. The goal is to get Terraform set up and ready to use.

:p How do you install Terraform on Windows if you are using Chocolatey?
??x
To install Terraform on Windows using Chocolatey, follow these steps:

1. Run the following command in your terminal:
   ```bash
   $ choco install terraform
   ```

This command will download and install the latest version of Terraform from the Chocolatey repository.
x??

---

#### Checking Installation with Terraform Command
Background context: This section explains how to verify that Terraform has been installed correctly by running a simple command. The goal is to ensure that Terraform can be used for further operations.

:p How do you check if Terraform is working correctly after installation?
??x
To check if Terraform is working correctly, run the following command in your terminal:
```bash
$ terraform
```

This will display the usage instructions for Terraform, indicating that it has been installed successfully and can be used to create or manage infrastructure.
x??

---

#### Setting AWS Credentials as Environment Variables
Background context: This section explains how to securely store and set up AWS credentials using environment variables. The goal is to enable Terraform to make changes in your AWS account.

:p How do you set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables on a Unix/Linux/macOS terminal?
??x
To set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables on a Unix/Linux/macOS terminal, use the following commands:
```bash
$ export AWS_ACCESS_KEY_ID=(your access key id)
$ export AWS_SECRET_ACCESS_KEY=(your secret access key)
```

These commands export the required credentials as environment variables. Note that these environment variables are only available in the current shell session and need to be set again after a reboot or opening a new terminal window.
x??

---

#### Setting AWS Credentials on Windows Command Terminal
Background context: This section explains how to securely store and set up AWS credentials using environment variables for Windows users.

:p How do you set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables in a Windows command terminal?
??x
To set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables in a Windows command terminal, use the following commands:
```cmd
$ set AWS_ACCESS_KEY_ID=(your access key id)
$ set AWS_SECRET_ACCESS_KEY=(your secret access key)
```

These commands export the required credentials as environment variables. Note that these environment variables are only available in the current shell session and need to be set again after a reboot or opening a new terminal window.
x??

---

#### Terraform Configuration Language (HCL) and Syntax
Background context: This section explains the use of HCL for writing Terraform configurations, including file extensions and basic syntax.

:p What is the extension used for Terraform configuration files?
??x
The extension used for Terraform configuration files is `.tf`.

Terraform code is written in the HashiCorp Configuration Language (HCL), which uses this file extension to denote configuration files.
x??

---

#### Example of a Basic AWS Provider Configuration
Background context: This section provides an example of how to configure the AWS provider in your Terraform setup, specifying the region for infrastructure deployment.

:p What is the content of a basic `main.tf` file that configures the AWS provider?
??x
A basic `main.tf` file that configures the AWS provider and sets the region might look like this:

```hcl
provider "aws" {
  region = "us-east-2"
}
```

This configuration tells Terraform to use the AWS provider and deploy infrastructure in the us-east-2 region.
x??

---

---

#### AWS Regions and Availability Zones

AWS organizes its infrastructure into regions, which are geographically separate areas. Within each region, there are multiple isolated data centers called Availability Zones (AZs).

Each AZ is designed to be independent of others within the same region, meaning that an outage in one AZ should not affect another.

:p What is an AWS Region?
??x
An AWS Region is a geographic area where all the underlying hardware and services are hosted. For example, us-east-2 represents the Ohio region.
x??

---

#### Creating Resources with Terraform

Terraform allows you to define infrastructure as code using configuration files written in HCL (HashiCorp Configuration Language). The general syntax for creating resources is:

```hcl
resource "<PROVIDER>_<TYPE>" "<NAME>" {
  [CONFIG ...]
}
```

Where:
- PROVIDER: Specifies the provider, like `aws`.
- TYPE: Specifies the type of resource to create, such as `instance`.
- NAME: An identifier for this resource.
- CONFIG: Arguments specific to that resource.

:p What is the general syntax for creating resources in Terraform?
??x
The syntax involves specifying a provider and a resource type within the `resource` block. For example:
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0fb653ca2d3203ac1"
  instance_type = "t2.micro"
}
```
x??

---

#### AWS EC2 Instance Example

To deploy a single server (EC2 instance) in AWS using Terraform, you use the `aws_instance` resource.

:p How do you create an AWS EC2 instance with Terraform?
??x
You can create an AWS EC2 instance by defining the `aws_instance` resource and specifying the AMI ID and instance type. Here’s a basic example:
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0fb653ca2d3203ac1"
  instance_type = "t2.micro"
}
```
This sets up an EC2 instance using the `t2.micro` type, which is part of the AWS Free Tier and provides one virtual CPU and 1 GB of memory.
x??

---

#### AMI ID Considerations

AWS AMIs (Amazon Machine Images) are specific to each region. The example provided uses an Ubuntu 20.04 image in the `us-east-2` region.

:p Why does the AMI ID need to be region-specific?
??x
The AMI ID is unique per AWS region because it identifies a particular operating system or software stack configured by Amazon for use with EC2 instances. For example, if you change the region parameter to `eu-west-1`, you would need to find and replace the AMI ID with the corresponding Ubuntu 20.04 image ID for that region.
x??

---

#### AWS Instance Types

AWS offers various instance types with different CPU, memory, disk space, and networking capacities.

:p What is an EC2 Instance type?
??x
An EC2 Instance type is a specific configuration of computing resources available on AWS. For example, `t2.micro` provides one virtual CPU and 1 GB of memory. You can find the full list of instance types in the EC2 Instance Types documentation.
x??

---

#### Using Terraform Documentation

Terraform provides extensive documentation for each provider and resource type.

:p Where should you look when writing Terraform code?
??x
You should refer to the official Terraform documentation to understand available resources, their configuration options, and usage examples. The documentation can be particularly useful for finding specific arguments or understanding complex configurations.
x??

---

#### Terraform Initialization Process
Terraform is an open-source infrastructure as code (IaC) tool that allows you to define and provision your cloud resources using configuration files. The `terraform init` command initializes a working directory containing Terraform configuration, making sure all necessary provider plugins are downloaded.

:p What does the `terraform init` command do?
??x
The `terraform init` command is used to initialize a new Terraform project in a given working directory. It ensures that the correct version of each provider (e.g., AWS) is installed and available for use. The process involves downloading providers from their respective repositories, such as GitHub, and storing them locally in a `.terraform` folder within your project.

```sh
$ terraform init
```
x??

---

#### Running Terraform Plan
After initializing the backend and providers, you can run `terraform plan` to see what changes will be made. This command simulates the execution of Terraform commands without actually making any changes to the infrastructure.

:p What does the `terraform plan` command do?
??x
The `terraform plan` command generates a detailed report showing what Terraform intends to do if you run `terraform apply`. It checks your configuration and provider plugins, then outputs a list of resources that will be created, modified, or destroyed. This allows you to review the proposed changes before applying them.

```sh
$ terraform plan
```
Output:
```
Terraform will perform the following actions:

  # aws_instance.example will be created
  + resource "aws_instance" "example" {
      ...
  }
Plan: 1 to add, 0 to change, 0 to destroy.
```

x??

---

#### Terraform Apply Command
Once you are satisfied with the plan, you can execute `terraform apply` to make the proposed changes.

:p What does the `terraform apply` command do?
??x
The `terraform apply` command applies a previously generated execution plan to your infrastructure. It creates resources according to the configuration defined in your Terraform files and provider plugins. After running `terraform apply`, you will see detailed outputs indicating what actions are being taken.

```sh
$ terraform apply
```
Output:
```
Terraform will perform the following actions:

  # aws_instance.example will be created
  + resource "aws_instance" "example" {
      ...
  }
Plan: 1 to add, 0 to change, 0 to destroy.
```

x??

---

#### Terraform Backend Initialization
The `terraform init` command also initializes a backend, which is a storage mechanism used by Terraform for storing state and workspaces.

:p What does the "Initializing the backend" message indicate?
??x
When you see the "Initializing the backend..." message during the execution of `terraform init`, it means that Terraform is setting up a backend to store information about your infrastructure's state. The backend can be configured to use different storage mechanisms, such as remote servers or local files.

:x?

---

#### Understanding Plan Output
The plan output shows what actions Terraform intends to take before applying them with `terraform apply`.

:p What does the output of `terraform plan` show?
??x
The output of `terraform plan` provides a summary of the resources that will be created, modified, or destroyed. It lists each resource and indicates whether it is being added, changed, or removed.

For example:
```
Terraform will perform the following actions:

  # aws_instance.example will be created
  + resource "aws_instance" "example" {
      ...
  }
Plan: 1 to add, 0 to change, 0 to destroy.
```

This output tells you that Terraform plans to create one EC2 instance and does not intend to modify or delete any existing resources.

x??

---

#### Idempotence of `terraform init`
`terraform init` can be run multiple times without causing issues because it is idempotent; running the command again will not alter your project unless there are changes in configuration files.

:p Why is `terraform init` idempotent?
??x
`terraform init` is designed to be idempotent, meaning that running it repeatedly does not change the state of your Terraform project beyond its initial setup. This property ensures that you can safely run `terraform init` multiple times without worrying about unexpected behavior.

```sh
$ terraform init  # Initial setup
$ terraform init  # No-op as everything is already initialized
```

This idempotence makes it safe to include `terraform init` in your deployment scripts or CI/CD pipelines, ensuring that the environment always starts from a clean state when necessary.

x??

---

#### Terraform Apply Command
Terraform's apply command is used to execute the changes described in a Terraform plan. It shows you the same output as the `plan` command but asks for confirmation before executing the actions.

:p What does the `terraform apply` command do?
??x
The `terraform apply` command executes the changes described in a Terraform plan, showing the same output as the `plan` command and asking for confirmation to proceed. If you confirm with 'yes', it will carry out the actions.
```shell
Do you want to perform these actions? Terraform will perform the actions described above. Only 'yes' will be accepted to approve. Enter a value: yes
```
x??

---

#### EC2 Instance Deployment
Terraform allows you to deploy resources like EC2 instances in your AWS account by defining resource blocks in configuration files and running `terraform apply`.

:p How do you deploy an EC2 instance using Terraform?
??x
You define the resource block for the EC2 instance in your Terraform configuration file, then run `terraform apply`. This command will create or update the EC2 instance according to the defined configuration.
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0fb653ca2d3203ac1"
  instance_type = "t2.micro"
}
```
x??

---

#### Tagging Resources with Terraform
Terraform supports tagging resources, which allows you to add metadata like names or descriptions. When deploying a resource and adding tags, Terraform can update the existing resource if it already exists.

:p How do you add a tag to an EC2 instance using Terraform?
??x
You can add a `tags` block within your resource definition for the EC2 instance to specify tags. Running `terraform apply` will then either create or update the instance with these new tags.
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0fb653ca2d3203ac1"
  instance_type = "t2.micro"

  tags = {
    Name = "terraform-example"
  }
}
```
x??

---

#### Version Control with Git
Version control using Git is essential for managing changes to your Terraform configurations, allowing you to track history and collaborate with team members.

:p How do you set up a local Git repository for your Terraform configuration?
??x
To set up a local Git repository for your Terraform configuration, initialize a new Git repository, add your Terraform configuration files, and commit the changes. You also need to create and commit a `.gitignore` file to exclude unnecessary files from version control.
```sh
$ git init
$ git add main.tf .terraform.lock.hcl
$ git commit -m "Initial commit"
```
Create a `.gitignore` file with:
```plaintext
.terraform
*.tfstate
*.tfstate.backup
```
Then commit the `.gitignore` file.
```sh
$ git add .gitignore
$ git commit -m "Add a .gitignore file"
```
x??

---


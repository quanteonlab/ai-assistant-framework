# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 2)

**Starting Chapter:** Chapter 1. Why Terraform. What Is DevOps

---

#### What is DevOps?
Background context explaining the concept. In the past, developers and operations teams worked in isolation, with manual processes for deploying applications. This led to inefficiencies and reliability issues as companies scaled.
If applicable, add code examples with explanations:
```java
public class Example {
    // Code that might be developed by a Dev team
}
```
:p What is DevOps?
??x
DevOps is a set of practices intended to improve the efficiency and automation in software delivery. It aims to break down barriers between development (Dev) and operations (Ops) teams, fostering collaboration and continuous improvement.
x??

---

#### What is Infrastructure as Code?
Background context explaining the concept. Traditionally, infrastructure setup was done manually or via scripts specific to each environment. Infrastructure as code (IaC) involves treating infrastructure configuration as a software asset, written in a declarative language and managed with version control.
:p Define Infrastructure as Code (IaC)?
??x
Infrastructure as Code (IaC) refers to the practice of managing and provisioning infrastructure resources using code. It treats infrastructure configuration as a software artifact that can be stored in a source control system and automated for deployment, ensuring consistency and repeatability across environments.
x??

---

#### What Are the Benefits of Infrastructure as Code?
Background context explaining the concept. IaC allows for the automation of repetitive tasks, reduces human errors through consistent application of policies, and enables better collaboration between development and operations teams by treating infrastructure like any other code.
:p List three benefits of using IaC.
??x
1. Consistency: Ensures that infrastructure is consistently configured across all environments.
2. Automation: Reduces manual effort and increases deployment speed.
3. Collaboration: Facilitates better communication and coordination between Dev and Ops teams.

For example, consider a Terraform script:
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "example-instance"
  }
}
```
This script is managed in version control and can be applied across different environments with ease.
x??

---

#### How Does Terraform Work?
Background context explaining the concept. Terraform is a tool for infrastructure management that allows you to define, manage, and deploy multi-cloud infrastructure as code. It uses configuration files written in HCL (HashiCorp Configuration Language) or JSON.
:p Explain how Terraform works.
??x
Terraform works by defining your infrastructure using configuration files in HashiCorp Configuration Language (HCL). These files describe the desired state of your infrastructure, and Terraform manages the steps required to achieve that state. You can apply these configurations to different environments like development, testing, or production.

Example HCL file:
```hcl
provider "aws" {
  region = "us-west-2"
}

resource "aws_vpc" "example" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name = "example-vpc"
  }
}
```
Terraform will manage the lifecycle of resources defined in this file, ensuring they match the desired state you have specified.
x??

---

#### How Does Terraform Compare to Other Infrastructure-as-Code Tools?
Background context explaining the concept. Terraform is part of a broader category of IaC tools like Ansible, Chef, and Puppet. Each tool has its strengths, but Terraform stands out with its simplicity, flexibility, and multi-cloud support.
:p Compare Terraform to another IaC tool (e.g., Ansible).
??x
Terraform compared to Ansible:
- **Language**: Terraform uses HCL or JSON for configuration, while Ansible is written in YAML.
- **Deployment Method**: Terraform manages infrastructure as code and applies changes using a `terraform apply` command. Ansible uses playbooks that are executed on target machines.
- **Multi-cloud Support**: Terraform supports multiple cloud providers like AWS, Azure, GCP, etc., making it highly versatile for multi-cloud strategies. Ansible primarily focuses on configuration management rather than infrastructure as code.

Example Terraform vs. Ansible playbook:
Terraform:
```hcl
resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "example-instance"
  }
}
```

Ansible playbook (YAML):
```yaml
- name: Create an EC2 instance
  hosts: localhost
  connection: local

  tasks:
  - name: Launch an EC2 instance
    ec2:
      region: us-west-2
      image_id: ami-0c55b159cbfafe1f0
      instance_type: t2.micro
      tags:
        Name: example-instance
```

Both tools serve different purposes, but Terraform is often chosen for its ability to manage infrastructure as code.
x??

---

#### Infrastructure as Code (IaC)
Background context: Infrastructure as Code (IaC) is a practice where infrastructure is managed through source control and defined with programming languages. This approach aims to treat all aspects of operations, including hardware configuration, as software. It provides an alternative to ad hoc scripts and specialized tools like Chef, Puppet, Ansible, and others.

:p What is Infrastructure as Code (IaC)?
??x
Infrastructure as Code (IaC) refers to managing infrastructure through source control using programming languages. This practice transforms the provisioning and management of infrastructure into a software development process.
??x

---

#### Ad Hoc Scripts
Background context: Ad hoc scripts are custom, one-off solutions written for specific tasks without following strict conventions or structures. These scripts can be written in any general-purpose language like Bash, Ruby, Python.

:p What are ad hoc scripts?
??x
Ad hoc scripts are custom, one-time-use scripts designed to handle specific tasks. They are typically written in a general-purpose programming language and lack the structure and consistency of specialized IaC tools.
??x

---

#### Configuration Management Tools (Chef, Puppet, Ansible)
Background context: Configuration management tools like Chef, Puppet, and Ansible automate the installation and maintenance of software on servers. These tools enforce consistent structures, making it easier to manage infrastructure across multiple machines.

:p What are configuration management tools?
??x
Configuration management tools are specialized IaC tools designed to install and manage software on existing servers. They provide a structured approach to managing infrastructure, ensuring consistency and idempotency.
??x

---

#### Idempotence in Configuration Management Tools
Background context: Idempotent code performs the same action regardless of how many times it is executed. This property ensures that running the same script multiple times does not lead to unintended side effects.

:p What is idempotence?
??x
Idempotence refers to a function or process where performing an operation multiple times has the same effect as performing it once, ensuring consistent and predictable outcomes.
??x

---

#### Ansible Role Example (web-server.yml)
Background context: An Ansible role provides a structured way to define tasks that are idempotent and easy to manage. The provided example shows how to configure an Apache web server using Ansible.

:p What is the purpose of the `web-server.yml` Ansible role?
??x
The `web-server.yml` Ansible role configures an Apache web server by installing necessary packages, cloning a repository, and starting the service. It ensures that these operations are idempotent.
??x

---

#### Ad Hoc Script Example (setup-webserver.sh)
Background context: The example ad hoc script shows how to set up a web server using Bash commands.

:p What does the `setup-webserver.sh` script do?
??x
The `setup-webserver.sh` script configures a web server by updating the package manager, installing PHP and Apache2, cloning a repository, and starting the Apache service.
```bash
# Update the apt-get cache
sudo apt-get update

# Install PHP and Apache
sudo apt-get install -y php apache2

# Copy the code from the repository
sudo git clone https://github.com/brikis98/php-app.git /var/www/html/app

# Start Apache
sudo service apache2 start
```
x??
This script automates the setup of a web server by executing several commands in sequence, ensuring that each step is idempotent and can be run multiple times without issues.

#### Configuration Management vs. Server Templating Tools

Configuration management tools like Ansible allow for dynamic server configuration, where roles and playbooks can be applied to multiple servers simultaneously or in batches.

:p What is an example of using Ansible to apply a role to multiple servers?
??x
You would create a `hosts` file listing the IP addresses of the servers you want to manage. For instance:
```
[webservers]
11.11.11.11
11.11.11.12
11.11.11.13
11.11.11.14
11.11.11.15
```

Then, define a playbook in `webserver.yml` that specifies the roles to apply:
```yaml
- hosts: webservers
  roles:
    - webserver
```

Finally, you execute this playbook using the command:
```bash
ansible-playbook webserver.yml
```
This configures all five servers as defined. You can also use `serial` parameter to control how many servers get updated at once.

x??

---

#### Server Templating Tools

Server templating tools like Docker, Packer, and Vagrant are used to create self-contained images of operating systems, software, and configurations that can be deployed across multiple servers.

:p What is the primary purpose of server templating tools?
??x
The primary purpose is to create a snapshot or image of a fully configured server environment. This image includes the OS, applications, files, and other relevant details, making it easy to deploy consistent environments across different machines or cloud instances.

For example, using Packer, you can create an Amazon Machine Image (AMI) that encapsulates your application stack:
```json
{
  "builders": [{
    "ami_name": "packer-example-",
    "instance_type": "t2.micro",
    "region": "us-east-2",
    "type": "amazon-ebs",
    "source_ami": "ami-0fb653ca2d3203ac1",
    "ssh_username": "ubuntu"
  }],
  "provisioners": [{
    "type": "shell",
    "inline": [
      "sudo apt-get update",
      "sudo apt-get install -y php apache2",
      "sudo git clone https://github.com/brikis98/php-app.git /var/www/html/app"
    ],
    "environment_vars": ["DEBIAN_FRONTEND=noninteractive"],
    "pause_before": "60s"
  }]
}
```

:p How does Packer build an AMI from the template?
??x
To build an AMI, you run the `packer build` command with your JSON configuration file:
```bash
packer build webserver.json
```
This process creates a VM image with all specified configurations. After the build completes, this image can be deployed on AWS servers.

x??

---

#### Virtual Machines vs. Containers

Virtual machines (VMs) and containers are two types of server templating tools that provide varying degrees of isolation and performance characteristics.

:p What is a virtual machine?
??x
A virtual machine emulates an entire computer system, including hardware, by running on top of a hypervisor like VMware or VirtualBox. Each VM has its own operating system and runs in isolated environments, ensuring consistency across different deployment stages (development, QA, production).

The main benefits include:
- Full isolation from the host and other VMs.
- Consistent behavior regardless of environment.

However, there are significant drawbacks:
- High overhead due to emulating full hardware.
- Longer boot times compared to containers.

x??

---

#### Containers

Containers emulate user space environments of an operating system, providing a lightweight alternative to virtual machines by sharing the host's kernel and hardware resources.

:p What is a container?
??x
A container emulates only the user space environment (e.g., running applications) without the overhead of a full OS. Container engines like Docker, CoreOS rkt, or cri-o run isolated processes, memory, mount points, and networking on top of a shared kernel.

The main benefits include:
- High isolation at the application level.
- Extremely fast boot times due to no need to virtualize hardware.

However, there are some drawbacks:
- Shared kernel and hardware resources can introduce security risks if misconfigured.
- Less robust isolation compared to VMs.

x??

---


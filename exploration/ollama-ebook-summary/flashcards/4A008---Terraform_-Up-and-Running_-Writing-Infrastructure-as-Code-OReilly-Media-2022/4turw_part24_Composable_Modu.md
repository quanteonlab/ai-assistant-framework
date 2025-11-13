# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 24)

**Starting Chapter:** Composable Modules

---

#### Unix Philosophy: Write Programs That Do One Thing and Do It Well

Background context explaining the concept. The Unix philosophy emphasizes simplicity and modularity, encouraging developers to create programs that perform a single task efficiently.

:p What is the main idea of the Unix philosophy mentioned by Doug McIlroy?
??x
The main idea is to write programs that are focused on doing one thing well, rather than creating complex monolithic applications. This approach enhances reusability and maintainability.
x??

---

#### Composable Modules in Terraform

Background context explaining how modules can be used to break down infrastructure into reusable components. The example provided uses two small modules—`asg-rolling-deploy` and `alb`.

:p How do you make modules like `asg-rolling-deploy` and `alb` work together in a composable way?
??x
To make the modules work together, they need to be designed with clear inputs (input variables) and outputs (output variables). This allows them to pass necessary data between each other, making the overall infrastructure more modular and reusable.

For example, if `asg-rolling-deploy` creates an Auto Scaling Group and `alb` creates a Load Balancer, they can share information about resources like security groups or subnets.

```hcl
# asg-rolling-deploy module
variable "security_group_id" {
  description = "The ID of the security group to associate with the ASG"
  type        = string
}

resource "aws_security_group" "example_sg" {
  name        = var.security_group_name
  description = "Security group for ASG"
  vpc_id      = data.aws_vpc.example.id
}
```

```hcl
# alb module
variable "alb_name" {
  description = "The name to use for this ALB"
  type        = string
}

resource "aws_lb" "example" {
  name               = var.alb_name
  load_balancer_type = "application"
  subnets            = data.aws_subnets.default.ids
  security_groups    = [var.security_group_id]
}
```
x??

---

#### Function Composition in Programming

Background context explaining how function composition works, using the Ruby example provided.

:p What is function composition and how can it be applied in programming?
??x
Function composition involves combining multiple functions to create a new, more complex function. In the provided Ruby example, simpler functions like `add`, `sub`, and `multiply` are combined to form a more complex `do_calculation` function.

Here's a brief explanation of how this works:

```ruby
# Simple function to do addition
def add(x, y)
  return x + y
end

# Simple function to do subtraction
def sub(x, y)
  return x - y
end

# Simple function to do multiplication
def multiply (x, y)
  return x * y
end

# Complex function that composes several simpler functions
def do_calculation (x, y)
  return multiply(add(x, y), sub(x, y))
end
```

In this example, `do_calculation` takes two inputs and uses the outputs of other functions as its own parameters.
x??

---

#### Reusability in Terraform Modules

Background context explaining how minimizing side effects can improve code reusability and maintainability.

:p Why is it important to minimize side effects when designing reusable Terraform modules?
??x
Minimizing side effects enhances the reusability of Terraform modules because it ensures that functions are predictable and deterministic. By avoiding reading state from the outside world (side effects) and returning results via output parameters, you can ensure that modules work correctly in different contexts.

For example, a module might be designed to create an Auto Scaling Group, but its functionality should not depend on external states unless passed as input variables. This makes it easier to use the same module in multiple environments without modifications.
x??

---

#### Terraform Module Composition

Background context explaining how to combine simpler modules into more complex ones.

:p How can you build more complicated modules by combining simpler modules in Terraform?
??x
You can build more complicated modules by combining simpler modules using input and output variables. The `asg-rolling-deploy` and `alb` examples show this process:

1. Define inputs (variables) that the simpler modules require.
2. Pass these inputs when calling one module from another.
3. Use outputs of simpler modules as inputs for more complex ones.

For instance, if `asg-rolling-deploy` creates an Auto Scaling Group with a specific security group ID, and `alb` needs this ID to configure its security groups, you can pass the security group ID output by `asg-rolling-deploy` into `alb`.

```hcl
# asg-rolling-deploy module (output)
output "security_group_id" {
  value = aws_security_group.example.id
}

# alb module (input)
variable "security_group_id" {
  description = "The ID of the security group to associate with the ALB"
  type        = string
}
```

x??

---

#### Adding Subnet IDs Variable
Background context: The `subnet_ids` variable allows the module to be used with any VPC or subnets, making it more flexible and reusable. This change is part of transforming a hardcoded deployment approach into a generic one.

:p What does the `subnet_ids` variable enable in the asg-rolling-deploy module?
??x
The `subnet_ids` variable enables the module to be deployed across different VPCs and subnets, providing flexibility and reusability. By defining this variable, users can specify which subnets they want the auto-scaling group (ASG) to target during deployment.

```hcl
variable "subnet_ids" {
  description = "The subnet IDs to deploy to"
  type        = list(string)
}
```
x??

---

#### Adding Target Group ARNs and Health Check Type Variables
Background context: The `target_group_arns` and `health_check_type` variables configure the ASG's integration with load balancers, making it more generic and adaptable. These changes allow for various use cases such as no load balancer, one ALB, multiple NLBs, etc.

:p What do the `target_group_arns` and `health_check_type` variables enable in the module?
??x
The `target_group_arns` and `health_check_type` variables enable the ASG to integrate with different types of load balancers and health check mechanisms. This makes the module more flexible, allowing it to be used in a wide variety of scenarios without being hardcoded for specific resources.

```hcl
variable "target_group_arns" {
  description = "The ARNs of ELB target groups in which to register Instances"
  type        = list(string)
  default     = []
}

variable "health_check_type" {
  description = "The type of health check to perform. Must be one of: EC2, ELB."
  type        = string
  default     = "EC2"
}
```
x??

---

#### Adding User Data Variable
Background context: The `user_data` variable allows for a customizable User Data script, enabling the deployment of any application across an ASG. This makes the module more versatile and adaptable to different use cases.

:p What does the `user_data` variable enable in the asg-rolling-deploy module?
??x
The `user_data` variable enables users to pass in a customized User Data script that can be executed on each instance at boot time. This allows for deploying any application, not just a "Hello, World" app.

```hcl
variable "user_data" {
  description = "The User Data script to run in each Instance at boot"
  type        = string
  default     = null
}
```
x??

---

#### Passing Through Variables to AWS Auto Scaling Group Resource
Background context: The `aws_autoscaling_group` resource now uses the new input variables (`subnet_ids`, `target_group_arns`, and `health_check_type`) instead of hardcoded references. This makes the module more flexible, allowing it to be used with various configurations.

:p How are the new input variables passed through to the AWS Auto Scaling Group resource?
??x
The new input variables (`subnet_ids`, `target_group_arns`, and `health_check_type`) are passed directly into the `aws_autoscaling_group` resource. This allows for more flexibility in deploying the ASG with different subnets, load balancers, and health check types.

```hcl
resource "aws_autoscaling_group" "example" {
  name                 = var.cluster_name
  launch_configuration = aws_launch_configuration.example.name
  vpc_zone_identifier   = var.subnet_ids
  target_group_arns     = var.target_group_arns
  health_check_type     = var.health_check_type
  min_size              = var.min_size
  max_size              = var.max_size
}
```
x??

---

#### Passing Through User Data to AWS Launch Configuration Resource
Background context: The `user_data` variable is passed through to the `aws_launch_configuration` resource, making it possible to deploy any application via the ASG.

:p How is the `user_data` variable passed through to the AWS Launch Configuration resource?
??x
The `user_data` variable is passed directly into the `aws_launch_configuration` resource. This allows for executing a custom User Data script on each instance during boot time, enabling deployment of any application.

```hcl
resource "aws_launch_configuration" "example" {
  image_id         = var.ami
  instance_type    = var.instance_type
  security_groups  = [ aws_security_group.instance.id ]
  user_data        = var.user_data
  lifecycle {
    create_before_destroy = true
  }
}
```
x??

---

#### Adding Output Variables for ASG and Security Group IDs
Background context: Output variables (`asg_name` and `instance_security_group_id`) are added to make the module more reusable. These outputs can be used by consumers of the module to add new behaviors, such as attaching custom rules to the security group.

:p What output variables were added to the ASG deployment?
??x
The following output variables were added:
- `asg_name`: Provides the name of the Auto Scaling Group.
- `instance_security_group_id`: Provides the ID of the EC2 instance security group.

These outputs make the module more reusable by allowing consumers to use these data points for additional configurations, such as attaching custom rules to the security group.

```hcl
output "asg_name" {
  value       = aws_autoscaling_group.example.name
  description = "The name of the Auto Scaling Group"
}

output "instance_security_group_id" {
  value       = aws_security_group.instance.id
  description = "The ID of the EC2 Instance Security Group"
}
```
x??

---

#### Adding Output Variables for ALB
Background context: Output variables (`alb_dns_name`, `alb_http_listener_arn`, and `alb_security_group_id`) are added to the ALB module. These outputs can be used by consumers to add new behaviors or integrate with other components.

:p What output variables were added to the ALB deployment?
??x
The following output variables were added:
- `alb_dns_name`: Provides the domain name of the load balancer.
- `alb_http_listener_arn`: Provides the ARN of the HTTP listener.
- `alb_security_group_id`: Provides the ID of the ALB security group.

These outputs make the module more reusable by allowing consumers to use these data points for additional configurations or integrations, such as attaching custom rules to the security group.

```hcl
output "alb_dns_name" {
  value       = aws_lb.example.dns_name
  description = "The domain name of the load balancer"
}

output "alb_http_listener_arn" {
  value       = aws_lb_listener.http.arn
  description = "The ARN of the HTTP listener"
}

output "alb_security_group_id" {
  value       = aws_security_group.alb.id
  description = "The ALB Security Group ID"
}
```
x??

---

#### Module Renaming and Variable Addition
Background context: The provided text describes how to rename a module from `webserver-cluster` to `hello-world-app`. It also introduces new variables for environment-specific settings. This is crucial for maintaining clear naming conventions and ensuring that different environments (e.g., stage, prod) are well-differentiated.
:p What should you do if you want to use the `hello-world-app` module instead of the old `webserver-cluster` module?
??x
You should rename the existing `module/services/webserver-cluster` directory to `module/services/hello-world-app`. Additionally, in the new `main.tf` file within this renamed directory, add a variable definition for `environment`, which will help name your resources based on the environment (e.g., hello-world-stage, hello-world-prod).
x??

---

#### ASG Rolling Deploy Module Integration
Background context: The text explains how to integrate the existing `asg-rolling-deploy` module into the new `hello-world-app` module. This involves setting up an Auto Scaling Group for deploying a "Hello, World" application.
:p How do you integrate the `asg-rolling-deploy` module within the `hello-world-app` module?
??x
You need to add a module block in the `main.tf` file of `module/services/hello-world-app`. This module block should source the `asg-rolling-deploy` module and pass relevant variables such as `cluster_name`, `ami`, `instance_type`, etc. Here is an example:
```hcl
module "asg" {
   source  = "../../cluster/asg-rolling-deploy"
   cluster_name   = "hello-world-${var.environment}"
   ami           = var.ami
   instance_type  = var.instance_type
   user_data      = templatefile("${path.module}/user-data.sh", {
     server_port  = var.server_port
     db_address   = data.terraform_remote_state.db.outputs.address
     db_port      = data.terraform_remote_state.db.outputs.port
     server_text  = var.server_text
   })
   min_size            = var.min_size
   max_size            = var.max_size
   enable_autoscaling  = var.enable_autoscaling
   subnet_ids         = data.aws_subnets.default.ids
   target_group_arns  = [aws_lb_target_group.asg.arn]
   health_check_type  = "ELB"
   custom_tags        = var.custom_tags
}
```
x??

---

#### ALB Module Integration
Background context: The text describes how to integrate the `alb` module into the new `hello-world-app` module. This involves setting up a load balancer for your application.
:p How do you add the `alb` module within the `hello-world-app` module?
??x
You should add a module block in the `main.tf` file of `module/services/hello-world-app`. This block sources the `alb` module and sets necessary variables such as `alb_name`, `subnet_ids`, etc. Here is an example:
```hcl
module "alb" {
   source  = "../../networking/alb"
   alb_name    = "hello-world-${var.environment}"
   subnet_ids  = data.aws_subnets.default.ids
}
```
x??

---

#### Target Group and Listener Rule Update
Background context: The text explains how to update the target group and listener rule for your application's load balancer. This ensures that traffic is correctly routed to the Auto Scaling Group.
:p How do you update the `aws_lb_target_group` resource in `hello-world-app` to use the environment variable?
??x
You need to modify the name of the target group resource to include the `environment` variable. Here is an updated example:
```hcl
resource "aws_lb_target_group" "asg" {
   name     = "hello-world-${var.environment}"
   port     = var.server_port
   protocol = "HTTP"
   vpc_id    = data.aws_vpc.default.id
   health_check  {
     path                 = "/"
     protocol             = "HTTP"
     matcher              = "200"
     interval             = 15
     timeout              = 3
     healthy_threshold    = 2
     unhealthy_threshold  = 2
   }
}
```
x??

---

#### Listener Rule Configuration Update
Background context: The text explains how to configure the listener rule for your load balancer. This ensures that HTTP requests are correctly routed to the target group.
:p How do you update the `aws_lb_listener_rule` resource in `hello-world-app` to use the ALB's HTTP listener ARN?
??x
You need to set the `listener_arn` parameter of the listener rule to point at the output `alb_http_listener_arn` from the `alb` module. Here is an example:
```hcl
resource "aws_lb_listener_rule" "asg" {
   listener_arn  = module.alb.alb_http_listener_arn
   priority      = 100
   condition    {
     path_pattern  {
       values  = ["*"]
     }
   }
   action {
     type              = "forward"
     target_group_arn  = aws_lb_target_group.asg.arn
   }
}
```
x??

---

#### Outputs Configuration for the `hello-world-app` Module
Background context: The text explains how to configure outputs from the `asg-rolling-deploy` and `alb` modules as outputs of the `hello-world-app` module. This allows you to retrieve important information such as the ALB DNS name.
:p How do you configure outputs in the `hello-world-app` module?
??x
You need to define outputs for the ALB's DNS name. Here is an example:
```hcl
output "alb_dns_name" {
   value       = module.alb.alb_dns_name
}
```
This allows users of this module to retrieve the ALB's DNS name once it has been provisioned.
x??

#### Testable Modules
Background context: The provided text discusses creating a test harness for infrastructure modules using Terraform. This involves writing example configurations that use your modules to ensure they function as expected before deploying them in production.

:p How can you create and run an example configuration to test your infrastructure modules?
??x
You can create an `examples` folder with sample `main.tf` files that utilize the modules you have written. For instance, the provided code shows how to use the `asg-rolling-deploy` module to deploy a small Auto Scaling Group (ASG) in a specific region and with certain parameters.

To test this configuration:
1. Run `terraform init` to initialize the backend and provider.
2. Run `terraform apply` to create the resources defined in your `main.tf`.

Here's an example of such a configuration for testing:

```hcl
provider "aws" {
  region = "us-east-2"
}

module "asg" {
  source     = "../../modules/cluster/asg-rolling-deploy"
  cluster_name   = var.cluster_name
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  min_size      = 1
  max_size      = 1
  enable_autoscaling = false
  subnet_ids    = data.aws_subnets.default.ids
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name    = "vpc-id"
    values  = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name    = "name"
    values  = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```

This configuration sets up a minimal ASG with one instance in the default VPC of `us-east-2`. Run these commands to test this setup:

```sh
terraform init
terraform apply
```

If everything works as expected, you can proceed to deploy this in production.

x??

---

#### Manual Test Harness
Background context: The text explains how creating an example configuration within the `examples` folder serves as a manual test harness. This means that developers can repeatedly run `terraform apply` and `terraform destroy` commands to check if their modules behave as expected during development.

:p How does using the `examples` folder serve as a manual test harness?
??x
The `examples` folder provides a practical way for developers to manually verify that their infrastructure modules work correctly. By writing small, specific configurations in `main.tf` files within this directory, you can deploy and destroy resources on demand.

For example, if you have an `asg-rolling-deploy` module, you could create an `examples/asg/main.tf` file like the one shown earlier:

```hcl
provider "aws" {
  region = "us-east-2"
}

module "asg" {
  source     = "../../modules/cluster/asg-rolling-deploy"
  cluster_name   = var.cluster_name
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  min_size      = 1
  max_size      = 1
  enable_autoscaling = false
  subnet_ids    = data.aws_subnets.default.ids
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name    = "vpc-id"
    values  = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name    = "name"
    values  = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```

You can then manually run:

```sh
terraform init
terraform apply
terraform destroy
```

This process allows you to quickly iterate and debug your modules without needing a full CI/CD pipeline.

x??

---

#### Automated Test Harness
Background context: The text mentions that the `examples` folder can also be used for automated testing. As described in Chapter 9, these example configurations are how tests are created for your modules, ensuring they behave as expected even when the code changes.

:p How do you use the `examples` folder to create an automated test harness?
??x
The `examples` folder not only serves as a manual test harness but can also be used to write automated tests. By committing example configurations and their corresponding tests into version control, you ensure that your modules' behavior remains consistent over time.

To set up automated testing:
1. Write tests in the `test` folder.
2. Use these test files alongside the examples in the `examples` directory.
3. Run these tests as part of a CI/CD pipeline to automatically validate module changes.

For instance, you might create a test file named `asg_test.tf` in the `test` directory:

```hcl
provider "aws" {
  region = "us-east-2"
}

module "asg" {
  source     = "../../modules/cluster/asg-rolling-deploy"
  cluster_name   = var.cluster_name
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  min_size      = 1
  max_size      = 1
  enable_autoscaling = false
  subnet_ids    = data.aws_subnets.default.ids
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name    = "vpc-id"
    values  = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name    = "name"
    values  = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```

Then, you can use Terraform's `test` command to run these tests as part of your CI/CD pipeline.

x??

---

#### Executable Documentation
Background context: The text highlights the value of including example configurations and a README in version control. This approach allows team members to understand how modules work and test them without writing additional code, making it both an educational tool and a way to ensure documentation remains accurate.

:p How does having examples and a README improve your module's maintainability?
??x
Having examples and a README in version control significantly enhances the maintainability of your infrastructure modules. These files provide executable documentation that can be used by developers to understand and test how the modules work, ensuring that they are correctly implemented and functioning as intended.

For instance, you might include a `README.md` file with instructions on how to use and test the module:

```markdown
# asg-rolling-deploy Module Example

## Usage

1. Ensure Terraform is initialized:
   ```sh
   terraform init
   ```

2. Apply the example configuration:
   ```sh
   terraform apply
   ```

3. Destroy the resources when done:
   ```sh
   terraform destroy
   ```

## Example `main.tf` Configuration

```hcl
provider "aws" {
  region = "us-east-2"
}

module "asg" {
  source     = "../../modules/cluster/asg-rolling-deploy"
  cluster_name   = var.cluster_name
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  min_size      = 1
  max_size      = 1
  enable_autoscaling = false
  subnet_ids    = data.aws_subnets.default.ids
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name    = "vpc-id"
    values  = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name    = "name"
    values  = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```

By committing these files into version control, you ensure that other team members can easily understand and test the modules.

x??

---

---

#### Module Example and Testing Structure
Module development often involves a specific folder structure for organizing modules, their examples, and tests. This organization ensures that each module has corresponding documentation and testing.

A typical structure might look like:
```
modules/
└ examples/
    └ alb/
    └ asg-rolling-deploy/
      └ one-instance/
      └ auto-scaling/
      └ with-load-balancer/
      └ custom-tags/
    └ hello-world-app/
    └ mysql/
  └ modules/
    └ alb/
    └ asg-rolling-deploy/
    └ hello-world-app/
    └ mysql/
└ test/
    └ alb/
    └ asg-rolling-deploy/
    └ hello-world-app/
    └ mysql/
```

:p How is the structure for organizing modules, examples, and tests typically set up in a Terraform project?
??x
The structure ensures that each module has corresponding examples and tests. Examples demonstrate different configurations of how a module can be used, while tests verify the behavior of these configurations.

```terraform
# Example folder structure
modules/
└ examples/
    └ asg-rolling-deploy/
      └ one-instance/  # One instance configuration example
      └ auto-scaling/  # Auto-scaling policy example
```

x??

---

#### Test-Driven Development (TDD) in Terraform Module Development
Writing tests and examples first can lead to a better understanding of the module's API and usage patterns. This approach is known as Test-Driven Development (TDD).

:p Why should you write test code before writing module code?
??x
Writing test code first helps in designing an intuitive and user-friendly API for the module. It allows developers to think through ideal use cases and APIs while ensuring that tests are written early, making it easier to validate behavior.

```terraform
# Example of a validation block
variable "instance_type" {
  description = "The type of EC2 Instances to run (e.g. t2.micro)"
  type        = string
  validation {
    condition      = contains(["t2.micro", "t3.micro"], var.instance_type)
    error_message  = "Only free tier is allowed: t2.micro | t3.micro."
  }
}
```

x??

---

#### Validations in Terraform
Starting from Terraform 0.13, you can add validation blocks to input variables to perform checks beyond basic type constraints.

:p How does the `validation` block work in Terraform?
??x
The `validation` block allows for more complex checks than simple type constraints. The `condition` parameter should evaluate to true if the value is valid and false otherwise. If the condition fails, an error message can be provided.

```terraform
# Example of a validation block
variable "instance_type" {
  description = "The type of EC2 Instances to run (e.g. t2.micro)"
  type        = string
  validation {
    condition      = contains(["t2.micro", "t3.micro"], var.instance_type)
    error_message  = "Only free tier is allowed: t2.micro | t3.micro."
  }
}
```

x??

---

---

#### Validation vs. Precondition Blocks
Background context: In Terraform, validation blocks are used to validate input variables at apply time and provide error messages if a condition is not met. However, they have limitations such as being unable to reference other input variables or perform dynamic checks across multiple variables. To address these limitations, Terraform introduced precondition and postcondition blocks in version 1.2.

:p What is the difference between validation and precondition blocks?
??x
Validation blocks are used for basic input sanitization, checking conditions like ensuring a variable's value meets certain criteria (e.g., being greater than zero) at apply time. They cannot reference other variables or perform more complex checks dynamically. On the other hand, precondition blocks allow you to enforce more robust and dynamic checks before Terraform applies changes, such as verifying if an instance type is part of the AWS Free Tier.
x??

---

#### Using `aws_ec2_instance_type` Data Source
Background context: The `aws_ec2_instance_type` data source can be used to retrieve information about EC2 instance types from AWS. This includes attributes like whether a specific instance type is eligible for the AWS Free Tier.

:p How do you use the `aws_ec2_instance_type` data source in Terraform?
??x
To use the `aws_ec2_instance_type` data source, you need to define it and specify the instance type as an argument. Then, you can reference its attributes within your resource blocks or other parts of your configuration.

```terraform
data "aws_ec2_instance_type" "instance" {
  instance_type = var.instance_type
}

resource "aws_launch_configuration" "example" {
  # Other properties...
  
  lifecycle {
    create_before_destroy = true
    precondition {
      condition      = data.aws_ec2_instance_type.instance.free_tier_eligible
      error_message   = "${var.instance_type} is not part of the AWS Free Tier."
    }
  }
}
```
x??

---

#### Precondition Block for Validating Instance Types
Background context: A precondition block can be used to ensure that certain conditions are met before Terraform applies a resource. In this example, we're validating whether an instance type is eligible for the AWS Free Tier.

:p What does the `precondition` block do in this scenario?
??x
The `precondition` block checks if the specified instance type is eligible for the AWS Free Tier before applying the resources. If the condition evaluates to false, Terraform will throw an error with a custom message.

Example:
```terraform
resource "aws_launch_configuration" "example" {
  # Other properties...
  
  lifecycle {
    create_before_destroy = true
    precondition {
      condition      = data.aws_ec2_instance_type.instance.free_tier_eligible
      error_message   = "${var.instance_type} is not part of the AWS Free Tier."
    }
  }
}
```
If `var.instance_type` does not belong to the AWS Free Tier, Terraform will raise an error:
```plaintext
Error: Resource precondition failed

data.aws_ec2_instance_type.instance.free_tier_eligible is false
${var.instance_type} is not part of the AWS Free Tier.
```
x??

---

---
#### Postcondition Blocks for Error Checking
Postcondition blocks are used to ensure certain conditions hold true after a resource is applied. This is particularly useful in ensuring that resources are deployed correctly and meet specific requirements post-deployment.

For example, consider an `aws_autoscaling_group` resource where you want to ensure the Auto Scaling Group spans multiple availability zones (AZs) for high availability:
```hcl
resource "aws_autoscaling_group" "example" {
  name                 = var.cluster_name
  launch_configuration = aws_launch_configuration.example.name
  vpc_zone_identifier  = var.subnet_ids

  lifecycle {
    postcondition {
      condition      = length(self.availability_zones) > 1
      error_message  = "You must use more than one AZ for high availability."
    }
  }

  # Additional configuration...
}
```

:p What is the purpose of using a `postcondition` block in an AWS Auto Scaling Group resource?
??x
The purpose of using a `postcondition` block is to ensure that after the Auto Scaling Group is deployed, it spans more than one availability zone. This guarantees high availability by ensuring that even if one AZ fails, other instances can still be utilized.

This check ensures that Terraform will show an error during `apply` if the deployment does not meet this requirement.
```hcl
lifecycle {
    postcondition {
        condition      = length(self.availability_zones) > 1
        error_message  = "You must use more than one AZ for high availability."
    }
}
```
x??

---
#### Validations for Input Sanitization
Validation blocks are used to sanitize inputs, preventing users from passing invalid variables into the module. This is crucial for catching basic input errors early in the process.

:p How do validation blocks help prevent issues with Terraform modules?
??x
Validation blocks help by ensuring that any variable passed into a Terraform module meets certain criteria before changes are applied. By using validation blocks, you can catch and prevent invalid inputs, thus avoiding deployment failures due to basic configuration errors.

For example:
```hcl
variable "subnet_ids" {
  description = "List of subnet IDs"
  type        = list(string)
  validation {
    condition     = length(element(var.subnet_ids, 0)) > 2
    error_message = "Each subnet ID must be a non-empty string."
  }
}
```
x??

---
#### Precondition Blocks for Assumption Checks
Precondition blocks are used to verify assumptions about the state of resources and variables before any changes are made. They help in catching issues early that could otherwise lead to deployment failures.

:p How do precondition blocks work?
??x
Precondition blocks allow you to check various conditions, such as dependencies between variables or data sources, before Terraform applies any changes. This helps ensure that the state of your resources and configurations is correct prior to making changes.

For example:
```hcl
precondition {
  condition      = length(var.subnet_ids) > 0 && var.region == "us-east-1"
  error_message  = "You need at least one subnet ID in us-east-1 region."
}
```
x??

---
#### Postconditions for Enforcing Basic Guarantees
Postcondition blocks are used to ensure that the module behaves as expected after deployment. They provide confidence that the module either performs its intended function or exits with an error if it doesn't.

:p What is the purpose of using postcondition blocks in Terraform modules?
??x
The purpose of using postcondition blocks is to enforce basic guarantees about how your module behaves after changes have been deployed. This gives users and maintainers confidence that the module will either perform its intended function or exit with an error if it doesn't.

For example, ensuring that a web service can respond to HTTP requests:
```hcl
lifecycle {
  postcondition {
    condition      = true # Logic to check if service is responding
    error_message  = "Web service did not start correctly."
  }
}
```
x??

---

---
#### Versioning of Terraform Core
Background context: When working on production-grade Terraform code, it is essential to ensure that deployments are predictable and repeatable. One way to achieve this is by version pinning your dependencies, starting with the Terraform core version.

:p How do you ensure the correct version of Terraform core is used in your modules?
??x
To ensure the correct version of Terraform core is used, you can use the `required_version` argument in your Terraform configuration. This allows you to specify a specific major or minor version of Terraform that your code depends on.

```terraform
terraform {
  # Require any 1.x version of Terraform
  required_version = ">= 1.0.0, < 2.0.0"
}
```

This will allow you to use only versions from the `1.x` series, such as `1.0.0` or `1.2.3`. If you try to run a Terraform version outside this range (e.g., `0.14.3` or `2.0.0`), you will receive an error.

For production-grade code, it is recommended to pin not only the major version but also the minor and patch versions:

```terraform
terraform {
  # Require Terraform at exactly version 1.2.3
  required_version = "1.2.3"
}
```

This ensures that you are using a specific version of Terraform, avoiding accidental upgrades to potentially incompatible versions.
x??

---
#### Versioning of Providers
Background context: In addition to pinning the core version of Terraform, it is also important to manage the versions of providers used in your modules. Providers define how resources from different services (like AWS or GCP) are managed.

:p How do you specify the version of a provider in your Terraform configuration?
??x
To specify the version of a provider in your Terraform configuration, you include it within the `provider` block. For example, if you are using the AWS provider, you would define its version as follows:

```terraform
provider "aws" {
  # Specify the version of the provider
  version = "=2.64.0"
}
```

Here, `version = "=2.64.0"` ensures that only the exact version `2.64.0` is used. If you want to use a broader range, you can specify it like this:

```terraform
provider "aws" {
  # Specify a range of provider versions
  version = "[2.58.0,2.70.0)"
}
```

This means the AWS provider version must be at least `2.58.0` but less than `2.70.0`.

Versioning providers is crucial to maintain compatibility and avoid breaking changes in future releases.
x??

---
#### Versioning of Modules
Background context: Module versioning ensures that your code remains consistent across deployments by specifying the exact versions of modules you depend on. This helps prevent accidental upgrades to incompatible module versions.

:p How do you specify the version of a module in Terraform?
??x
To specify the version of a module, you include it within the `module` block. For example:

```terraform
module "example_module" {
  # Specify the exact version of the module
  source = "git://github.com/your_org/your_module"
  version = "1.2.0"
}
```

Here, `version = "1.2.0"` ensures that only the specific version `1.2.0` is used for the module named `example_module`.

You can also use a range of versions if needed:

```terraform
module "example_module" {
  # Specify a range of module versions
  source = "git://github.com/your_org/your_module"
  version = "[1.1.0,1.3.0)"
}
```

This means the `example_module` must be at least `1.1.0` but less than `1.3.0`.

Versioning modules is essential to maintain consistency and avoid unintended changes.
x??

---

#### Pinning Terraform Versions
Background context: When using different versions of Terraform, it can lead to issues when mixing environments. To avoid these problems and test new features or bug fixes, pinning specific versions is recommended.
:p How do you ensure consistent use of a specific version of Terraform across an environment?
??x
To ensure consistency, you can use the `tfenv` tool to manage different versions of Terraform. This involves installing a specific version using the command `tfenv install <version>`, and then setting up your project to use that version.
```sh
$tfenv install 1.2.3
```
Once installed, you can set the default version for your environment by running:
```sh$ tfenv use 1.2.3
```
You can also specify versions in `.terraform-version` files within project directories to automatically use a specific version in that directory and its subdirectories.
x??

---

#### Using .terraform-version Files
Background context: To manage multiple versions of Terraform, `tfenv` allows the use of `.terraform-version` files. These files enable you to specify which version of Terraform should be used in different parts of your project or environment.
:p How do you configure a specific Terraform version for a particular directory using `tfenv`?
??x
To configure a specific Terraform version for a particular directory, create or modify the `.terraform-version` file within that directory. The content of this file should be the desired version number of Terraform.

For example:
```sh
$echo "1.2.3" > stage/vpc/.terraform-version
```
This sets the `stage/vpc` directory and its subdirectories to use Terraform version 1.2.3 by default.
x??

---

#### Managing Different Versions on Apple Silicon (M1, M2)
Background context: As of June 2022, `tfenv` had issues installing the correct version of Terraform for Apple Silicon Macs (M1 or M2 processors). To address this, you can set the environment variable `TFENV_ARCH`.
:p How do you work around the issue with `tfenv` on Apple Silicon?
??x
To work around the issue with `tfenv` on Apple Silicon, you need to manually set the `TFENV_ARCH` environment variable to `arm64`. This tells `tfenv` to use the ARM architecture version of Terraform.

The steps are:
1. Export the `TFENV_ARCH` variable:
   ```sh$ export TFENV_ARCH=arm64
   ```
2. Install the desired version using `tfenv install <version>`.

For example:
```sh
$export TFENV_ARCH=arm64$ tfenv install 1.2.3
```
This will ensure that `tfenv` installs the correct version of Terraform for Apple Silicon.
x??

---

#### Using .terraform-version Files Across Environments
Background context: The `.terraform-version` files allow you to specify different versions of Terraform for different environments within a project. This is particularly useful when testing new features or bug fixes in pre-production environments before deploying them to production.
:p How do you set up different Terraform versions for the `stage` and `prod` environments?
??x
To set up different Terraform versions for the `stage` and `prod` environments, create or modify `.terraform-version` files within these directories.

For example:
- In the `live/stage/` directory:
  ```sh
  $echo "1.2.3" > stage/.terraform-version
  ```
- In the `live/prod/` directory:
  ```sh$ echo "1.0.0" > prod/.terraform-version
  ```

These files will instruct `tfenv` to use Terraform version 1.2.3 in the `stage` environment and version 1.0.0 in the `prod` environment.
x??

---

#### Pinning Provider Versions
Background context: In addition to managing Terraform versions, it is crucial to pin provider versions to avoid breaking changes or unexpected behavior when upgrading providers. The `required_providers` block within the Terraform configuration allows you to specify which version of a provider should be used.
:p How do you pin the AWS provider version in your Terraform configuration?
??x
To pin the AWS provider version in your Terraform configuration, use the `required_providers` block as shown below. This example pins the AWS provider to any 4.x version.

```hcl
terraform {
   required_version = ">= 1.0.0, < 2.0.0"
   required_providers {
     aws = {
       source = "hashicorp/aws"
       version = "~> 4.0"
     }
   }
}
```
The `version = "~> 4.0"` syntax ensures that the AWS provider will use any minor updates within the 4.x series but won't upgrade to a major update (5.0 or later).
x??

---

#### Pinning Provider Versions to a Specific Major Version Number
Background context explaining that pinning to a specific major version number is recommended to avoid accidentally pulling in backward-incompatible changes. With Terraform 0.14.0 and above, minor or patch versions are automatically handled due to the lock file.
:p What should be done if you want to avoid accidental backward-incompatible changes with providers?
??x
You should pin to a specific major version number of the provider in your `required_providers` block. For example:
```hcl
required_providers {
  myprovider = {
    source  = "hashicorp/myprovider"
    version = "~> 2.0"  # Pinning to any version >= 2.0 and < 3.0
  }
}
```
This ensures that only changes compatible with the major version are pulled in, avoiding potential issues.
x??

---

#### Lock File for Provider Version Control
Background context explaining how `terraform init` creates a `.terraform.lock.hcl` file to record exact provider versions used during initialization. This file is checked into version control to ensure consistency across environments.
:p What does Terraform create the first time you run `terraform init`?
??x
Terraform creates a `.terraform.lock.hcl` file, which records the exact version of each provider used in your configuration.
This file should be checked into version control so that when you or another developer runs `terraform init` again on any computer, Terraform will download the same versions of providers as initially configured.
x??

---

#### Upgrading Provider Versions Explicitly
Background context explaining how the lock file ensures consistency but does not need minor and patch pinning from version 0.14.0 onwards due to automatic behavior. Explicit upgrades can be done by modifying the `required_providers` block and running `terraform init -upgrade`.
:p How do you explicitly upgrade a provider version in Terraform?
??x
You update the version constraint in the `required_providers` block, for example:
```hcl
required_providers {
  myprovider = {
    source  = "hashicorp/myprovider"
    version = "~> 2.1"  # Pinning to any version >= 2.1 and < 3.0
  }
}
```
Then run `terraform init -upgrade` to download new versions of the providers that match your updated constraints.
The `.terraform.lock.hcl` file will be updated with these changes, which should then be reviewed and committed to version control.
x??

---

#### Security Measures via Checksums
Background context explaining how Terraform records checksums for downloaded providers to ensure integrity. It also mentions validating signatures if the provider is cryptographically signed.
:p How does Terraform ensure that the provider code hasn't been tampered with?
??x
Terraform records the checksum of each provider it downloads and checks these against the recorded values during subsequent `terraform init` runs. This ensures that any changes to the provider binaries are detected, preventing malicious code from being substituted.
If the provider is cryptographically signed (most official HashiCorp providers are), Terraform also validates the signature as an additional security check.
x??

---

#### Lock Files Across Multiple Operating Systems
Background context explaining that by default, Terraform only records checksums for the platform on which `terraform init` was run. If this file is shared across multiple OSes, a command like `terraform providers lock -platform=...` needs to be run to record checksums from all relevant platforms.
:p How do you ensure that the `.terraform.lock.hcl` file works across different operating systems?
??x
You need to run `terraform providers lock` with the `-platform` option for each OS on which the code will run. For example:
```sh
terraform providers lock \
  -platform=windows_amd64 \ 
  # 64-bit Windows  
  -platform=darwin_amd64 \  # 64-bit macOS  
  -platform=darwin_arm64 \  # 64-bit macOS (ARM)  
  -platform=linux_amd64     # 64-bit Linux
```
This command records the checksums for each platform, ensuring that `terraform init` on any of these systems will download the correct versions.
x??

---

#### Pinning Module Versions Using Git Tags
Background context explaining why it's important to pin module versions using source URLs with a specific ref parameter. This ensures consistency across different environments when initializing modules.
:p How do you pin a module version in Terraform?
??x
You should use the `source` URL along with the `ref` parameter set to a Git tag or branch, like:
```hcl
module "hello_world" {
  source  = "git@github.com:foo/modules.git//services/hello-world-app?ref=v0.0.5"
}
```
This ensures that every time you run `terraform init`, the exact same version of the module is downloaded and used, maintaining consistency.
x??

---

#### Versioning Code Using Git Tags
Background context explaining how versioning can be applied to software development using Git tags. Semantic versioning is a common approach, where version numbers are structured as `x.y.z`.

:p How do you create and push a Git tag for semantic versioning?
??x
To create and push a Git tag with semantic versioning, use the following commands:
```sh
$git tag -a "v0.0.5" -m "Create new hello-world-app module"$ git push --follow-tags
```
These commands create an annotated tag named `v0.0.5` and push it along with any associated tags.
x??

---

#### Deploying a Versioned Module to Staging Environment
Explanation of deploying a specific version of a Terraform module in the staging environment using configuration files.

:p How do you configure Terraform to deploy a specific version of your module?
??x
To deploy a specific version (e.g., `v0.0.5`) of your module, update the `source` argument in the module block with the correct Git tag:

```hcl
module "hello_world_app"  {
   # TODO: replace this with your own module URL and version..
   source = "git@github.com:foo/modules.git//services/hello-world-app?ref=v0.0.5"
   server_text             = "New server text"
   environment             = "stage"
   db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
   db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
   instance_type           = "t2.micro"
   min_size                = 2
   max_size                = 2
   enable_autoscaling      = false
   ami                     = data.aws_ami.ubuntu.id
}
```

Then, initialize and apply the Terraform configuration:

```sh
$terraform init$ terraform apply
```
x??

---

#### Outputting Deployment Information
Explanation of how to output deployment information such as the ALB DNS name for monitoring or testing purposes.

:p How do you configure an output in Terraform to display the ALB DNS name?
??x
To configure an output that displays the ALB DNS name, add the following block to your `outputs.tf` file:

```hcl
output "alb_dns_name"  {
   value       = module.hello_world_app.alb_dns_name
   description = "The domain name of the load balancer"
}
```

After running `terraform apply`, you will see the ALB DNS name outputted as part of the deployment:

```sh
Outputs:
alb_dns_name = "hello-world-stage-477699288.us-east-2.elb.amazonaws.com"
```
x??

---

#### Publishing Modules to Terraform Registry
Explanation on how to publish a module to the Public Terraform Registry and the requirements.

:p What are the steps for publishing a module to the Public Terraform Registry?
??x
To publish a module to the Public Terraform Registry, follow these steps:

1. Ensure your module is hosted in a public GitHub repository.
2. Name your repository `terraform-provider-name` where `provider` is the target provider (e.g., `aws`) and `name` is the name of the module (e.g., `rds`).
3. Structure your module directory as follows:
   - `main.tf`
   - `variables.tf`
   - `outputs.tf`
4. Use semantic versioning with Git tags for releases.
5. Log in to the Terraform Registry using your GitHub account and use the web UI to publish.

Once published, you can share it with your team via the web UI or through the registry.
x??

---


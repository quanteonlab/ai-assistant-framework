# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 11)

**Starting Chapter:** Module Inputs

---

#### Defining Input Variables in Terraform Modules
Background context: In Terraform, modules can have input parameters that control their behavior across different environments. These variables are defined in the `variables.tf` file and used within the module's configuration files to make resources dynamic.

:p What is a variable in the context of Terraform modules?
??x
A variable in Terraform modules acts as an input parameter, allowing you to customize the behavior of your infrastructure configuration based on different environments. Variables are defined using the `variable` keyword and can be of various types such as string, number, list, etc.
x??

---
#### Adding Input Variables for Cluster Name
:p How do you add a cluster name variable in Terraform?
??x
To add a cluster name variable, you define it in the `variables.tf` file using the `variable` keyword. For example:

```terraform
variable "cluster_name" {
   description = "The name to use for all the cluster resources"
   type        = string
}
```

This creates a new input parameter named `cluster_name`, which can be of type `string`. This variable can then be used throughout the module configuration.
x??

---
#### Using Input Variables in Resource Names
:p How do you update resource names using input variables in Terraform?
??x
To use an input variable for resource naming, replace hardcoded values with `${var.input_variable}`. For example:

```terraform
resource "aws_security_group" "alb" {
   name = "${var.cluster_name}-alb"
}
```

This ensures that the `name` of the security group is dynamic based on the value provided by `cluster_name`. Similarly, update other resource names such as the auto-scaling group and ALB with the same variable.
x??

---
#### Setting Remote State Parameters
:p How do you configure remote state parameters in Terraform?
??x
To configure remote state parameters, use the `terraform_remote_state` data source. For instance:

```terraform
data "terraform_remote_state" "db" {
   backend = "s3"
   config = {
     bucket  = var.db_remote_state_bucket
     key     = var.db_remote_state_key
     region  = "us-east-2"
   }
}
```

This sets the `bucket` and `key` for remote state based on input variables. Replace hardcoded values with `${var.input_variable}` to make them dynamic.
x??

---
#### Configuring Instance Type, Min Size, Max Size
:p What are instance_type, min_size, max_size in Terraform?
??x
Instance type (`instance_type`), minimum size (`min_size`), and maximum size (`max_size`) are input variables used to customize the behavior of EC2 instances and auto-scaling groups. They are defined as follows:

```terraform
variable "instance_type" {
   description = "The type of EC2 Instances to run (e.g., t2.micro)"
   type        = string
}

variable "min_size" {
   description = "The minimum number of EC2 Instances in the ASG"
   type        = number
}

variable "max_size" {
   description = "The maximum number of EC2 Instances in the ASG"
   type        = number
}
```

These variables allow you to set different configurations for instance types and scaling limits based on your environment needs.
x??

---
#### Applying Instance Type, Min Size, Max Size in Staging Environment
:p How do you configure a small web server cluster in the staging environment?
??x
In the `main.tf` file of the staging environment (e.g., `stage/services/webserver-cluster/main.tf`), set the input variables for instance type and scaling limits to appropriate values:

```terraform
module "webserver_cluster" {
   source          = "../../../modules/services/webserver-cluster"
   cluster_name    = "webservers-stage"
   db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
   db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
   instance_type         = "t2.micro"
   min_size              = 2
   max_size              = 2
}
```

This configuration ensures that the web server cluster is set up with smaller instances and a limited number of instances, suitable for testing.
x??

---
#### Applying Instance Type, Min Size, Max Size in Production Environment
:p How do you configure a larger web server cluster in the production environment?
??x
In the `main.tf` file of the production environment (e.g., `prod/services/webserver-cluster/main.tf`), set the input variables for instance type and scaling limits to appropriate values:

```terraform
module "webserver_cluster" {
   source          = "../../../modules/services/webserver-cluster"
   cluster_name    = "webservers-prod"
   db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
   db_remote_state_key     = "prod/data-stores/mysql/terraform.tfstate"
   instance_type         = "m4.large"
   min_size              = 2
   max_size              = 10
}
```

This configuration allows for a larger, more powerful instance type and a higher maximum number of instances to handle increased traffic.
x??

---

#### Local Variables in Terraform Modules

Background context: In Terraform, using local variables within a module can help manage and encapsulate reusable values for resources. This approach keeps your code more readable, maintainable, and DRY (Don't Repeat Yourself). Local variables are only accessible within the module they are defined in and cannot be overridden from outside.

:p How do you define and use local variables in Terraform modules to avoid hardcoding values?
??x
You can define local variables using a `locals` block. For example, in the `webserver-cluster` module, you might have:

```hcl
locals {
  http_port     = 80
  any_port      = 0
  any_protocol  = "-1"
  tcp_protocol  = "tcp"
  all_ips       = ["0.0.0.0/0"]
}
```

To use a local variable, you reference it with the `local.<NAME>` syntax. For instance:

```hcl
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.example.arn
  port              = local.http_port
  protocol          = "HTTP"
  # By default, return a simple 404 page
  default_action {
    type = "fixed-response"
    fixed_response {
      content_type  = "text/plain"
      message_body  = "404: page not found"
      status_code   = 404
    }
  }
}
```

This approach ensures that the `http_port` value is reused consistently throughout your module without hardcoding it.

x??

---
#### Using Local Variables for Security Group Rules

Background context: Local variables can be used to define security group rules in a Terraform module. This helps maintain consistency and reduces redundancy by defining common values once and reusing them across different resources like ingress and egress rules.

:p How do you update the `aws_security_group` resource to use local variables for its rules?
??x
You can reference local variables within the security group rules as follows:

```hcl
resource "aws_security_group" "alb" {
  name = "${var.cluster_name}-alb"
  ingress {
    from_port    = local.http_port
    to_port      = local.http_port
    protocol     = local.tcp_protocol
    cidr_blocks  = local.all_ips
  }
  egress {
    # Default: Allow all traffic from and to anywhere
    from_port    = 0
    to_port      = 0
    protocol     = "-1"
    cidr_blocks  = ["0.0.0.0/0"]
  }
}
```

Here, `local.http_port` is used for the port range in both ingress and egress rules, ensuring that the HTTP service can be accessed on port 80.

x??

---

#### Using Local Values for Ease of Maintenance
Local values can be used to make Terraform configurations more readable and maintainable. By defining variables like `local.any_port`, `local.any_protocol`, and `local.all_ips` in your configuration, you avoid hardcoding these values directly into the resource blocks.
:p How do local values improve Terraform configurations?
??x
Local values help by allowing you to reuse common settings across multiple resources without repeating them. This makes your code easier to read and maintain. For example:
```hcl
locals {
  any_port = 80
  any_protocol = "tcp"
  all_ips = "0.0.0.0/0"
}
```
You can then reference these locals in resource blocks like this:
```hcl
resource "aws_security_group_rule" "example" {
    from_port      = local.any_port
    to_port        = local.any_port
    protocol       = local.any_protocol
    cidr_blocks    = [local.all_ips]
}
```
x??

---

#### Defining Auto Scaling Schedules Directly in Production
Auto scaling schedules can be used to adjust the number of servers based on traffic patterns. In this example, you define two scheduled actions: one for scaling out during business hours and another for scaling in at night.
:p What are the benefits of defining auto scaling schedules directly in production configurations?
??x
Defining auto scaling schedules directly in production allows for immediate control over server scalability without needing to manage these actions through separate modules. This can be useful when you want more granular control or when conditional definitions (like moving scheduled actions into a module) are not yet required.
```hcl
resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
    scheduled_action_name = "scale-out-during-business-hours"
    min_size               = 2
    max_size               = 10
    desired_capacity       = 10
    recurrence             = "0 9 * * *" # This means "9 a.m. every day"
}
resource "aws_autoscaling_schedule" "scale_in_at_night" {
    scheduled_action_name = "scale-in-at-night"
    min_size               = 2
    max_size               = 10
    desired_capacity       = 2
    recurrence             = "0 17 * * *" # This means "5 p.m. every day"
}
```
x??

---

#### Accessing Module Outputs for Resource Parameters
Modules can output values that can be used in other configurations or modules. In this scenario, you need to access the name of an Auto Scaling Group (ASG) defined within a module.
:p How do you use module outputs to set parameters in AWS Auto Scaling resources?
??x
To use the ASG name from the `webserver-cluster` module, you first define it as an output variable in `/modules/services/webserver-cluster/outputs.tf`. Then, in your main configuration file (`prod/services/webserver-cluster/main.tf`), you can reference this output using the appropriate syntax.
```hcl
output "asg_name" {
    value       = aws_autoscaling_group.example.name
    description = "The name of the Auto Scaling Group"
}
```
In `main.tf`, you use the following syntax to set the `autoscaling_group_name` parameter:
```hcl
resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
    scheduled_action_name       = "scale-out-during-business-hours"
    min_size                    = 2
    max_size                    = 10
    desired_capacity            = 10
    recurrence                  = "0 9 * * *" # Means "9 a.m. every day"
    autoscaling_group_name      = module.webserver_cluster.asg_name
}
resource "aws_autoscaling_schedule" "scale_in_at_night" {
    scheduled_action_name       = "scale-in-at-night"
    min_size                    = 2
    max_size                    = 10
    desired_capacity            = 2
    recurrence                  = "0 17 * * *" # Means "5 p.m. every day"
    autoscaling_group_name      = module.webserver_cluster.asg_name
}
```
x??

---


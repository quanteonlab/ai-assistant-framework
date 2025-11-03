# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 16)

**Starting Chapter:** Terraform Gotchas. count and for_each Have Limitations

---

#### Count and For_each Limitations

Background context explaining that `count` and `for_each` parameters allow for dynamic resource creation based on conditions. However, there are limitations when using these parameters with outputs of resources.

:p Can you use resource outputs within a `count` parameter in Terraform?
??x
No, you cannot reference any resource outputs directly in the `count` or `for_each` parameters because Terraform requires that it can compute these values during the plan phase. The count value needs to be known before any resources are created or modified.

For example, if you try to use a random integer output from a `random_integer` resource as the `count` parameter:
```hcl
resource "random_integer" "num_instances" {
  min = 1
  max = 3
}

resource "aws_instance" "example_3" {
  count         = random_integer.num_instances.result
  ami           = "ami-0fb653ca2d3203ac1"
  instance_type = "t2.micro"
}
```
Running `terraform plan` will result in an error because Terraform cannot predict how many instances will be created based on the output of a resource during planning.

To work around this, you can use the `-target` argument to apply only the resources that the count depends on first:
```sh
terraform apply -target=random_integer.num_instances -auto-approve
```

:x??

---

#### Zero-Downtime Deployment Limitations

Background context explaining that while Terraform can perform zero-downtime deployments, there are limitations in certain scenarios. For instance, if a resource failure occurs during the deployment, such as an app version not booting properly and failing to register with the ALB.

:p Can Terraform automatically roll back in case of deployment issues?
??x
Yes, Terraform can automatically roll back in case of deployment issues by waiting for the new Auto Scaling Group (ASG) instances to register with the Application Load Balancer (ALB). If the v2 ASG does not register within `wait_for_capacity_timeout` (default 10 minutes), Terraform will consider the deployment a failure, delete the v2 ASG, and exit with an error. The original ASG (v1) continues to run without interruption.

For example:
```hcl
resource "aws_autoscaling_group" "v2" {
  // ASG configuration

  min_elb_capacity = 3
}

resource "aws_load_balancer" "alb" {
  // ALB configuration
}
```
If v2 fails to register within the timeout period, Terraform will handle the rollback as described.

:x??

---

#### Valid Plans Can Fail

Background context explaining that even if a plan appears valid during `terraform plan`, it can still fail during `apply` due to various reasons such as resource constraints or external dependencies.

:p What can happen after a seemingly successful `terraform plan`?
??x
Even if a plan seems valid and applies without issues, there is still a risk of failure during the `terraform apply` phase. This can be due to several factors:
- Resource limits (e.g., quotas, permissions).
- External dependencies failing.
- Incorrect configuration assumptions not catching in planning.

For example, a plan might look good but fail if an unexpected external dependency like a missing resource or permission change occurs at runtime.

:x??

---

#### Refactoring Can Be Tricky

Background context explaining that refactoring Terraform configurations can be challenging due to interdependencies and the immutable nature of resources once created. This often requires careful planning and understanding of how changes will affect the existing infrastructure.

:p Why is refactoring Terraform configurations tricky?
??x
Refactoring Terraform configurations can be tricky because:
- Resources are immutable; once created, they cannot be changed in place.
- Changes to one resource may have unintended consequences on others due to interdependencies.
- You need to carefully manage state and ensure that changes do not break the existing infrastructure.

For example, refactoring an AWS Lambda function's code might require updating its configuration and dependencies. However, if you forget to update a related CloudWatch event rule, it could lead to unexpected behavior or errors.

:x??

---

#### Zero-Downtime Deployment Limitations with create_before_destroy

Background context: The `create_before_destroy` approach for zero-downtime deployment using an Auto Scaling Group (ASG) has limitations, especially when auto scaling policies are involved. This method might reset the ASG size back to its minimum after each deployment, which can disrupt the intended scale of your instances.

:p What is a key limitation of using `create_before_destroy` with auto scaling policies in zero-downtime deployments?
??x
This approach may cause the ASG to revert to its minimal size post-deployment, thereby reducing the number of running servers if those servers were scaled up via an auto scaling policy. For example, if you scale up from 2 to 10 instances at a specific time and then deploy changes, the replacement ASG might start with only 2 instances until the next scheduled increase.
??x
---

#### Workarounds for Zero-Downtime Deployment

Background context: To mitigate the limitations of `create_before_destroy`, you could adjust parameters like recurrence or desired capacity. However, these are not ideal solutions as they require complex workarounds involving custom scripts and configurations.

:p What are some potential workarounds to handle the limitations of zero-downtime deployment with `create_before_destroy`?
??x
Potential workarounds include tweaking the recurrence parameter on `aws_autoscaling_schedule` or setting the desired_capacity parameter dynamically. However, these methods require additional scripting and complexity, making them less ideal compared to native solutions.
??x
---

#### Introduction to AWS Instance Refresh

Background context: AWS provides a native solution for zero-downtime deployments through instance refresh. This feature allows you to update your Auto Scaling Group instances without downtime by replacing old instances with new ones.

:p How does the `instance_refresh` block work in an ASG?
??x
The `instance_refresh` block within an `aws_autoscaling_group` resource enables AWS to replace existing instances with newer versions of them. When you modify the launch configuration, AWS starts an instance refresh process that gradually replaces old instances with new ones while maintaining service availability.
??x
---

#### Implementing Instance Refresh in Terraform

Background context: To use instance refresh in your ASG via Terraform, you need to configure the `aws_autoscaling_group` resource appropriately. This involves setting up a rolling strategy and defining preferences for the minimum healthy percentage.

:p How do you configure an ASG with instance refresh using Terraform?
??x
To configure an ASG with instance refresh in Terraform, use the following block within your `aws_autoscaling_group` resource:
```hcl
resource "aws_autoscaling_group" "example" {
  name                 = var.cluster_name
  launch_configuration = aws_launch_configuration.example.name
  vpc_zone_identifier   = data.aws_subnets.default.ids
  target_group_arns     = [aws_lb_target_group.asg.arn]
  health_check_type     = "ELB"
  min_size              = var.min_size
  max_size              = var.max_size

  instance_refresh {
    strategy  = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }
}
```
This configuration sets up a rolling refresh strategy and ensures that at least half of the instances remain healthy during the update process.
??x
---

#### Example of Instance Refresh in Practice

Background context: After configuring instance refresh, changes to your launch configuration will be applied without downtime. The ASG will replace old instances with new ones over time.

:p What happens when you modify a parameter and run `terraform plan` after setting up instance refresh?
??x
When you change a parameter such as server_text and run `terraform plan`, Terraform will show the following diff:
```
Terraform will perform the following actions:

  # module.webserver_cluster.aws_autoscaling_group.ex will be updated in-place
  ~ resource "aws_autoscaling_group" "example" {
        id                        = "webservers-stage-terraform-20190516"
    ~ launch_configuration      = "terraform-20190516" -> (known after apply)
        ...
  }
  # module.webserver_cluster.aws_launch_configuration.ex must be replaced
  +/- resource "aws_launch_configuration" "example" {
         id                          = "terraform-20190516" -> (known after apply)
         image_id                    = "ami-0fb653ca2d3203ac1"
         instance_type               = "t2.micro"
    ~ name                        = "terraform-20190516" -> (known after apply)
         ...
  }
```
Running `terraform apply` will quickly replace the old instances with new ones, ensuring zero downtime.
??x
---

---
#### Prefer Native Deployment Options
Background context explaining why native deployment options should be preferred, including examples of ECS and Kubernetes resources that support zero-downtime deployments through specific parameters or settings. 
:p When deploying a service using Terraform, why might you prefer to use first-class, native deployment options like instance refresh whenever possible?
??x
It is recommended to use native deployment options because they are designed specifically for the resource types being managed and can often provide features such as zero-downtime updates, which are difficult or impossible to achieve with custom scripts or manual processes. For example, in ECS, you can specify `deployment_maximum_percent` and `deployment_minimum_healthy_percent` to ensure smooth service transitions without downtime. Similarly, Kubernetes supports rolling updates via the `strategy` parameter set to `RollingUpdate`.
```terraform
resource "aws_ecs_service" "example" {
  # Other configuration...
  deployment_maximum_percent = 200
  deployment_minimum_healthy_percent = 50
}
```
x??
---

---
#### Valid Plans Can Fail
Background context explaining how Terraform plans and applies actions based on its state file, which may not include resources created manually or through other means. The example provided shows a case where an IAM user resource was planned but failed during apply due to the user already existing.
:p Why might a plan generated by `terraform plan` look valid but fail when you run `terraform apply`?
??x
A plan generated by `terraform plan` may appear valid because it only considers resources in its state file. However, if any of those resources were created manually or via other means (such as AWS Console), they are not included in the state file and will cause errors during `apply`. For instance, when you try to create an IAM user with a name that already exists, Terraform cannot proceed due to the conflict.
```terraform
resource "aws_iam_user" "existing_user" {
  name = "yevgeniy.brikman"
}
```
To resolve this issue, use `terraform import` to sync your existing infrastructure with Terraform’s state file. 
x??
---


# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 15)

**Starting Chapter:** Conditionals with for_each and for Expressions

---

#### Ternary Syntax vs Concat and One Functions
Background context: The text discusses different ways to handle conditional logic in Terraform, specifically focusing on ternary syntax and a safer approach using `concat` and `one` functions. This is important for ensuring code robustness as conditions become more complex.

:p What are the limitations of using ternary syntax for conditional logic in Terraform?
??x
Using ternary syntax can lead to brittle code that becomes difficult to maintain, especially if you modify conditions or resource definitions. For instance, changing the `count` parameter in `aws_iam_user_policy_attachment` resources might require updating multiple places, leading to potential errors.
x??

---
#### Concat and One Functions for Conditional Logic
Background context: The text explains using `concat` and `one` functions to handle conditional logic more safely. This method ensures that no matter how the condition changes, the output remains consistent.

:p How do you use `concat` and `one` functions in Terraform to achieve conditional logic?
??x
You can use `concat` to combine multiple lists into one and then use `one` to return a single value if there is exactly one element. Here's how it works:

```hcl
output "neo_cloudwatch_policy_arn" {
  value = one(concat(
    aws_iam_user_policy_attachment.neo_cloudwatch_full_access[*].policy_arn,
    aws_iam_user_policy_attachment.neo_cloudwatch_read_only[*].policy_arn
  ))
}
```

- `concat` combines the two lists of policy ARNs.
- `one` ensures that only a single value is returned, handling cases where no elements or multiple elements are present.

This method guarantees that your output remains consistent regardless of how conditions change.
x??

---
#### Conditional Logic with for_each and for Expressions
Background context: The text explains using `for_each` combined with the `for` expression to implement more complex conditional logic in Terraform. This approach allows you to conditionally create resources based on dynamic collections.

:p How does combining `for_each` and the `for` expression enable more complex conditional logic?
??x
Combining `for_each` with a nested `for` expression enables you to apply conditions directly within loops, allowing for arbitrary logical decisions. For example:

```hcl
dynamic "tag" {
  for_each = { 
    for key, value in var.custom_tags: 
    key => upper(value) 
    if key == "Name"
  }
  content {
    key                  = tag.key
    value                = tag.value
    propagate_at_launch  = true
  }
}
```

- `for_each` iterates over `var.custom_tags`.
- The nested `for` expression processes each tag, converting the value to uppercase and filtering out any "Name" tags.
- If no conditions match or the collection is empty, zero copies of the resource block are created.

This approach provides a flexible way to implement conditional logic without cluttering your code with multiple explicit conditions.
x??

---

#### Using `if` String Directive to Conditionally Render Output
Background context explaining how the `if` string directive works within Terraform and its use for conditional rendering. The `if` directive allows evaluating a boolean condition and only rendering content if that condition is true.

:p How can you use the `if` string directive in Terraform to avoid adding an extra trailing comma when using loops?
??x
You can use the `if` string directive in combination with the loop to conditionally render commas. For example, consider the following output block:

```terraform
output "for_directive_index_if" {
    value = <<EOF
percent{ for i, name in var.names }
${name}
percent { if i < length(var.names ) - 1 }, 
percent{ endif }
percent{ endfor } 
EOF
}
```

In this example, the `if` directive checks whether the current index `i` is less than one less than the total number of items (`length(var.names ) - 1`). If it is true, a comma and space are rendered. Otherwise, nothing is rendered.

This approach ensures that no trailing commas are added to the output string.
x??

---

#### Using `if` with `else` Clause for Conditional Rendering
Background context explaining how the `else` clause can be used within Terraform's `if` directive to handle cases when a condition evaluates to false. This is useful for appending additional text, such as punctuation marks, at the end of your rendered output.

:p How can you use the `else` clause in an `if` string directive to ensure no trailing whitespace or extra characters are added during rendering?
??x
You can use the `else` clause within the `if` string directive to handle cases where a condition evaluates to false. For example, consider this output block that adds a period at the end of the rendered string:

```terraform
output "for_directive_index_if_else_strip" {
    value = <<EOF
percent{~ for i, name in var.names ~}
${name}
percent { if i < length(var.names ) - 1 }, 
percent{ else }. percent{ endif }
percent{~ endfor ~} 
EOF
}
```

In this example:
- The `if` directive checks whether the current index `i` is less than one less than the total number of items.
- If it is true, a comma and space are rendered.
- If it is false (meaning it's the last item), an additional period is appended.

This ensures that no trailing whitespace or extra characters are added to the output string.
x??

---

#### Using `strip` Markers with HEREDOC
Background context explaining how `strip` markers can be used within Terraform's HEREDOC to remove unnecessary whitespaces. This is particularly useful when using loops and conditionals to ensure clean, formatted outputs.

:p How do you use `strip` markers in a HEREDOC to avoid adding extra whitespace or trailing commas during rendering?
??x
You can use `strip` markers around the content of your HEREDOC to automatically remove any leading or trailing whitespaces. For example:

```terraform
output "for_directive_index_if_strip" {
    value = <<EOF
percent{~ for i, name in var.names ~}
${name}
percent { if i < length(var.names ) - 1 }, 
percent{ endif }
percent{~ endfor ~} 
EOF
}
```

In this example:
- The `strip` markers (~) around the content of the HEREDOC are used to automatically strip any leading or trailing whitespaces.
- This ensures that no extra whitespace or commas are added to the rendered output.

This approach helps maintain clean and well-formatted outputs, especially when dealing with multiple lines and nested loops.
x??

---

---

#### Zero-Downtime Deployment Concept
In a web server cluster, ensuring minimal or no downtime during updates is crucial. This involves updating the Amazon Machine Image (AMI) without affecting user access. The core idea is to use Terraform configuration to manage these changes smoothly.

:p How do you ensure zero-downtime deployment in a web server cluster using AMI updates?
??x
To ensure zero-downtime deployment, you need to make your Terraform configurations flexible and capable of updating the AMI without disrupting service. Specifically, this involves:

1. Defining the AMI as an input variable in `variables.tf`.
2. Modifying the `user-data.sh` script to accept a variable that controls the text returned by the web server.
3. Updating the launch configuration in `main.tf` to use these variables.

Here's how you can do it:

1. **Define Variables:**
   ```hcl
   variable "ami" {
     description = "The AMI to run in the cluster"
     type        = string
     default      = "ami-0fb653ca2d3203ac1"
   }

   variable "server_text"  {
     description = "The text the web server should return"
     type        = string
     default      = "Hello, World"
   }
   ```

2. **Update User Data Script:**
   ```hcl
   #./bin/bash
   cat > index.html <<EOF
   <h1>${server_text}</h1>
   <p>DB address: ${db_address}</p>
   <p>DB port: ${db_port}</p>
   EOF

   nohup busybox httpd -f -p ${server_port} &
   ```

3. **Update Launch Configuration:**
   ```hcl
   resource "aws_launch_configuration" "example"  {
     image_id         = var.ami
     instance_type    = var.instance_type
     security_groups  = [aws_security_group.instance.id]
     user_data        = templatefile("${path.module}/user-data.sh", {
       server_port  = var.server_port
       db_address   = data.terraform_remote_state.db.outputs.address
       db_port      = data.terraform_remote_state.db.outputs.port
       server_text  = var.server_text
     })
     # Required when using a launch configuration with an auto scaling group.
     lifecycle  {
       create_before_destroy  = true
     }
   }
   ```

4. **Apply Changes in Staging Environment:**
   ```hcl
   module "webserver_cluster"  {
     source      = "../../../../modules/services/webserver-cluster"
     ami         = "ami-0fb653ca2d3203ac1"
     server_text = "New server text"
     cluster_name            = "webservers-stage"
     db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
     db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
     instance_type       = "t2.micro"
     min_size            = 2
     max_size            = 2
     enable_autoscaling  = false
   }
   ```

By following these steps, you can update the AMI and server text without downtime.

x??

---

#### Launch Configuration Update in Terraform
When updating an existing launch configuration in a web server cluster, you need to ensure that the changes are applied seamlessly. The launch configuration is crucial for defining how instances should be launched within an auto-scaling group.

:p How do you update the launch configuration to use a new AMI and variable inputs in Terraform?
??x
To update the launch configuration using a new AMI and variable inputs, follow these steps:

1. **Define Variables:**
   ```hcl
   variable "ami" {
     description = "The AMI to run in the cluster"
     type        = string
     default      = "ami-0fb653ca2d3203ac1"
   }

   variable "server_text"  {
     description = "The text the web server should return"
     type        = string
     default      = "Hello, World"
   }
   ```

2. **Update User Data Script:**
   ```hcl
   #./bin/bash
   cat > index.html <<EOF
   <h1>${server_text}</h1>
   <p>DB address: ${db_address}</p>
   <p>DB port: ${db_port}</p>
   EOF

   nohup busybox httpd -f -p ${server_port} &
   ```

3. **Update Launch Configuration in `main.tf`:**
   ```hcl
   resource "aws_launch_configuration" "example"  {
     image_id         = var.ami
     instance_type    = var.instance_type
     security_groups  = [aws_security_group.instance.id]
     user_data        = templatefile("${path.module}/user-data.sh", {
       server_port  = var.server_port
       db_address   = data.terraform_remote_state.db.outputs.address
       db_port      = data.terraform_remote_state.db.outputs.port
       server_text  = var.server_text
     })
     # Required when using a launch configuration with an auto-scaling group.
     lifecycle  {
       create_before_destroy  = true
     }
   }
   ```

4. **Apply Changes in Staging Environment:**
   ```hcl
   module "webserver_cluster"  {
     source      = "../../../../modules/services/webserver-cluster"
     ami         = "ami-0fb653ca2d3203ac1"
     server_text = "New server text"
     cluster_name            = "webservers-stage"
     db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
     db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
     instance_type       = "t2.micro"
     min_size            = 2
     max_size            = 2
     enable_autoscaling  = false
   }
   ```

By updating the `ami` and `server_text` variables, you can ensure that new instances are launched with the updated AMI and server text. Terraform will handle the transition seamlessly.

x??

---

#### Zero-Downtime Deployment for AWS Auto Scaling Groups
In order to perform an update on your AWS Auto Scaling Group (ASG) without downtime, you need to ensure that a new ASG is created first and only then destroyed after confirming its stability. This can be achieved using Terraform's `create_before_destroy` lifecycle setting.

Background context: When updating the launch configuration of an ASG, simply referencing the updated launch config won't immediately affect the running instances. The ASG needs to deploy new instances before the changes take effect, which might lead to downtime if done incorrectly.

:p How can you ensure a zero-downtime deployment for your AWS Auto Scaling Group?
??x
To achieve zero-downtime deployment, you need to set up the `create_before_destroy` lifecycle setting in Terraform. This ensures that when Terraform tries to replace an ASG, it first creates a new one before destroying the old one.

```hcl
resource "aws_autoscaling_group" "example" {
  name = "${var.cluster_name}-${aws_launch_configuration.example.name}"
  
  # Other parameters...

  lifecycle {
    create_before_destroy = true
  }
}
```

x??

---
#### Dependent ASG Name on Launch Configuration
To ensure that the ASG's name changes every time the launch configuration is updated, making it necessary for Terraform to replace the ASG.

Background context: By directly depending on the name of the launch configuration in the ASG’s name parameter, you can force a replacement whenever there are updates to the launch config. This ensures that the ASG and its associated instances stay synchronized with any changes made to the underlying infrastructure.

:p How do you ensure the ASG's name changes when the launch configuration is updated?
??x
By explicitly depending on the name of the launch configuration in the ASG’s `name` parameter, you can force a replacement every time there are updates to the launch config. This ensures that the ASG and its associated instances stay synchronized with any changes made to the underlying infrastructure.

```hcl
resource "aws_autoscaling_group" "example" {
  name = "${var.cluster_name}-${aws_launch_configuration.example.name}"
  
  # Other parameters...
}
```

x??

---
#### Min Elb Capacity Parameter for Zero-Downtime ASG
The `min_elb_capacity` parameter in the ASG ensures that at least a certain number of instances from the new ASG pass health checks before the old ASG is destroyed, thus maintaining service availability.

Background context: Setting the `min_elb_capacity` to the minimum size (`var.min_size`) of the cluster helps ensure that there are enough healthy instances in the load balancer to maintain service continuity during the transition period.

:p How does the `min_elb_capacity` parameter help achieve zero-downtime deployment?
??x
The `min_elb_capacity` parameter ensures that at least a certain number of instances from the new ASG pass health checks before the old ASG is destroyed. By setting it to the minimum size (`var.min_size`) of the cluster, you ensure that there are enough healthy instances in the load balancer to maintain service continuity during the transition period.

```hcl
resource "aws_autoscaling_group" "example" {
  min_elb_capacity = var.min_size
  
  # Other parameters...
}
```

x??

---

---
#### Initial State of ASG
Background context: The deployment process starts with an existing Auto Scaling Group (ASG) running version 1 of your application. This state is a prerequisite for understanding how zero-downtime deployments work.

:p What is the initial state of the Auto Scaling Group before deploying a new version?
??x
The initial state has an ASG running v1 of the code.
x??

---
#### Apply Command and New ASG Deployment
Background context: When you run the `apply` command, Terraform initiates the deployment of a new ASG with updated configurations (e.g., using a different AMI). This process ensures that no single point in time is down during the transition.

:p What happens when you run the `apply` command?
??x
Running the `apply` command triggers Terraform to start deploying a new ASG with the updated launch configuration, such as a newer version of your code. During this phase, both old and new versions coexist.
x??

---
#### Simultaneous Execution of v1 and v2
Background context: After the new ASG starts, it takes some time for instances to fully boot up and connect to services like the database and load balancer (ALB). During this period, both version 1 and version 2 of your application run concurrently.

:p What happens during the initial boot-up phase of the new ASG?
??x
During the initial boot-up, the new ASG's servers start booting, connecting to the database, registering in the ALB, and passing health checks. At this point, both v1 and v2 versions are running simultaneously.
x??

---
#### Load Balancer Routing
Background context: Once the new ASG has enough healthy instances (determined by `min_elb_capacity`), traffic starts being routed to the new version of your application.

:p How does the load balancer route traffic between old and new versions?
??x
The ALB routes traffic based on which servers are available. Initially, both v1 and v2 versions might be hit alternately. Once enough instances of v2 have registered, it will start handling more requests.
x??

---
#### Phased Undeployment of Old ASG
Background context: After a successful transition, the old ASG starts deregistering from the ALB, followed by shutting down until only the new version remains.

:p What happens when `min_elb_capacity` instances are registered in the new ASG?
??x
Once `min_elb_capacity` instances of the v2 ASG have registered and passed health checks, Terraform begins undeploying the old ASG. This involves deregistering servers from the ALB and then shutting them down.
x??

---
#### Zero-Downtime Deployment Demonstration
Background context: To verify the zero-downtime deployment, you can make changes to parameters like `server_text` and observe traffic alternation using a continuous curl command.

:p How can you demonstrate the zero-downtime process?
??x
You can update a parameter (e.g., `server_text`) and run `apply`. Using a Bash one-liner, continuously hit the ALB URL with `curl`, observing how requests alternate between old and new versions.
x??

---
#### Real-Time Observation of Deployment
Background context: The curl command output helps visualize the transition from old to new version as instances register and deregister.

:p What will be the output if you run the provided curl loop?
??x
The output will show alternating responses, starting with `New server text`, then switching between it and `foo bar` as new v2 instances come online. Eventually, only `foo bar` remains.
```
while true; do curl http://<load_balancer_url>; sleep 1; done
```

Example Output:
```bash
New server text
New server text
foo bar
New server text
foo bar
foo bar
foo bar
...
```
x??

---


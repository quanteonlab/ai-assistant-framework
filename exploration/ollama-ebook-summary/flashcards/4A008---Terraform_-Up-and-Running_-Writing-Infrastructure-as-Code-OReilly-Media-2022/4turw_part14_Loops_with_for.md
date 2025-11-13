# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 14)

**Starting Chapter:** Loops with for Expressions

---

#### For Expressions in Terraform
Background context: In Chapter 5, you'll learn about loops and conditionals to manage resources more efficiently. One of the advanced features is the `for` expression, which allows for transforming lists and maps into output values or resource instances.

The basic syntax of a `for` expression is as follows:
```terraform
[for <ITEM> in <LIST> : <OUTPUT> ]
```
- **LIST**: A list to loop over.
- **ITEM**: The local variable name assigned to each item in the LIST.
- **OUTPUT**: An expression that transforms the ITEM.

For example, you can transform a list of names into uppercase:
```terraform
output "upper_names" {
  value = [for name in var.names : upper(name)]
}
```

:p How does Terraform's `for` expression work for transforming lists?
??x
The `for` expression allows you to loop over each item in the provided list and apply a transformation, such as converting all names to uppercase. For example:
```terraform
output "upper_names" {
  value = [for name in var.names : upper(name)]
}
```
When this Terraform code is applied, it will output an array of the transformed items.

If you need to filter or add conditions based on each item, you can do so within the `for` expression:
```terraform
output "short_upper_names" {
  value = [for name in var.names : upper(name) if length(name) < 5]
}
```
This will only include names with a length less than 5 and convert them to uppercase.

x??

---
#### For Expressions with Maps
Background context: In addition to lists, Terraform's `for` expression can also be used on maps. This allows you to loop over the key-value pairs in a map and apply transformations or conditions based on each pair.

The syntax for looping over a map is:
```terraform
[for <KEY>, <VALUE> in <MAP> : <OUTPUT> ]
```
- **MAP**: A map to loop over.
- **KEY**: The local variable name assigned to the key of each key-value pair in the MAP.
- **VALUE**: The local variable name assigned to the value of each key-value pair in the MAP.
- **OUTPUT**: An expression that transforms KEY and VALUE.

For example, you can define a map for heroes with their roles:
```terraform
variable "hero_thousand_faces" {
  description = "map"
  type        = map(string)
  default     = {
    neo      = "hero"
    trinity  = "love interest"
    morpheus = "mentor"
  }
}
```

:p How do you use a `for` expression to loop over a map in Terraform?
??x
To use a `for` expression with a map, you specify the keys and values using `<KEY>, <VALUE>`, where `<KEY>` is assigned to each key and `<VALUE>` is assigned to each value. You can then perform operations on both or apply conditions.

Example:
```terraform
variable "hero_thousand_faces" {
  description = "map"
  type        = map(string)
  default     = {
    neo      = "hero"
    trinity  = "love interest"
    morpheus = "mentor"
  }
}
output "roles_of_heroes" {
  value = [for key, value in var.hero_thousand_faces : { role = key, description = value }]
}
```

This will output an array of maps where each map contains the key (hero name) and the corresponding value (role).

x??

---
#### For Expressions - Filtering and Transforming
Background context: Terraform's `for` expression allows for both transforming elements in a list or map and filtering them based on specific conditions. This is particularly useful when you need to process complex data structures.

For example, combining transformations with conditions:
```terraform
output "short_upper_names" {
  value = [for name in var.names : upper(name) if length(name) < 5]
}
```

:p Can you provide an example of using `for` expression with filtering?
??x
Yes, you can use the `for` expression to both transform and filter elements. Here’s an example:

```terraform
output "short_upper_names" {
  value = [for name in var.names : upper(name) if length(name) < 5]
}
```

In this example:
- The `names` variable is a list of names.
- The `upper(name)` function converts each name to uppercase.
- The `if length(name) < 5` condition ensures only names with fewer than 5 characters are included in the final output.

This will result in an array containing only the uppercase versions of names that have less than five characters.

x??

---
#### List Comprehensions in Python
Background context: Python offers a concise and readable way to create lists using list comprehensions. This can be seen as an analogy for how Terraform's `for` expressions work, but with some differences. In Python, the syntax is:

```python
upper_case_names = [name.upper() for name in names]
```

:p How does Python use list comprehensions to transform a list?
??x
In Python, you can create lists of transformed elements using list comprehensions. For example:
```python
names = ["neo", "trinity", "morpheus"]
upper_case_names = [name.upper() for name in names]
```

This code will output `['NEO', 'TRINITY', 'MORPHEUS']`. You can also add conditions to filter the resulting list:
```python
short_upper_case_names = [name.upper() for name in names if len(name) < 5]
```
This will result in `['NEO']`.

x??

---

#### For Expressions for Lists

Background context: The provided text explains how to use `for` expressions in Terraform to generate a list of strings. This is useful for creating dynamic outputs based on input variables.

:p What does the `for` expression used with lists do?
??x
The `for` expression with lists allows you to create a list by iterating over an input collection (like a variable containing a list) and generating new elements in the output list.

Example:
```terraform
output "bios" {
  value = [for name, role in var.hero_thousand_faces : "${name} is the${role}"]
}
```
This will loop through each element in `var.hero_thousand_faces` (a map where keys are names and values are roles) and generate a string for each pair.

??x
The answer with detailed explanations.
```terraform
output "bios" {
  value = [for name, role in var.hero_thousand_faces : "${name} is the${role}"]
}
```
In this example, `var.hero_thousand_faces` is a map where keys are names and values are roles. The `for` expression iterates over each key-value pair (name, role), and for each iteration, it generates a string in the format `${name} is the${role}`.

The resulting list will be:
```plaintext
[
  "morpheus is the mentor",
  "neo is the hero",
  "trinity is the love interest"
]
```
??x

---

#### For Expressions for Maps

Background context: The text also explains how to use `for` expressions with maps to output a map of key-value pairs, where each pair consists of the transformed key and value.

:p How can you transform keys and values in a map using `for` expressions?
??x
You can transform both keys and values in a map by using a `for` expression. This is done by specifying an output key and value for each iteration of the collection (map).

Example:
```terraform
output "upper_roles" {
  value = {for name, role in var.hero_thousand_faces : upper(name) => upper(role)}
}
```
This will loop through each key-value pair in `var.hero_thousand_faces`, transform both keys and values to uppercase using the `upper` function, and create a new map with these transformed key-value pairs.

??x
The answer with detailed explanations.
```terraform
output "upper_roles" {
  value = {for name, role in var.hero_thousand_faces : upper(name) => upper(role)}
}
```
In this example, the `for` expression iterates over each key-value pair (name, role) in `var.hero_thousand_faces`. For each iteration, it transforms both the key (`name`) and value (`role`) to uppercase using the `upper` function.

The resulting map will be:
```plaintext
{
  "MORPHEUS" = "MENTOR"
  "NEO" = "HERO"
  "TRINITY" = "LOVE INTEREST"
}
```
??x

---

#### For String Directive without Index

Background context: The text describes using the `for` string directive to loop over a collection and output strings in a specific format. This is particularly useful for generating dynamic strings based on input variables.

:p How does the `for` string directive work with collections?
??x
The `for` string directive allows you to loop over a collection (like a list) and generate strings dynamically within a Terraform output or variable.

Example:
```terraform
variable "names" {
  description = "Names to render"
  type        = list(string)
  default     = ["neo", "trinity", "morpheus"]
}

output "for_directive" {
  value = "%{ for name in var.names }${name}, %{ endfor }"
}
```
This will loop over each element in `var.names` and generate a string with commas separating the names.

??x
The answer with detailed explanations.
```terraform
variable "names" {
  description = "Names to render"
  type        = list(string)
  default     = ["neo", "trinity", "morpheus"]
}

output "for_directive" {
  value = "%{ for name in var.names }${name}, %{ endfor }"
}
```
In this example, the `for` string directive is used to loop over each element (`name`) in `var.names`. For each iteration, it appends `${name}` followed by a comma.

The resulting output will be:
```plaintext
"neo, trinity, morpheus,"
```
Notice that there's an extra trailing comma. This can be fixed using conditionals as described later.

??x

---

#### For String Directive with Index

Background context: The text explains how to use the `for` string directive with an index for more detailed control over the output strings. This is useful when you need to include additional information like indices or different formatting.

:p How does the `for` string directive work with an index?
??x
The `for` string directive can also accept an index along with each item in a collection, allowing for more detailed control over the generated output strings.

Example:
```terraform
output "for_directive_index" {
  value = "%{ for i, name in var.names }(${i})${name}, %{ endfor }"
}
```
This will loop over each element (`name`) and its index (`i`), appending the index in parentheses followed by a space and the name.

??x
The answer with detailed explanations.
```terraform
output "for_directive_index" {
  value = "%{ for i, name in var.names }(${i})${name}, %{ endfor }"
}
```
In this example, the `for` string directive is used to loop over each element (`name`) and its index (`i`). For each iteration, it appends `(${i})${name}` followed by a comma.

The resulting output will be:
```plaintext
"(0) neo, (1) trinity, (2) morpheus,"
```
Notice the extra trailing comma. This can be fixed using conditionals as described later.

??x

---

#### Conditionals Using count Parameter
Background context: In Terraform, while direct `if` statements aren't supported, you can use the `count` parameter to conditionally create resources. The `count` parameter allows you to specify how many copies of a resource should be created based on an integer value.

:p How can you conditionally enable auto-scaling using the `count` parameter in Terraform?
??x
By setting `count` to 1, one instance of the resource is created; by setting it to 0, no instances are created. In this case, we use a Boolean input variable to control whether auto-scaling should be enabled.

For example:
```hcl
variable "enable_autoscaling" {
  description = "If set to true, enable auto scaling"
  type        = bool
}

resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
  count                  = var.enable_autoscaling ? 1 : 0

  scheduled_action_name  = "${var.cluster_name}-scale-out-during-business-hours"
  min_size                = 2
  max_size                = 10
  desired_capacity        = 10
  recurrence              = "0 9 * * *"
  autoscaling_group_name  = aws_autoscaling_group.example.name
}

resource "aws_autoscaling_schedule" "scale_in_at_night" {
  count                  = var.enable_autoscaling ? 1 : 0

  scheduled_action_name  = "${var.cluster_name}-scale-in-at-night"
  min_size                = 2
  max_size                = 10
  desired_capacity        = 2
  recurrence              = "0 17 * * *"
  autoscaling_group_name  = aws_autoscaling_group.example.name
}
```
x??

---
#### If-Statements with count Parameter (Pseudo Code)
Background context: While Terraform doesn't support true `if` statements, you can achieve similar functionality using the `count` parameter. The idea is to conditionally create resources based on certain conditions.

:p Can you demonstrate how to use an `if` statement equivalent through `count` in pseudo code?
??x
In Terraform's pseudo-code representation:

```hcl
# This is just pseudo code.
160 | Chapter 5: Terraform Tips and Tricks: Loops, If-Statements, Deployment, and Gotchas

if var.enable_autoscaling {
  resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
    scheduled_action_name = "${var.cluster_name}-scale-out-during-business-hours"
    min_size                = 2
    max_size                = 10
    desired_capacity        = 10
    recurrence              = "0 9 * * *"
    autoscaling_group_name  = aws_autoscaling_group.example.name
  }

  resource "aws_autoscaling_schedule" "scale_in_at_night" {
    scheduled_action_name   = "${var.cluster_name}-scale-in-at-night"
    min_size                = 2
    max_size                = 10
    desired_capacity        = 2
    recurrence              = "0 17 * * *"
    autoscaling_group_name  = aws_autoscaling_group.example.name
  }
}
```

However, Terraform doesn't support the above code directly. Instead, you use `count` to conditionally create resources.
x??

---
#### count Parameter Example in Action
Background context: The `count` parameter can be used not only for basic loops but also as a conditional mechanism. By setting `count` to 1 or 0 based on a variable value, Terraform decides whether to create the resource.

:p How does using `count = var.enable_autoscaling ? 1 : 0` in a Terraform configuration work?
??x
By using the ternary operator within the `count` parameter, you conditionally create resources. If `var.enable_autoscaling` is true (or non-zero), then `count` will be set to 1 and the resource will be created; otherwise, it will be set to 0 and no instance of that resource will be created.

Example:
```hcl
resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
  count                  = var.enable_autoscaling ? 1 : 0

  scheduled_action_name  = "${var.cluster_name}-scale-out-during-business-hours"
  min_size                = 2
  max_size                = 10
  desired_capacity        = 10
  recurrence              = "0 9 * * *"
  autoscaling_group_name  = aws_autoscaling_group.example.name
}
```
This ensures that the `aws_autoscaling_schedule` resource is only created when auto-scaling is enabled.
x??

---

#### Conditional Logic with `count` Parameter
Conditional logic can be implemented using the ternary syntax in Terraform. Specifically, you can control whether resources are created or not based on a boolean condition. The `count` parameter is used to decide how many instances of a resource should be created.

In this case, the `enable_autoscaling` variable determines whether auto-scaling schedules (`aws_autoscaling_schedule`) will be configured for your web server cluster.

:p How does Terraform use conditional logic in the `count` parameter for resources like `aws_autoscaling_schedule`?
??x
Terraform uses a ternary expression to conditionally set the `count` value. If `var.enable_autoscaling` is true, the count will be 1, creating one instance of the resource (in this case, an autoscaling schedule). If it's false, the count will be 0, meaning no instances of that resource will be created.

Here’s how you can implement this in Terraform:

```hcl
resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
    count = var.enable_autoscaling ? 1 : 0
    scheduled_action_name   = "${var.cluster_name}-scale-out-during-business-hours"
    min_size                = 2
    max_size                = 10
    desired_capacity        = 10
    recurrence              = "0 9 * * *"
    autoscaling_group_name  = aws_autoscaling_group.example.name
}

resource "aws_autoscaling_schedule" "scale_in_at_night" {
    count = var.enable_autoscaling ? 1 : 0
    scheduled_action_name   = "${var.cluster_name}-scale-in-at-night"
    min_size                = 2
    max_size                = 10
    desired_capacity        = 2
    recurrence              = "0 17 * * *"
    autoscaling_group_name  = aws_autoscaling_group.example.name
}
```

The `count` parameter effectively enables or disables the creation of these resources based on the value of `var.enable_autoscaling`.
x??

---

#### Conditional Logic with `count` Parameter for Different Environments
In Terraform, you can use conditional logic to enable or disable specific configurations in different environments by setting environment-specific variables.

:p How does one conditionally configure auto-scaling in Terraform based on the environment?
??x
You define a variable `enable_autoscaling`, and set it to true for production and false for staging. This way, the resources will be created only if `var.enable_autoscaling` is true, allowing you to enable or disable auto-scaling schedules depending on the environment.

Here’s an example of how this can be done:

For Staging (in live/stage/services/webserver-cluster/main.tf):
```hcl
module "webserver_cluster" {
    source  = "../../../../modules/services/webserver-cluster"
    cluster_name            = "webservers-stage"
    db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
    db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
    instance_type           = "t2.micro"
    min_size                = 2
    max_size                = 2
    enable_autoscaling      = false
}
```

For Production (in live/prod/services/webserver-cluster/main.tf):
```hcl
module "webserver_cluster" {
    source  = "../../../../modules/services/webserver-cluster"
    cluster_name            = "webservers-prod"
    db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
    db_remote_state_key     = "prod/data-stores/mysql/terraform.tfstate"
    instance_type           = "m4.large"
    min_size                = 2
    max_size                = 10
    enable_autoscaling      = true
    custom_tags             = {
        Owner      = "team-foo"
        ManagedBy  = "terraform"
    }
}
```

In the above example, setting `enable_autoscaling` to false in staging means that no auto-scaling schedules will be created for this environment.
x??

---

#### Conditional Logic with Ternary Syntax
Terraform supports conditional logic using a ternary syntax. The format is `<CONDITION> ? <TRUE_VAL> : <FALSE_VAL>`.

:p How does the ternary operator work in Terraform?
??x
The ternary operator in Terraform allows you to conditionally set values based on a boolean condition. If the `condition` evaluates to true, the result will be `true_val`; otherwise, it will return `false_val`.

Example:
```hcl
count = var.enable_autoscaling ? 1 : 0
```

In this example, if `var.enable_autoscaling` is set to true, the count value will be 1. If it's false, the count will be 0.

This syntax can be used in various parts of Terraform configurations, such as resource counts or attribute values.
x??

---

#### Conditional Logic for Resource Creation
Conditional logic using the `count` parameter in Terraform helps manage which resources are created based on a condition. The ternary operator is particularly useful here to dynamically control resource creation.

:p How does one use the ternary operator with `count` in Terraform?
??x
You can use the ternary operator within the `count` attribute of Terraform resources to conditionally create them. For example, if you want to create an autoscaling schedule only when auto-scaling is enabled:

```hcl
resource "aws_autoscaling_schedule" "scale_out_during_business_hours" {
    count = var.enable_autoscaling ? 1 : 0
    // Other resource attributes
}
```

If `var.enable_autoscaling` is true, one instance of the `aws_autoscaling_schedule` will be created. If it's false, no instances will be created.

This approach allows you to dynamically manage resources in your Terraform configurations based on conditionally set variables.
x??

---

#### Using Count Parameter for Conditional Resource Creation
Background context: In Terraform, you can use the `count` parameter to conditionally create resources based on a boolean variable. This is useful when you need to decide between creating one of several similar resources depending on some input.

:p How can you use the count parameter to conditionally attach different IAM policies to an IAM user?
??x
You can use the `count` parameter in Terraform to conditionally create resources based on a boolean variable. If `var.give_neo_cloudwatch_full_access` is true, it creates an attachment for full access; otherwise, it attaches read-only permissions.

```terraform
resource "aws_iam_user_policy_attachment" "neo_cloudwatch_full_access" {
  count = var.give_neo_cloudwatch_full_access ? 1 : 0
  user       = aws_iam_user.example[0].name
  policy_arn = aws_iam_policy.cloudwatch_full_access.arn
}

resource "aws_iam_user_policy_attachment" "neo_cloudwatch_read_only" {
  count = var.give_neo_cloudwatch_full_access ? 0 : 1
  user       = aws_iam_user.example[0].name
  policy_arn = aws_iam_policy.cloudwatch_read_only.arn
}
```

x??

---
#### Conditional Output Based on Resource Creation
Background context: After conditionally creating resources using the `count` parameter, you might want to output an attribute of the resource that was actually created.

:p How can you ensure that an output variable correctly reflects the policy attached based on a conditional resource creation?
??x
To handle this, you use the `try` function in Terraform. The `try` function returns the value if the resource is attached and throws an error if it isn't, allowing you to conditionally set your output.

```terraform
output "neo_cloudwatch_policy_arn" {
  value = try(element(split("\n", aws_iam_user_policy_attachment.neo_cloudwatch_full_access.*.policy_arn),0), aws_iam_user_policy_attachment.neo_cloudwatch_read_only[0].policy_arn)
}
```

This `try` function checks if the full access policy is attached; if not, it falls back to the read-only policy.

x??

---
#### Using Try Function for Conditional Outputs
Background context: The `try` function in Terraform is used to conditionally return a value based on whether a resource exists or not. This can be useful when you have multiple resources that might get created under different conditions and need an output depending on which one was actually attached.

:p How does the `try` function work for handling conditional outputs?
??x
The `try` function in Terraform allows you to return a value if it exists; otherwise, it returns another default value. Here’s how it works:

```terraform
output "neo_cloudwatch_policy_arn" {
  value = try(element(split("\n", aws_iam_user_policy_attachment.neo_cloudwatch_full_access.*.policy_arn),0), aws_iam_user_policy_attachment.neo_cloudwatch_read_only[0].policy_arn)
}
```

In this example, `try` checks if the full access policy attachment exists by splitting its list and getting the first element. If it doesn’t exist (i.e., no full access was attached), it falls back to the read-only policy.

x??

---
#### Conditional Resource Creation Logic
Background context: In Terraform, you can conditionally create resources using the `count` parameter based on a boolean input variable. This example demonstrates how to attach either a full-access or read-only IAM policy to an IAM user.

:p How does the conditional logic work in Terraform for creating different types of resource attachments?
??x
The conditional logic works by using the `count` parameter, which evaluates to 1 if the condition is true and 0 otherwise. Here’s how it’s implemented:

```terraform
resource "aws_iam_user_policy_attachment" "neo_cloudwatch_full_access" {
  count = var.give_neo_cloudwatch_full_access ? 1 : 0
  user       = aws_iam_user.example[0].name
  policy_arn = aws_iam_policy.cloudwatch_full_access.arn
}

resource "aws_iam_user_policy_attachment" "neo_cloudwatch_read_only" {
  count = var.give_neo_cloudwatch_full_access ? 0 : 1
  user       = aws_iam_user.example[0].name
  policy_arn = aws_iam_policy.cloudwatch_read_only.arn
}
```

In this example, if `var.give_neo_cloudwatch_full_access` is true, the full access policy will be attached (count=1), otherwise, only read-only access will be given (count=0).

x??

---


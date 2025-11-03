# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 12)

**Starting Chapter:** Module Gotchas. Inline Blocks

---

#### File Paths and Path References
Background context explaining how file paths can be a challenge when using Terraform, especially with modules. Discuss why relative paths are necessary and introduce path references.

:p What is the issue with file paths when using the `templatefile` function in a module?
??x
The issue arises because the `templatefile` function requires a relative path to the template file, but this path can vary depending on where Terraform code runs. By default, it uses the current working directory (path.cwd), which might not be suitable for modules defined in separate folders.

To solve this, you should use path references such as `path.module`, which returns the filesystem path of the module where the expression is defined. This ensures that your file paths are consistent and relative to the correct location within the module.

Example:
```hcl
user_data  = templatefile("${path.module}/user-data.sh", {
    server_port  = var.server_port
    db_address   = data.terraform_remote_state.db.outputs.address
    db_port      = data.terraform_remote_state.db.outputs.port
})
```
x??

---

#### Inline Blocks vs Separate Resources in Terraform Modules
Background context explaining the difference between using inline blocks and separate resources for configurations within a module, and the potential issues when mixing both.

:p What is the main reason to prefer using separate resources over inline blocks in Terraform modules?
??x
Using separate resources provides more flexibility and configurability. Inline blocks are tied to specific resource definitions and cannot be added outside of their parent module. Separate resources can be added anywhere, making them easier to customize and extend by users or other modules.

For example, consider the security group configuration:
Inline block approach (not recommended):
```hcl
resource "aws_security_group" "alb" {
   name = "${var.cluster_name}-alb"
   ingress { ... }
   egress { ... }
}
```
Separate resource approach (recommended):
```hcl
resource "aws_security_group" "alb" {
   name = "${var.cluster_name}-alb"
}

resource "aws_security_group_rule" "allow_http_inbound" {
   type              = "ingress"
   security_group_id = aws_security_group.alb.id
   from_port    = local.http_port
   to_port      = local.http_port
   protocol     = local.tcp_protocol
   cidr_blocks  = local.all_ips
}

resource "aws_security_group_rule" "allow_all_outbound" {
   type              = "egress"
   security_group_id = aws_security_group.alb.id
   from_port    = local.any_port
   to_port      = local.any_port
   protocol     = local.any_protocol
   cidr_blocks  = local.all_ips
}
```
x??

---

#### Exporting Outputs in Terraform Modules
Background context on why it's useful to export specific outputs, such as the DNS name of an ALB, so users can easily reference them outside the module.

:p How do you expose the DNS name of the ALB as an output variable in a Terraform module?
??x
To expose the DNS name of the ALB as an output variable, add an `output` block in your module's outputs file. This allows users to access this information easily when using the module.

Example:
```hcl
output "alb_dns_name" {
   value       = aws_lb.example.dns_name
   description = "The domain name of the load balancer"
}
```

You can then pass through this output in other modules that use it, like so:

```hcl
output "alb_dns_name" {
   value       = module.webserver_cluster.alb_dns_name
   description = "The domain name of the load balancer"
}
```
x??

---

#### Security Group IDs as Outputs
Background context on why exporting the ID of a security group attached to an ALB can be useful for extending or modifying configurations in other parts of the infrastructure.

:p How do you export the ID of the AWS security group as an output variable in a Terraform module?
??x
To export the ID of the AWS security group, add an `output` block in your module's outputs file. This allows users to reference this ID when adding additional rules or configuring other resources that need access control.

Example:
```hcl
output "alb_security_group_id" {
   value       = aws_security_group.alb.id
   description = "The ID of the Security Group attached to the load balancer"
}
```

Now, if you need to add an extra ingress rule for testing in a specific environment (like staging), you can do this:

```hcl
resource "aws_security_group_rule" "allow_testing_inbound" {
   type              = "ingress"
   security_group_id  = module.webserver_cluster.alb_security_group_id
   from_port    = 12345
   to_port      = 12345
   protocol     = "tcp"
   cidr_blocks  = ["0.0.0.0/0"]
}
```

This ensures that the code works correctly, as it references the ID of the security group consistently.
x??

---

#### Network Isolation
Background context: The provided text discusses how the network environments created using Terraform are not isolated at a network level, which can pose risks. Resources from one environment (e.g., staging) can communicate with another environment (e.g., production), leading to potential issues like configuration mistakes affecting both or security breaches compromising multiple environments.
:p What is a significant risk when running both staging and production environments in the same VPC?
??x
There are several risks, but a key one is that any mistake in the configuration of resources in the staging environment could affect the production environment. Additionally, if an attacker gains access to the staging environment, they can also gain access to the production environment due to their interconnectedness.
x??

---

#### Module Versioning
Background context: The text explains why it’s important to version modules when working with Terraform, especially for separate environments like staging and production. This helps in making changes in one environment without affecting another by using different versions of the same module.
:p Why is module versioning critical when managing multiple environments?
??x
Module versioning ensures that changes made in a staging environment do not inadvertently affect the production environment until they are thoroughly tested. By maintaining separate versions, developers can test new configurations or features in a controlled environment before deploying them to production.
x??

---

#### Separating Module and Live Repositories
Background context: The text suggests storing reusable modules in one Git repository and the configuration for live environments in another. This separation allows for better management of infrastructure changes without affecting live deployments directly.
:p How should you structure your Terraform repositories according to the best practices mentioned?
??x
You should separate your repositories such that one (e.g., `modules`) contains reusable, versioned modules which define the "blueprints" or infrastructure components. The other repository (e.g., `live`) contains the configuration for deploying these blueprints into different environments like staging and production.
Example folder structure:
```
/modules
  /common
  /network
  /database

/live
  /staging
    main.tf
  /production
    main.tf
```
x??

---

#### Code Example: Versioned Modules in Repositories
Background context: The text provides an example of how to use different versions of the same module for different environments.
:p How do you set up versioned modules in separate repositories?
??x
You can set up your repository structure as follows:
- `modules`: Contains the reusable, versioned Terraform modules. Each module is a blueprint that defines specific infrastructure components.
- `live`: Contains the configuration files to deploy these modules into different environments.

Here’s an example setup:

```plaintext
/modules
  /network
    main.tf     # Module for network resources
    outputs.tf
    variables.tf
/live
  /staging
    main.tf      # Configures the 'network' module using v0.0.2
  /production
    main.tf      # Configures the 'network' module using v0.0.1
```

In `main.tf` of each environment’s live repository, you would reference the versioned modules like this:
```terraform
module "network" {
  source = "git::https://github.com/your-repo/modules.git//network"
  version = "0.0.2" # or "0.0.1" depending on the environment
}
```
x??

---

#### Setting Up Folder Structure for Live Environment
Background context: To manage different environments effectively, you need to organize your project into separate folders and repositories. This process involves moving specific folders into a live directory and then setting up those directories as Git repositories.

:p What is the first step in setting up the folder structure?
??x
The initial step is to move the stage, prod, and global folders into a folder called live.
x??

---
#### Initializing Git Repositories for Live and Modules Folders
Background context: Once you have organized your project files under the live directory, the next steps are to initialize these directories as separate Git repositories. This allows you to track changes and collaborate effectively.

:p How do you initialize the modules folder as a Git repository?
??x
To initialize the modules folder as a Git repository, follow these commands:
```sh
$ cd modules
$ git init
$ git add .
$ git commit -m "Initial commit of modules repo"
```
This will create an empty Git repository in the modules directory and stage and commit all files.
x??

---
#### Adding Remote Repository and Pushing Code
Background context: After initializing your Git repositories, you need to set up a remote repository on platforms like GitHub. This allows you to push changes to a central location for collaboration or backup.

:p How do you add a remote origin and push the code?
??x
You can add a remote origin and push the code using these commands:
```sh
$ git remote add origin "(URL OF REMOTE GIT REPOSITORY)"
$ git push origin main
```
Replace `(URL OF REMOTE GIT REPOSITORY)` with your actual Git repository URL.
x??

---
#### Tagging for Versioning in Git Repositories
Background context: Using tags is crucial for version control, especially when managing modules. Tags provide a stable reference point and are more user-friendly than commit hashes.

:p How do you create a tag in the Git repository?
??x
To create a tag in your Git repository, use this command:
```sh
$ git tag -a "v0.0.1" -m "First release of webserver-cluster module"
```
This will create a tagged version `v0.0.1` with a message explaining its purpose.
x??

---
#### Specifying Module Version in Terraform Code
Background context: When using modules, specifying the correct version ensures that you are using the intended code. This is especially important when multiple environments or teams need to use consistent versions.

:p How do you specify a specific Git tag as a module version in your Terraform configuration?
??x
You can specify a specific Git tag as a module version by adding the `ref` parameter in your `module` block:
```hcl
module "webserver_cluster" {
  source = "github.com/foo/modules//services/webserver-cluster?ref=v0.0.1"
  cluster_name            = "webservers-stage"
  db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
  db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
  instance_type           = "t2.micro"
  min_size                = 2
  max_size                = 2
}
```
This ensures that your configuration uses the exact version `v0.0.1` of the webserver-cluster module.
x??

---
#### Semantic Versioning for Module Versions
Background context: Semantic versioning provides a structured way to manage versions and communicate changes. It helps in understanding the nature of updates and their impact on users.

:p What is semantic versioning, and why should you use it for modules?
??x
Semantic versioning (MAJOR.MINOR.PATCH) allows you to define different types of changes with specific rules:
- MAJOR: Incompatible API changes.
- MINOR: Backward-compatible functionality additions.
- PATCH: Backward-compatible bug fixes.

Use semantic version tags for module versions because they provide a stable reference and are user-friendly. This helps in clearly communicating the nature of updates to users.
x??

---

#### Using Versioned Modules in Terraform

Background context: When using versioned modules in Terraform, you need to instruct Terraform to download the module code from a specified version. This process is initiated by running `terraform init` with the appropriate module URL.

:p What is the command used to initialize and download the specific version of a module?
??x
The command used is `$ terraform init` followed by specifying the module source URL, for example:
```
$ terraform init -modules-scan-root=modules -module-version=0.3.0
```

This command initializes the modules and downloads the specified version (e.g., `v0.3.0`) from a Git repository.
x??

---

#### Private Git Repositories for Modules

Background context: If your Terraform module is stored in a private Git repository, you need to provide Terraform with authentication credentials to access it. SSH keys are recommended as they do not require hardcoding credentials into the code.

:p How should the source URL be formatted when using a private Git repo?
??x
The source URL for a private Git repo should be of the form:
```
git@github.com:<OWNER>/<REPO>.git//<PATH>?ref=<VERSION>
```

For example, to use the `webserver-cluster` module from a private GitHub repository, the URL would look like this:
```
git@github.com:acme/modules.git//services/webserver-cluster?ref=v0.1.2
```

This ensures that Terraform can authenticate using SSH keys.
x??

---

#### Updating and Testing Modules in Staging

Background context: When making changes to a module, you need to commit the updates, tag them as a version, and then update the `main.tf` file with the new version number. This process is repeated for testing in staging before moving to production.

:p How do you push a new tag to a Git repository after committing changes?
??x
To push a new tag to a Git repository, use the following command:
```
$ git tag -a "v0.0.2" -m "Second release of webserver-cluster"
$ git push origin main --follow-tags
```

This process tags the commit and pushes it to the `origin` remote repository, ensuring that all relevant tags are also pushed.
x??

---

#### Deploying Modules in Different Environments

Background context: Versioned modules allow you to deploy different versions of a module in different environments (e.g., staging, production). This ensures that changes can be tested before being applied to the live environment.

:p How do you update the `main.tf` file for a specific module in a different environment?
??x
To update the `main.tf` file for a specific module in an environment like staging or production, modify the source URL to use the appropriate version. For example:

For staging:
```hcl
module "webserver_cluster" {
  source = "github.com/foo/modules//services/webserver-cluster?ref=v0.0.2"
  cluster_name            = "webservers-stage"
  db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
  db_remote_state_key     = "stage/data-stores/mysql/terraform.tfstate"
  instance_type           = "t2.micro"
  min_size                = 2
  max_size                = 2
}
```

For production:
```hcl
module "webserver_cluster" {
  source = "github.com/foo/modules//services/webserver-cluster?ref=v0.0.1"
  cluster_name            = "webservers-prod"
  db_remote_state_bucket  = "(YOUR_BUCKET_NAME)"
  db_remote_state_key     = "prod/data-stores/mysql/terraform.tfstate"
  instance_type           = "m4.large"
  min_size                = 2
  max_size                = 10
}
```

These changes ensure that the correct version is applied to each environment.
x??

---

#### Using Local File Paths for Modules

Background context: When testing modules on your local machine, using local file paths can help you iterate faster. This avoids the need to commit and push code every time you make a small change.

:p Why might one prefer using local file paths over versioned URLs when developing modules?
??x
When developing modules, using local file paths is preferred because it allows for rapid iteration without the overhead of committing changes, publishing new versions, and running `terraform init` each time. This speeds up development cycles significantly.
```
module "webserver_cluster" {
  source = "./path/to/modules/services/webserver-cluster"
}
```

This approach lets you make changes directly in the module folder and immediately see them reflected in the Terraform plan or apply command.
x??

---

#### Infrastructure as Code
Background context: Infrastructure as Code (IaC) involves managing and provisioning infrastructure resources using machine-readable definition files, similar to how source code is managed. This approach allows for leveraging software engineering best practices in managing infrastructure, making it more reliable, scalable, and easier to maintain.
:p How does IaC help in managing infrastructure?
??x
IaC helps manage infrastructure by applying software engineering best practices such as code reviews, automated testing, versioning, and modular deployment. This ensures that changes are validated before deployment and allows for safe experimentation with different versions in various environments.
x??

---

#### Modules in Infrastructure as Code
Background context: In IaC, modules represent reusable components of infrastructure defined by Terraform configuration files. These modules can be semantically versioned and shared among teams to ensure consistency and reduce redundancy.
:p What is the benefit of using modules in IaC?
??x
The benefit of using modules in IaC includes reusability, maintainability, and ease of testing. By defining infrastructure components as modules, you can reuse tested and documented pieces of code across projects and teams, reducing the risk of errors and increasing deployment reliability.
x??

---

#### Conditional Statements in Terraform
Background context: Terraform provides a way to handle conditional logic using expressions within its configuration language. This allows for creating flexible and configurable infrastructure definitions that can adapt to different requirements or scenarios.
:p How do you implement conditional statements in Terraform?
??x
In Terraform, you can use expressions with logical operators such as `==`, `!=`, `<`, `>`, etc., combined with ternary operators or `case` blocks for more complex conditions. Here's an example using a ternary operator:
```hcl
variable "use_load_balancer" {
  description = "Boolean indicating whether to use a load balancer"
  type        = bool
}

resource "aws_instance" "example" {
  # Other instance configurations...
  tags = {
    Name       = "Microservice"
    LoadBalanced = var.use_load_balancer ? "true" : "false"
  }
}
```
x??

---

#### For-Loops in Terraform
Background context: While Terraform does not natively support for-loops, it provides a `for_each` statement that allows you to iterate over collections of data and create resources based on the elements. This is particularly useful for deploying multiple instances or managing lists of configurations.
:p Can you use for-loops in Terraform?
??x
No, Terraform does not directly support traditional for-loops like those found in C/Java. However, it provides a `for_each` statement that can be used to iterate over collections and create resources based on the elements. Here's an example:
```hcl
variable "microservices" {
  type = list(object({
    name     = string
    instances = number
  }))
}

resource "aws_instance" "microservice_instances" {
  for_each               = var.microservices
  ami                    = "ami-0c55b159210EXAMPLE"
  instance_type          = "t2.micro"
  tags                   = { Name = each.value.name }
  count                  = each.value.instances

  # Other configurations...
}
```
x??

---

#### Zero Downtime Deployment with Terraform
Background context: Achieving zero downtime in infrastructure deployments is crucial for maintaining service availability. Terraform can be used to manage state transitions and rolling updates, ensuring that changes are applied smoothly without disrupting services.
:p How can you use Terraform to roll out changes to a microservice without downtime?
??x
To achieve zero-downtime deployment with Terraform, you can use techniques like blue-green deployments or canary releases. Here's an example of a simple blue-green deployment:
```hcl
resource "aws_instance" "blue" {
  ami                    = "ami-0c55b159210EXAMPLE"
  instance_type          = "t2.micro"
  tags                   = { Color = "Blue", Name = "Microservice Blue" }
  # Other configurations...
}

resource "aws_instance" "green" {
  count                  = var.green_instances
  ami                    = "ami-0c55b159210EXAMPLE"
  instance_type          = "t2.micro"
  tags                   = { Color = "Green", Name = "Microservice Green" }
  # Other configurations...

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_route_table_association" "blue" {
  route_table_id      = aws_route_table.blue.id
  subnet_id           = aws_subnet.default.id
}

resource "aws_route_table_association" "green" {
  count               = var.green_instances
  route_table_id      = aws_route_table.green.id
  subnet_id           = aws_subnet.default.id
}

# Change routing to green instances after they are up and running.
```
x??

---


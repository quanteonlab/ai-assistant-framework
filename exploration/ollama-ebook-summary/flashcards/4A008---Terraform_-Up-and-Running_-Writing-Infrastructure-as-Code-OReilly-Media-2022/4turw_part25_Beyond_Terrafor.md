# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 25)

**Starting Chapter:** Beyond Terraform Modules

---

#### Using Terraform Modules from the Registry

Background context: Terraform allows you to use modules from the Terraform Registry, a central repository for reusable infrastructure components. This approach simplifies dependency management and promotes code reuse by leveraging pre-built, tested modules.

:p How do you consume an open-source module from the Terraform Registry in your Terraform configuration?
??x
You can specify a module using a shorter URL in the `source` argument along with its version via the `version` argument. The general syntax is:

```hcl
module "<NAME>"  {
   source   = "<OWNER>/<REPO>/<PROVIDER>"
   version  = "<VERSION>"
   #(...)
}
```

Here, replace `<NAME>` with a unique identifier for your module in Terraform code, and provide the appropriate values for `source` (owner/repo/provider) and `version`.

For example, to use an RDS module from the Terraform AWS modules registry:

```hcl
module "rds"  {
   source   = "terraform-aws-modules/rds/aws"
   version  = "4.4.0"
   #(...)
}
```

x??

---

#### Private Terraform Registry

Background context: In addition to public modules in the Terraform Registry, you can also use a private registry hosted within your Git repositories for security and control reasons. This allows you to share custom-built or modified modules among team members while keeping them isolated from external dependencies.

:p How can you utilize a private Terraform Registry?
??x
To use a private Terraform Registry, you need to host it on your private Git server (e.g., GitHub Enterprise, Bitbucket Server) and configure it properly. You then reference the hosted modules in the same way as public ones but point to your internal repository URL.

Example of using a private module:

```hcl
module "example"  {
   source = "<git-repo-url>/<path-to-module>"
   version = "0.12.3"
}
```

Where `<git-repo-url>` is the URL of your Git server and `<path-to-module>` points to the specific directory containing the Terraform module.

x??

---

#### Beyond Terraform Modules

Background context: While Terraform is a powerful tool for infrastructure as code, building comprehensive production-grade environments often requires integration with other DevOps tools like Docker, Packer, Chef, Puppet, or Bash scripts. These tools can be used to create custom AMIs, automate the configuration of EC2 instances, and perform other tasks that complement what Terraform can do.

:p How can you integrate non-Terraform code within a Terraform module?
??x
You can use provisioners in Terraform to execute scripts directly from your Terraform configuration. Provisioners allow you to run commands on either the local machine or remote resources, enabling integration with other DevOps tools and workarounds for limitations in Terraform.

Example using `local-exec` provisioner:

```hcl
resource "aws_instance" "example"  {
   ami           = data.aws_ami.ubuntu.id
   instance_type = "t2.micro"
   provisioner "local-exec"  {
      command = "echo \"Hello, World from $(uname -smp)\""
   }
}
```

This example demonstrates running a simple script on the local machine during `terraform apply`.

x??

---

#### Using Provisioners in Terraform

Background context: Provisioners are a key feature of Terraform that allow you to run scripts or commands at various stages of your infrastructure deployment. They can be used for bootstrapping, configuration management, and cleanup tasks.

:p What are the types of provisioners available in Terraform?
??x
Terraform provides several types of provisioners:

- `local-exec`: Executes a script on the local machine.
- `remote-exec`: Executes a script on remote resources (e.g., EC2 instances).
- `file`: Copies files to a remote resource.

Example using `local-exec` provisioner for bootstrapping an instance:

```hcl
resource "aws_instance" "example"  {
   ami           = data.aws_ami.ubuntu.id
   instance_type = "t2.micro"
   provisioner "local-exec"  {
      command = "echo \"Hello, World from $(uname -smp)\""
   }
}
```

x??

---

#### Remote-Exec Provisioner

Background context: The `remote-exec` provisioner is particularly useful for executing scripts on remote resources. It can be configured to run commands on specific instances after they are created by Terraform.

:p How do you use the `remote-exec` provisioner?
??x
To use the `remote-exec` provisioner, you need to specify it within a resource block and configure it with necessary parameters like `connection`, `user`, `private_key`, etc. Here's an example of using `remote-exec` on an AWS EC2 instance:

```hcl
resource "aws_instance" "example"  {
   ami           = data.aws_ami.ubuntu.id
   instance_type = "t2.micro"
   provisioner "remote-exec"  {
      connection  {
         type        = "ssh"
         user        = "ec2-user"
         private_key = file("~/.ssh/id_rsa")
         host        = self.public_ip
      }
      inline  = [
         "echo 'Configuring the instance...'",
         "apt-get update && apt-get install -y nginx",
         "service nginx start"
      ]
   }
}
```

In this example, a script is executed on the EC2 instance after it's launched to configure Nginx.

x??

---

#### Creating a Security Group for SSH Access
Background context: To enable remote execution on an EC2 instance using Terraform, you need to configure a security group that allows inbound connections over SSH. The default SSH port is 22, and this needs to be opened up through a security group.

:p How do you create a security group in Terraform to allow SSH access?
??x
To create a security group in Terraform for allowing SSH access, you use the `aws_security_group` resource. You define an ingress rule that permits traffic on port 22 (the default SSH port). For simplicity in this example, all IP addresses are allowed (`0.0.0.0/0`). In real-world scenarios, it's recommended to restrict this to trusted IPs only.

```hcl
resource "aws_security_group" "instance" {
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```
x??

---

#### Generating SSH Keys in Terraform
Background context: For SSH authentication, you need a private key that is stored securely and can be used to authenticate with the EC2 instance. This example uses `tls_private_key` to automatically generate an RSA private key with 4096 bits.

:p How do you generate a private key using `tls_private_key` in Terraform?
??x
To generate an RSA private key with 4096 bits, use the `tls_private_key` resource. This example demonstrates how to create such a key directly within the Terraform configuration.

```hcl
resource "tls_private_key" "example" {
  algorithm = "RSA"
  rsa_bits  = 4096
}
```
x??

---

#### Associating Public Key with EC2 Instance
Background context: After generating an SSH private and public key pair, the next step is to associate the public key with the EC2 instance so that you can SSH into it. This is done using the `aws_key_pair` resource.

:p How do you upload a public key to AWS using Terraform?
??x
To upload a public key to AWS in Terraform, use the `aws_key_pair` resource and provide its public key value.

```hcl
resource "aws_key_pair" "generated_key" {
  public_key = tls_private_key.example.public_key_openssh
}
```
This step ensures that the public key is associated with the EC2 instance, allowing you to SSH into it using the corresponding private key.
x??

---

#### Deploying an EC2 Instance with SSH Key
Background context: The final step is deploying an EC2 instance and associating it with the security group and the generated SSH key pair. This ensures that the instance can be accessed via SSH.

:p How do you deploy an EC2 instance using Terraform?
??x
To deploy an EC2 instance, use the `aws_instance` resource. You need to specify the AMI ID, instance type, VPC security group IDs, and the key name.

```hcl
resource "aws_instance" "example" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type           = "t2.micro"
  vpc_security_group_ids  = [aws_security_group.instance.id]
  key_name                = aws_key_pair.generated_key.key_name
}
```
This configuration ensures that the EC2 instance is launched with the specified AMI, security group, and SSH key pair.
x??

---

#### Remote-Exec Provisioner Usage
Background context: The remote-exec provisioner is used to run commands on a newly created EC2 instance using SSH. It allows executing scripts during the creation of an AWS resource, such as an `aws_instance`. This provisioner can be particularly useful for running initial setup or bootstrap code.
:p How does the remote-exec provisioner work in Terraform?
??x
The remote-exec provisioner works by defining a set of commands to run on the newly created EC2 instance via SSH. These commands are specified within an inline argument, and Terraform will attempt to connect to the instance multiple times until it successfully connects or times out.

Example configuration:
```hcl
resource "aws_instance" "example" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type           = "t2.micro"
  vpc_security_group_ids  = [aws_security_group.instance.id]
  key_name                = aws_key_pair.generated_key.key_name

  provisioner "remote-exec" {
    inline = [
      "echo \"Hello, World from $(uname -smp)\"",
    ]
  }

  connection {
    type         = "ssh"
    host         = self.public_ip
    user         = "ubuntu"
    private_key  = tls_private_key.example.private_key_pem
  }
}
```
x??

---
#### Connection Block Configuration
Background context: The `connection` block in Terraform is used to specify the details of how to connect to a remote server. For the remote-exec provisioner, it’s crucial to configure this block correctly to ensure successful SSH connections.
:p What does the `connection` block do in the context of remote-exec provisioners?
??x
The `connection` block configures Terraform to use SSH to connect to the EC2 instance's public IP address. It specifies details such as the connection type, host (public IP), user name, and private key.

Example configuration:
```hcl
resource "aws_instance" "example" {
  # ... previous configuration ...
  
  connection {
    type         = "ssh"
    host         = self.public_ip
    user         = "ubuntu"
    private_key  = tls_private_key.example.private_key_pem
  }
}
```
x??

---
#### Creation-Time Provisioners vs. Destroy-Time Provisioners
Background context: Provisioners in Terraform can be configured to run either during the creation or destruction of resources, providing flexibility for different use cases such as initial setup or cleanup tasks.
:p What are the differences between creation-time and destroy-time provisioners?
??x
Creation-time provisioners execute during `terraform apply` and only on the first execution. They are typically used for setting up a resource, like installing dependencies or configuring settings.

Destroy-time provisioners run after `terraform destroy`, just before the resource is deleted. They can be useful for cleanup tasks such as removing temporary files or ensuring resources are properly shut down.

Example configuration for both types:
```hcl
# Creation-Time Provisioner
resource "aws_instance" "example" {
  # ... previous configuration ...
  
  provisioner "remote-exec" {
    when = "create"
    inline = [
      "echo \"Setting up the instance\"",
    ]
  }
}

# Destroy-Time Provisioner
resource "aws_instance" "example" {
  # ... previous configuration ...
  
  provisioner "remote-exec" {
    when = "destroy"
    inline = [
      "echo \"Cleaning up the instance\"",
    ]
  }
}
```
x??

---
#### User Data vs. Remote-Exec Provisioners
Background context: Both user data and remote-exec provisioners can be used to run scripts on a server, but they have different strengths and weaknesses.
:p Why might one prefer using user data over remote-exec provisioners?
??x
User Data is generally preferred over remote-exec provisioners because it requires less management and security overhead. User Data is stored in the AMI metadata and executed by the EC2 instance at launch time, which means you don't need to open SSH access or manage private keys.

Example user data configuration:
```hcl
resource "aws_instance" "example" {
  # ... previous configuration ...
  
  user_data = <<-EOF
              #!/bin/bash
              echo "Hello, World from $(uname -smp)"
              EOF
}
```
x??

---

#### User Data Scripts vs. Provisioners for Auto Scaling Groups (ASGs)
Background context: When working with Auto Scaling Groups (ASGs) in AWS, it's crucial to ensure that all instances within an ASG execute necessary scripts during bootup, including those launched due to auto-scaling or recovery events. Terraform provisioners are not compatible with ASGs; instead, User Data scripts can be used for this purpose.

User Data scripts are executed when the instance is created, and their content can be viewed in the EC2 console under "Actions → Instance Settings → View/Change User Data". Execution logs can also be found on the EC2 instance typically located in `/var/log/cloud-init*.log`. 

:p What is a key difference between using User Data scripts and provisioners for ASGs?
??x
User Data scripts are executed by the EC2 instances during bootup when they are launched, either manually or through an ASG. They can be viewed and debugged via the AWS Management Console or instance logs. Provisioners, on the other hand, are used within Terraform and only take effect while Terraform is running, making them incompatible with ASGs.

```bash
# Example User Data script in a launch template or ASG configuration
User Data:
  "echo 'Hello from $(uname -smp)'> /tmp/welcome.txt"
```
x??

---

#### null_resource for Independent Provisioning
Background context: Sometimes, you might need to run scripts as part of the Terraform lifecycle but not tied directly to a specific resource. The `null_resource` can be used for this purpose.

:p How do you define a `null_resource` in Terraform to execute local scripts?
??x
You define a `null_resource` with provisioners, which allows running scripts without being attached to any "real" resource. Here’s an example:

```hcl
resource "null_resource" "example" {
  provisioner "local-exec" {
    command = "echo 'Hello, World from $(uname -smp)'"
  }
}
```

This `null_resource` will execute the local script every time Terraform is applied.

x??

---

#### Triggers with null_resource
Background context: The `triggers` argument in a `null_resource` can be used to force re-creation of the resource whenever its value changes. This can be useful for executing scripts at specific times or intervals.

:p How do you use the `uuid()` function within `triggers` to execute a local script every time `terraform apply` is run?
??x
You can use the `uuid()` function in the `triggers` argument of a `null_resource` to force re-creation and thus re-execution of provisioners each time Terraform is applied.

```hcl
resource "null_resource" "example" {
  triggers = { uuid = uuid() }

  provisioner "local-exec" {
    command = "echo 'Hello, World from $(uname -smp)'"
  }
}
```

Every `terraform apply` will re-run the local script because the UUID changes each time.

x??

---

#### External Data Source for Fetching Dynamic Data
Background context: The `external` data source in Terraform allows fetching dynamic data and making it available within your code. It works by executing an external command that reads input via JSON on stdin and writes output to stdout, which is then accessible in the Terraform configuration.

:p How do you use the `external` data source to execute a Bash script and retrieve its results?
??x
You can use the `external` data source to fetch dynamic data by executing an external command. Here’s an example:

```hcl
data "external" "echo" {
  program = ["bash", "-c", "cat /dev/stdin"]
  query   = { foo = "bar" }
}

output "echo" {
  value = data.external.echo.result
}

output "echo_foo" {
  value = data.external.echo.result.foo
}
```

This will execute a Bash script that reads `foo=bar` via stdin and echoes it back to stdout. The result is then accessible in the Terraform outputs.

x??

---

#### Production-Grade Infrastructure Checklist
The process involves going through a checklist to ensure all necessary elements for production-grade Terraform code are considered. This includes aspects such as error handling, logging, security, and more, ensuring the infrastructure is robust and maintainable.

:p What is the purpose of the production-grade infrastructure checklist?
??x
The purpose of the production-grade infrastructure checklist is to systematically identify key components required for building reliable, secure, and maintainable Terraform code. By going through this checklist, developers can ensure that their infrastructure code meets certain standards before deployment. This includes checking for error handling, logging mechanisms, security practices, and other best practices.

```yaml
# Example of a simplified production-grade infrastructure checklist
- Error Handling: Implement robust error handling.
- Logging: Ensure logs are well configured and detailed.
- Security: Check for secure configuration and authentication methods.
- Reusability: Design modules to be reusable across projects.
```
x??

---

#### Time Estimate Calculation
After evaluating the production-grade infrastructure checklist, use it along with Table 8-1 to estimate the time required for your project. This involves identifying specific items you will implement versus those you will skip.

:p How do you determine a time estimate for implementing Terraform modules?
??x
To determine a time estimate for implementing Terraform modules, first go through the production-grade infrastructure checklist and identify which items you will be implementing and which you will be skipping. Use this information alongside Table 8-1 to get an idea of the typical effort required for each item.

For example:
- Error Handling: 2 hours
- Logging: 3 hours
- Security: 5 hours

By summing up these estimates, you can provide a detailed time estimate to your boss. This approach ensures that all critical aspects are considered and helps in realistic project planning.

```java
public class EstimateCalculator {
    public int calculateTimeEstimate() {
        int errorHandling = 2; // Hours
        int logging = 3;       // Hours
        int security = 5;      // Hours

        return errorHandling + logging + security;
    }
}
```
x??

---

#### Examples Folder and Best User Experience
Create an `examples` folder to define the best user experience for your modules. Write example code that defines a clean API, including examples for all important permutations of your module.

:p What is the purpose of writing example code in the `examples` folder?
??x
The purpose of writing example code in the `examples` folder is to demonstrate how users can effectively utilize your Terraform modules. This involves creating clear and well-documented examples that showcase different use cases, making it easy for others to understand and deploy your modules.

For instance:
```hcl
# Example: Creating a VPC with subnets
resource "aws_vpc" "example" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "example" {
  vpc_id     = aws_vpc.example.id
  cidr_block = "10.0.1.0/24"
}
```

Include sufficient documentation and reasonable defaults to ensure the examples are easy to deploy.

```java
public class ExampleGenerator {
    public String generateExampleCode() {
        return "# Example: Creating a VPC with subnets\n" +
               "resource \"aws_vpc\" \"example\" {\n" +
               "  cidr_block = \"10.0.0.0/16\"\n" +
               "}\n" +
               "\n" +
               "resource \"aws_subnet\" \"example\" {\n" +
               "  vpc_id     = aws_vpc.example.id\n" +
               "  cidr_block = \"10.0.1.0/24\"\n" +
               "}";
    }
}
```
x??

---

#### Modules Folder and Small Reusable Modules
Create a `modules` folder to implement the API defined in the `examples` folder using small, reusable, composable modules. Utilize Terraform along with tools like Docker, Packer, and Bash for implementation.

:p How do you ensure your Terraform modules are small, reusable, and composable?
??x
To ensure that your Terraform modules are small, reusable, and composable, design them to be modular and self-contained. Each module should handle a specific aspect of the infrastructure and have clear inputs and outputs.

For example:
- `module vpc` handles VPC creation.
- `module subnet` handles subnet configuration within an existing VPC.

Use Terraform's `output` and `variable` features to define clear interfaces between modules, making them composable. Additionally, pin versions for all dependencies, including Terraform core, providers, and any external modules.

```terraform
# Example: vpc/main.tf
module "vpc" {
  source = "./modules/vpc"

  cidr_block = var.cidr_block
}

# Example: vpc/outputs.tf
output "vpc_id" {
  value = module.vpc.id
}
```

Using tools like Docker, Packer, and Bash can help in creating consistent and reliable environments for your modules.

```bash
#!/bin/bash

docker build -t my-vpc-module .
packer build vpc.json
terraform init
terraform apply
```
x??

---

#### Test Folder and Automated Tests
Create a `test` folder to write automated tests for each example. This ensures that your infrastructure code is reliable and robust.

:p What are the benefits of writing automated tests for Terraform modules?
??x
Writing automated tests for Terraform modules provides several benefits:
1. **Reliability**: Ensures that changes in one part of the infrastructure do not break other parts.
2. **Maintainability**: Tests serve as documentation, helping new team members understand how to use and maintain the code.
3. **Regression Testing**: Prevents regressions by catching issues early in the development process.

For example:
```hcl
# Example: Test for VPC creation
resource "aws_vpc" "test_vpc" {
  cidr_block = var.cidr_block

  tags = {
    Name = "Test-VPC"
  }
}

output "vpc_id" {
  value = aws_vpc.test_vpc.id
}
```

Automated tests should cover various scenarios and edge cases to ensure comprehensive coverage.

```java
public class TestVpcCreation {
    public void testVpcCreation() throws Exception {
        // Setup Terraform state
        String terraformState = "provider \"aws\" {\n" +
                                "  region = \"us-west-2\"\n" +
                                "}\n" +
                                "\n" +
                                "resource \"aws_vpc\" \"test_vpc\" {\n" +
                                "  cidr_block = \"10.0.0.0/16\"\n" +
                                "}\n";
        
        // Run Terraform apply
        String output = runTerraform(terraformState);
        
        // Assert VPC ID is present in the output
        assertTrue(output.contains("aws_vpc.test_vpc.id"));
    }
}
```
x??

---

#### Manual Testing Basics
Background context: The process of testing Terraform code manually by deploying it to a real environment, as opposed to using localhost. This is due to the nature of infrastructure-as-code tools like Terraform that require deployment to actual resources.

:p What is the primary difference between manual testing with general-purpose programming languages and Terraform?

??x
In Terraform, you cannot use `localhost` for testing because it requires real AWS resources (e.g., ALBs, security groups) which are not available on your local machine. Therefore, manual testing in Terraform involves deploying to a real environment like AWS.

Manual tests with Terraform are conducted by running `terraform apply` and `terraform destroy`, just as you have done throughout the book for module examples. This process allows you to verify that the infrastructure behaves as expected before applying changes to production environments.
x??

---

#### Testing Web Server Code in Ruby
Background context: An example of manually testing code written in a general-purpose programming language, specifically a simple web server in Ruby.

:p How would you write a script to test a simple web server implemented in Ruby?

??x
You can create a script that runs the web server and tests its responses. Here's an example:

```ruby
class WebServer < WEBrick::HTTPServlet::AbstractServlet
  def do_GET(request, response)
    case request.path
    when "/"
      response.status = 200
      response['Content-Type'] = 'text/plain'
      response.body = 'Hello, World'
    when "/api"
      response.status = 201
      response['Content-Type'] = 'application/json'
      response.body = '{"foo":"bar"}'
    else
      response.status = 404
      response['Content-Type'] = 'text/plain'
      response.body = 'Not Found'
    end
  end
end

if __FILE__ == $0
  server = WEBrick::HTTPServer.new(:Port => 8000)
  server.mount '/', WebServer
  trap('INT') do
    server.shutdown
  end
  server.start
end
```

This script runs the web server on port 8000, and you can test it using a web browser or `curl` commands.

```sh
$ ruby web-server.rb
[2019-05-25 14:11:52] INFO  WEBrick 1.3.1 
[2019-05-25 14:11:52] INFO  ruby 2.3.7 (2018-03-28) [universal.x86_64-darwin17]
[2019-05-25 14:11:52] INFO  WEBrick::HTTPServer#start: pid=19767 port=8000

$ curl localhost:8000/
Hello, World
```

The `if __FILE__ == $0` condition ensures that the script runs only if it is called directly from the command line.
x??

---

#### Testing Terraform Code Manually
Background context: Manual testing of Terraform code by deploying and destroying resources on a real environment to ensure they behave as expected.

:p How do you manually test a module in Terraform, like the ALB example provided?

??x
You can manually test a Terraform module by creating an example configuration file that uses the module. For instance, if you have an `alb` module with resources defined in `modules/networking/alb/main.tf`, you can create an example configuration at `examples/alb/main.tf`:

```hcl
provider "aws" {
  region = "us-east-2"
}

module "alb" {
  source      = "../../modules/networking/alb"
  alb_name    = "terraform-up-and-running"
  subnet_ids  = data.aws_subnets.default.ids
}
```

Then, you can apply this example configuration:

```sh
$ terraform apply

Apply complete. Resources: 5 added, 0 changed, 0 destroyed.

Outputs:

alb_dns_name = "hello-world-stage-477699288.us-east-2.elb.amazonaws.com"
```

After applying the changes, you can test the ALB using tools like `curl` to ensure that it returns the expected responses:
```sh
$ curl -s -o /dev/null -w "%{http_code}" "hello-world-stage-477699288.us-east-2.elb.amazonaws.com"
404
```

This process ensures that your ALB is working correctly before deploying it to a production environment.
x??

---


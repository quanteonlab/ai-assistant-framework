# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 19)

**Starting Chapter:** Resources and Data Sources

---

#### IAM Role and GitHub Actions Authentication

Background context: The text explains how to authenticate GitHub Actions to an AWS account using an IAM role. It discusses setting up temporary credentials for Terraform to use when running builds.

:p How do you configure a GitHub Action to assume an IAM role during a build?
??x
You need to provide the `id-token` permission and specify the IAM role to assume in the `configure-aws-credentials` action. Here's how:

```yaml
permissions:
  id-token: write

- uses: aws-actions/configure-aws-credentials@v1
  with:
    role-to-assume: arn:aws:iam::123456789012:role/example-role
    aws-region: us-east-2

- uses: hashicorp/setup-terraform@v1
  with:
    terraform_version: 1.1.0
    terraform_wrapper: false
  run: |
    terraform init
    terraform apply -auto-approve
```

x??

---

#### Storing Secrets in Environment Variables

Background context: The text emphasizes the importance of keeping sensitive information out of your code and suggests using environment variables to pass secrets. It also discusses potential methods for securely storing and retrieving these environment variables.

:p How do you declare sensitive variables in Terraform?
??x
You can declare sensitive variables in Terraform like this:

```hcl
variable "db_username" {
  description = "The username for the database"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "The password for the database"
  type        = string
  sensitive   = true
}
```

These variables are marked with `sensitive = true`, which means Terraform will not log their values when you run commands like `terraform plan` or `terraform apply`.

x??

---

#### Using Environment Variables in Terraform

Background context: The text explains how to use environment variables as a secure way to pass secrets into your Terraform code. It provides an example of using these variables with the AWS database resource.

:p How do you pass sensitive values from environment variables to Terraform resources?
??x
You can pass sensitive values from environment variables to Terraform resources like this:

```hcl
resource "aws_db_instance" "example" {
  identifier_prefix = "terraform-up-and-running"
  engine            = "mysql"
  allocated_storage = 10
  instance_class    = "db.t2.micro"
  skip_final_snapshot = true
  db_name           = var.db_name

  # Pass the secrets to the resource
  username  = var.db_username
  password  = var.db_password
}
```

To set these environment variables, you would use:

```sh
export TF_VAR_db_username=YOUR_DB_USERNAME
export TF_VAR_db_password=YOUR_DB_PASSWORD
```

x??

---

#### Using Environment Variables for Secret Management
Environment variables are a common way to manage secrets, as they can be easily set and accessed. They do not require additional cost compared to some other secret management solutions.

:p What is one drawback of using environment variables for managing secrets?
??x
One major drawback is that the code itself does not enforce any security properties since all secret management happens outside of Terraform. This means it's possible for someone to manage secrets insecurely, such as storing them in plain text.
x??

---
#### Encrypted Files for Secret Management
Encrypted files involve encrypting secrets and checking the encrypted data into version control. An encryption key is needed to perform this operation securely.

:p What is the purpose of creating a KMS Customer Managed Key (CMK) in AWS?
??x
The purpose of creating a KMS CMK is to provide a secure way to manage encryption keys for encrypting secrets. By using KMS, you ensure that your secrets are encrypted and can be managed securely.
x??

---
#### Creating an IAM Policy Document for KMS
Creating an IAM policy document allows defining who has permission to use the CMK.

:p How do you create an IAM policy document in Terraform to give admin permissions over a CMK?
??x
To create an IAM policy document in Terraform that gives admin permissions over a CMK, you can use the `aws_iam_policy_document` data source. Here is how:

```hcl
data "aws_caller_identity" "self" {}

data "aws_iam_policy_document" "cmk_admin_policy" {
  statement {
    effect = "Allow"
    resources = ["*"]
    actions  = ["kms:*"]

    principals {
      type         = "AWS"
      identifiers  = [data.aws_caller_identity.self.arn]
    }
  }
}
```

This code fetches the current user's ARN and uses it to create an IAM policy that allows all KMS actions for the current user.
x??

---
#### Creating a CMK with AWS KMS
Creating a CMK involves defining its key policy, which is then used to create the CMK.

:p How do you create a CMK in AWS using Terraform?
??x
To create a CMK in AWS using Terraform, first define the key policy and then use it to create the CMK. Here’s an example:

```hcl
data "aws_caller_identity" "self" {}

data "aws_iam_policy_document" "cmk_admin_policy" {
  statement {
    effect = "Allow"
    resources = ["*"]
    actions  = ["kms:*"]

    principals {
      type         = "AWS"
      identifiers  = [data.aws_caller_identity.self.arn]
    }
  }
}

resource "aws_kms_key" "cmk" {
  policy = data.aws_iam_policy_document.cmk_admin_policy.json
}
```

This example creates a key policy that allows the current user to perform all KMS actions and then uses this policy to create a CMK.
x??

---
#### Creating an Alias for a CMK
An alias is created to provide a human-friendly identifier for your CMK.

:p How do you create an alias for a CMK in AWS using Terraform?
??x
To create an alias for a CMK in AWS, use the `aws_kms_alias` resource. Here’s how:

```hcl
resource "aws_kms_alias" "cmk" {
  name          = "alias/kms-cmk-example"
  target_key_id = aws_kms_key.cmk.id
}
```

This code creates an alias named `alias/kms-cmk-example` for the CMK, making it easier to reference in commands and scripts.
x??

---

#### Key Concepts in AWS KMS and Terraform for Secret Management

Background context explaining the concept. This section covers how to use AWS Key Management Service (KMS) to securely manage secrets, particularly database credentials, using Terraform. The focus is on encrypting sensitive data, storing it safely in version control, and decrypting it within Terraform configurations.

:p What are the steps involved in encrypting a file with AWS KMS?
??x
The steps involve creating a ciphertext from plaintext by using the `aws kms encrypt` command. Here's an example of how to do this using a Bash script:

```bash
CMK_ID="$1"
AWS_REGION="$2"
INPUT_FILE="$3"
OUTPUT_FILE="$4"

echo "Encrypting contents of $INPUT_FILE using CMK$ CMK_ID..."
ciphertext=$(aws kms encrypt \
   --key-id "$CMK_ID" \
   --region "$AWS_REGION" \
   --plaintext "fileb://$INPUT_FILE" \
   --output text \
   --query CiphertextBlob )

echo "Writing result to $OUTPUT_FILE..."
echo "$ciphertext" > "$ OUTPUT_FILE"

echo "Done."
```

This script takes a KMS CMK ID, AWS region, input file path, and output file path as parameters. It encrypts the contents of the input file using the specified CMK and writes the ciphertext to the output file.

x??

---
#### Decrypting Secrets with Terraform

Background context explaining how to use `aws_kms_secrets` data source in Terraform to decrypt secrets stored in encrypted files.

:p How do you decrypt a secrets file within a Terraform configuration?
??x
You can use the `data aws_kms_secrets` data source to read and decrypt a secrets file. Here's an example:

```hcl
data "aws_kms_secrets" "creds" {
   secret {
     name     = "db"
     payload  = file("${path.module}/db-creds.yml.encrypted")
   }
}
```

This code reads the encrypted `db-creds.yml.encrypted` file from disk and decrypts it using KMS, assuming you have appropriate permissions.

Next, parse the YAML content:

```hcl
locals {
   db_creds = yamldecode(data.aws_kms_secrets.creds.plaintext["db"])
}
```

This local variable `db_creds` contains the decrypted secrets which can be accessed in your Terraform configuration as needed.

x??

---
#### Using Decrypted Secrets in Resources

Background context explaining how to use decrypted secrets in resource configurations, specifically with the AWS database instance example provided.

:p How do you use the decrypted secrets in an AWS database instance resource?
??x
After decrypting and parsing the secrets, you can pass them as variables to the relevant Terraform resources. For instance:

```hcl
resource "aws_db_instance" "example" {
   identifier_prefix = "terraform-up-and-running"
   engine             = "mysql"
   allocated_storage  = 10
   instance_class     = "db.t2.micro"
   skip_final_snapshot = true
   db_name            = var.db_name

   # Pass the secrets to the resource
   username  = local.db_creds.username
   password  = local.db_creds.password
}
```

This configuration ensures that the database instance is created with the correct credentials, while keeping the sensitive information safe by not checking the plaintext file into version control.

x??

---

#### Working with Encrypted Files
Background context: When dealing with encrypted files, especially in a development or automated pipeline environment, managing secrets can be cumbersome. The process often involves local decryption, editing, and re-encryption, which introduces risks such as accidental exposure of plain-text data.

:p What is the primary issue when working with encrypted files locally?
??x
The primary issue is that you have to manually handle file encryption and decryption processes using commands like `aws kms decrypt` and `aws kms encrypt`, which can be error-prone and tedious. This process increases the risk of accidentally checking plain-text secrets into version control or leaving them on your computer.
x??

---

#### Using sops for Encrypted Files
Background context: The tool `sops` simplifies working with encrypted files by handling encryption and decryption transparently, reducing the complexity and risks associated with manual processes.

:p How does the `sops` tool simplify working with encrypted files?
??x
`sops` simplifies working with encrypted files by automatically decrypting them when you run a command like `sops <FILE>`. It opens your default text editor with the plain-text contents, allowing you to make changes. Upon exiting the editor, it re-encrypts the file seamlessly. This reduces the need for manual encryption and decryption commands and minimizes the risk of accidentally checking in plain-text secrets.

Example usage:
```sh
# Run sops to edit a file encrypted with AWS KMS
sops -e -d <filename>.ciphertext
```

Here, `-e` enables inline editing, and `-d` displays decrypted content. Upon saving and exiting the editor, `sops` re-encrypts the file.
x??

---

#### Advantages of Using Encrypted Files
Background context: Storing secrets in encrypted form provides several benefits over storing them as plain text. These include version control integration, ease of retrieval, and support for various encryption methods.

:p What are the main advantages of using encrypted files?
??x
The main advantages of using encrypted files are:
- **Keeping plain-text secrets out of your code and version control system**: This ensures that sensitive data is not accidentally checked into repositories.
- **Versioning secrets**: Since secrets are stored in an encrypted format, they can be managed alongside other code changes, reducing the risk of configuration errors across environments (e.g., staging vs. production).
- **Ease of retrieval**: Secrets can be easily retrieved if they support native decryption by Terraform or a third-party plugin.
- **Versatility in encryption options**: Supports AWS KMS, GCP KMS, PGP, etc., providing flexibility based on your environment's needs.
- **Code-centric approach**: Everything is defined within the codebase without requiring extra manual steps.

Example of using sops with Terraform:
```hcl
locals {
  decrypted_value = sops_decrypt_file("path/to/secret.ciphertext")
}
```
x??

---

#### Drawbacks of Using Encrypted Files
Background context: While encrypted files offer several benefits, they also come with challenges related to complexity and security management.

:p What are the main drawbacks of using encrypted files?
??x
The main drawbacks of using encrypted files include:
- **Complexity in storage**: You need to run commands like `aws kms encrypt` or use tools like sops, which may have a learning curve.
- **Harder integration with automated tests**: Requires extra effort to provide encryption keys and test data for different environments.
- **Secrets vulnerability due to version control**: Encrypted secrets are stored in version control but can still be compromised if the key is ever exposed or misused.
- **Limited auditability**: It's challenging to track who accessed specific secrets, especially when using cloud-based KMS services.

Example scenario:
```sh
# Example of running sops for decryption and re-encryption
sops -e -d <filename>.ciphertext > plainfile.txt  # Decryption step
<make changes in plainfile.txt>                # Edit the file
cat plainfile.txt | sops -e -i <filename>.ciphertext  # Re-encryption step
```
x??

---

#### Cost of Managed Key Services
Background context: This section explains the cost implications for using managed key services like AWS KMS. The costs are typically low, with minor charges per API call and key storage.

:p What is the monthly cost of storing keys in AWS KMS?
??x
The monthly cost to store a single key in AWS KMS can be as low as $1 due to minimal API call charges, but it could go up depending on usage. For typical usage, ranging from a few keys for small deployments to dozens or more for larger ones, costs are generally between $1 and $50 per month.
x??

---
#### Secret Management Practices
Background context: This section highlights the challenges in standardizing secret management practices across different teams using various methods.

:p What are some common mistakes developers make when managing secrets?
??x
Common mistakes include not encrypting sensitive data, incorrectly handling encryption keys, and accidentally checking plain-text files into version control.
x??

---
#### Using AWS Secrets Manager for Secret Storage
Background context: This section describes the steps to use AWS Secrets Manager to store and retrieve database credentials securely.

:p How can you read a secret stored in AWS Secrets Manager using Terraform?
??x
You can use the `aws_secretsmanager_secret_version` data source to read the secret. Here is how you can do it:

```hcl
data "aws_secretsmanager_secret_version" "creds" {
  secret_id = "db-creds"
}
```

After retrieving the secret, parse the JSON string using the `jsondecode` function:

```hcl
locals {
  db_creds = jsondecode(data.aws_secretsmanager_secret_version.creds.secret_string)
}
```

Now you can use `local.db_creds.username` and `local.db_creds.password` to pass the secrets into resources like an AWS database instance.

Example Terraform resource configuration:

```hcl
resource "aws_db_instance" "example" {
  identifier_prefix = "terraform-up-and-running"
  engine            = "mysql"
  allocated_storage = 10
  instance_class    = "db.t2.micro"
  skip_final_snapshot = true
  db_name           = var.db_name
  username          = local.db_creds.username
  password          = local.db_creds.password
}
```
x??

---
#### Advantages of Using Secret Stores
Background context: This section lists the benefits of using centralized secret stores like AWS Secrets Manager, Google Secret Manager, Azure Key Vault, or HashiCorp Vault.

:p What are some advantages of using a centralized secret store?
??x
Some key advantages include:
- Keeping plain-text secrets out of code and version control.
- Everything is defined in code with no manual steps required.
- Ease of storing secrets through web UIs.
- Support for rotating and revoking secrets as needed.
- Detailed audit logs showing access to data.
- Enforcing specific types of encryption, storage, and access patterns.

These features help standardize practices across teams.
x??

---
#### Drawbacks of Using Secret Stores
Background context: This section outlines the potential downsides of using managed secret stores such as AWS Secrets Manager.

:p What are some drawbacks associated with using a centralized secret store like AWS Secrets Manager?
??x
Some notable drawbacks include:
- Configuration errors due to not versioning secrets.
- Costs related to storing and retrieving data, which can add up in larger deployments.
- Additional costs for self-managed stores (e.g., running HashiCorp Vault on EC2 instances).
- High developer time costs required for setting up and managing the secret store.

These factors could increase overall expenses, especially if developer overhead is significant.
x??

---

#### Storing Secrets in Terraform State Files
Background context: When using Terraform, any secrets you pass to your resources and data sources will be stored in plain text within the state files. This is a critical security risk if these files are stored locally or checked into version control systems.

:p What happens to secrets when they are passed to Terraform resources?
??x
When secrets are passed to Terraform resources, they are stored in plain text within the `terraform.tfstate` file. For example:
```hcl
resource "aws_db_instance" "example" {
  identifier_prefix = "terraform-up-and-running"
  engine            = "mysql"
  allocated_storage = 10
  instance_class    = "db.t2.micro"
  skip_final_snapshot = true
  db_name           = var.db_name
  username          = local.db_creds.username
  password          = local.db_creds.password
}
```
The `username` and `password` are stored in plain text in the state file. This can be a significant security risk.

x??

---

#### Using Terraform Backends for State File Security
Background context: To mitigate the risks associated with storing secrets in plain text within the state files, it is recommended to use Terraform backends that support encryption such as S3, GCS, or Azure Blob Storage. These backends help ensure that the state file is both encrypted in transit and at rest.

:p How should you store your Terraform state for enhanced security?
??x
You should store your Terraform state using a backend that supports encryption, such as AWS S3, Google Cloud Storage (GCS), or Azure Blob Storage. These backends encrypt your state files not only during transmission via TLS but also when stored on disk using AES-256.

For example, to use an S3 backend, you can configure it in the `terraform.tf` file:
```hcl
backend "s3" {
  bucket = "your-s3-bucket"
  key    = "path/to/terraform/state"
  region = "us-west-2"
}
```
This configuration ensures that your state is stored securely and can only be accessed by authorized users.

x??

---

#### Managing Secrets in Plan Files
Background context: Similar to the state files, any secrets passed into resources during planning will also be stored in plain text within the plan file. This poses a significant security risk if the plan files are not properly secured.

:p How should you handle secrets when using Terraform plan files?
??x
To manage secrets in plan files, it is crucial to encrypt these files both in transit and at rest. You can store your plan files in an S3 bucket that supports encryption, ensuring that sensitive information remains secure.

For example:
```bash
$terraform plan -out=example.plan
```
If you save a plan file like `example.plan`, the database username and password will be stored in plain text within this file. To encrypt it, you could use AWS S3 with server-side encryption (SSE-S3):
```bash
# Assuming your S3 bucket supports SSE-S3$ aws s3 cp example.plan s3://your-s3-bucket/example.plan --sse AES256
```
This ensures that the plan file is encrypted on disk.

x??

---

#### Strict Access Control for Plan Files
Background context explaining why it is important to control access to Terraform plan files, especially when they may contain secrets. The example provided illustrates that sensitive information should be protected with as much care as the secrets themselves.

:p How do you ensure strict access control for your Terraform plan files stored in S3?
??x
To ensure strict access control for Terraform plan files stored in an S3 bucket, configure an IAM Policy that grants access to a small handful of trusted developers or the CI server used for deployment. This policy should be carefully crafted to minimize unnecessary permissions and restrict access only to those who need it.

For example:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::your-bucket-name/terraform-plan-files/*"
        }
    ]
}
```
This IAM Policy allows access to specific plan files in the S3 bucket but restricts it to only trusted users or a CI server.
x??

---

#### Storing Secrets in Plain Text
Background context explaining why storing secrets in plain text is not recommended and should be avoided.

:p Why should you avoid storing secrets in plain text?
??x
Storing secrets in plain text is not secure and can lead to unauthorized access if the code or configuration files are compromised. Secrets stored in plain text are easily readable by anyone with access to the file, which increases the risk of exposure.

Avoid storing secrets directly in your Terraform scripts or any other codebase. Instead, use environment variables, encrypted files, or centralized secret stores to manage and retrieve them securely.
x??

---

#### Passing Secrets to Providers
Background context explaining different methods for passing secrets to Terraform providers based on whether they are human users or machine users.

:p How can human users pass secrets to Terraform providers?
??x
Human users can use personal secrets managers, such as HashiCorp Vault, and set environment variables. This method allows the user to securely manage and retrieve secrets without hardcoding them in scripts or configurations.

Example using environment variables:
```bash
export TF_VAR_secret_key="your-secret-value"
terraform apply
```
x??

---

#### Passing Secrets to Resources and Data Sources
Background context explaining different methods for passing secrets to Terraform resources and data sources, including environment variables, encrypted files, and centralized secret stores.

:p How can you pass secrets to Terraform resources and data sources?
??x
To pass secrets to Terraform resources and data sources, use the following methods:
- **Environment Variables**: Store sensitive information in environment variables.
- **Encrypted Files**: Encrypt the files containing sensitive information before storing them.
- **Centralized Secret Stores**: Use a centralized secret store like HashiCorp Vault.

For example, using environment variables:
```bash
export TF_VAR_database_password="your-secret-value"
```
x??

---

#### Securing State and Plan Files
Background context explaining that Terraform stores secrets in state files and plan files, which should be encrypted and access-controlled.

:p How do you ensure the security of your Terraform state and plan files?
??x
To secure your Terraform state and plan files:
1. **Encrypt the files**: Use encryption both in transit and at rest.
2. **Strictly control access**: Limit who can read or modify these files to only trusted users.

For example, you can use AWS KMS to encrypt S3 objects containing state files:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": "arn:aws:s3:::your-bucket-name/terraform-state-files/*",
            "Condition": {"StringEquals": {"s3:x-amz-server-side-encryption": "aws:kms"}}
        }
    ]
}
```
x??

---


# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 18)

**Starting Chapter:** The Interface You Use to Access Secrets

---

#### Types of Secrets You Store
Background context: In the provided text, different types of secrets are categorized based on their ownership and purpose. Personal secrets belong to individuals, customer secrets pertain to customers or end-users, and infrastructure secrets relate to the underlying infrastructure.

:p What are the three primary types of secrets mentioned in this section?
??x
The three primary types of secrets are:
1. **Personal Secrets**: These belong to an individual and examples include usernames and passwords for websites, SSH keys, and PGP keys.
2. **Customer Secrets**: These belong to your customers or end-users. Examples include customer login credentials (usernames and passwords), personally identifiable information (PII), and personal health information (PHI).
3. **Infrastructure Secrets**: These belong to the infrastructure itself. Examples are database passwords, API keys, and TLS certificates.

This classification helps in understanding how different types of secrets should be handled differently from a security perspective.
x??

---

#### Storing Secrets: File-Based vs Centralized
Background context: The text discusses two common strategies for storing secrets—file-based secret stores and centralized secret stores. Each has its own merits and challenges, especially concerning the management of encryption keys.

:p What are the two main strategies discussed in the text for storing secrets?
??x
The two main strategies for storing secrets mentioned are:
1. **File-Based Secret Stores**: These store secrets in encrypted files that are typically checked into version control systems.
2. **Centralized Secret Stores**: These are web services that encrypt and store secrets using a data store like MySQL, PostgreSQL, or DynamoDB.

File-based secret stores require managing encryption keys securely, often through key management services (KMS) provided by cloud providers. Centralized secret stores handle the encryption and storage of secrets over a network.
x??

---

#### Storing Secrets: File-Based Strategy
Background context: The text explains that file-based secret stores use encrypted files stored in version control systems. Managing these keys securely is crucial to prevent unauthorized access.

:p What are the challenges associated with storing secrets using a file-based strategy?
??x
The challenges associated with storing secrets using a file-based strategy include:

1. **Key Management**: The encryption key itself needs secure storage, which can be problematic since it cannot be stored as plain text in version control.
2. **Security Risk**: If the key is compromised, all encrypted secrets become vulnerable.
3. **Complexity**: Implementing and managing the KMS (Key Management Service) securely adds complexity.

To address these challenges, common solutions involve storing keys in services like AWS KMS, GCP KMS, or Azure Key Vault. These services are trusted to securely store and manage access to the encryption keys.
x??

---

#### Storing Secrets: Centralized Strategy
Background context: The text describes centralized secret stores as web services that handle the encryption and storage of secrets using a data store such as MySQL or DynamoDB.

:p What is a key benefit of using a centralized secret store?
??x
A key benefit of using a centralized secret store is:

- **Simplified Key Management**: These systems manage the encryption keys for you, reducing the risk associated with manually managing keys.
- **Enhanced Security**: They provide robust security features and access controls to ensure that secrets are only accessible to authorized users.

Centralized secret stores simplify the process of storing and managing secrets by abstracting away the complexities of key management. This approach enhances overall security and reduces the burden on developers or administrators.
x??

---

#### Using Key Management Services (KMS)
Background context: The text explains how KMS services provided by cloud providers like AWS, GCP, and Azure are used to securely store encryption keys.

:p What is a key management service (KMS) and what role does it play in secret storage?
??x
A **Key Management Service (KMS)** is a cloud-based service that helps you create, control, and use encryption keys. It plays the following roles:

- **Secure Key Storage**: KMS securely stores your encryption keys without exposing them.
- **Access Controls**: It manages access controls to ensure that only authorized entities can use or manage these keys.

For example, using AWS KMS:
```java
// Pseudocode for accessing AWS KMS in Java
import com.amazonaws.services.kms.AWSKMS;
import com.amazonaws.services.kms.model.*;

public class KmsExample {
    public static void main(String[] args) throws Exception {
        // Initialize the AWS KMS client
        AWSKMS kmsClient = AWSKMSClientBuilder.defaultClient();
        
        // Get a key from KMS
        String keyId = "1234abcd-12ab-34cd-56ef-1234567890ab";
        DecryptRequest request = new DecryptRequest().withCiphertextBlob(ByteBuffer.wrap("encryptedData".getBytes()));
        DecryptResponse response = kmsClient.decrypt(request);
        
        // Use the decrypted data
        System.out.println(new String(response.getPlaintext().array()));
    }
}
```

In this example, AWS KMS is used to decrypt an encrypted blob of data. The key ID is securely managed by AWS, providing a secure and centralized way to handle encryption keys.
x??

---

#### Encryption Key Management
Background context: An encryption key is used to encrypt and decrypt data. Typically, it is managed by the service itself or relies on a cloud provider’s Key Management Service (KMS). The key can be stored in various secret management tools, which are designed to handle secure storage, retrieval, and usage of these keys.
:p How is an encryption key typically managed?
??x
An encryption key is usually managed by the service itself or through a cloud provider's Key Management Service (KMS).
x??

---

#### Accessing Secrets via API, CLI, UI
Background context: Secret management tools offer multiple ways to access secrets. APIs are useful for programmatic retrieval, while CLIs and Uis provide more convenient methods for developers and teams.
:p What are the different interfaces available to access secrets in secret management tools?
??x
Secrets can be accessed via an API, CLI, or UI. The API is suitable for programmatic access, whereas the CLI and UI offer convenience for developers and team members.
x??

---

#### Example of Using API to Retrieve Secrets
Background context: APIs are often used by applications to retrieve secrets securely during boot-up or runtime. This ensures that sensitive information like database passwords can be fetched without being hardcoded in the application code.
:p How might an application use an API to retrieve a secret, such as a database password?
??x
An application can use an API call to fetch a database password from a centralized secret store when booting up or running. For example:

```java
public class SecretRetriever {
    private String getDatabasePassword() {
        // Example REST API call using HTTP client
        HttpClient httpClient = HttpClients.createDefault();
        HttpGet httpGet = new HttpGet("https://secrets-store.example.com/api/secrets/db_password");
        
        try (CloseableHttpResponse response = httpClient.execute(httpGet)) {
            if (response.getStatusLine().getStatusCode() == 200) {
                String password = EntityUtils.toString(response.getEntity());
                return password;
            } else {
                throw new RuntimeException("Failed to retrieve secret: " + response.getStatusLine());
            }
        } catch (IOException e) {
            throw new RuntimeException("Error during API call", e);
        }
    }
}
```
x??

---

#### Types of Secret Storage
Background context: Different secret management tools store secrets in various ways, including centralized services or file-based storage. These methods differ based on whether the tool is infrastructure-focused or personal.
:p What are the main types of secret storage provided by different secret management tools?
??x
Secrets can be stored either centrally in a service (like HashiCorp Vault) or locally in files (like sops). Tools like AWS Secrets Manager and Azure Key Vault provide centralized services, while others such as git-secret use file-based storage.
x??

---

#### Comparison of Secret Management Tools
Background context: The table provided compares popular secret management tools based on the types of secrets they manage, where the secrets are stored, and how they can be accessed. This comparison helps in selecting a tool that fits specific needs.
:p How is the functionality of different secret management tools typically compared?
??x
The functionality of different secret management tools is compared by looking at three main aspects: what kinds of secrets the tool manages (infrastructure or personal), where these secrets are stored, and how they can be accessed (UI, API, CLI).
x??

---

#### Example Tool: HashiCorp Vault
Background context: HashiCorp Vault is an example of a centralized secret management service. It supports multiple interfaces for managing secrets.
:p What does HashiCorp Vault provide in terms of secret management?
??x
HashiCorp Vault provides a centralized service for managing secrets and can be accessed via UI, API, or CLI. It is designed to securely store and manage sensitive information like database credentials, API keys, etc.
x??

---

#### Example Tool: AWS Secrets Manager
Background context: AWS Secrets Manager is another example of a centralized secret management tool that supports secure storage and retrieval of secrets through APIs and CLIs.
:p How does AWS Secrets Manager handle the storage and retrieval of secrets?
??x
AWS Secrets Manager handles the storage and retrieval of secrets by providing a centralized service accessible via APIs and CLIs. It securely stores sensitive information such as database credentials and retrieves them when needed, ensuring they are not hard-coded in application code.
x??

---

#### Example Tool: 1Password
Background context: 1Password is an example of a personal secret management tool that offers both UI and API access for managing secrets on the user's devices or through network requests.
:p What makes 1Password unique among secret management tools?
??x
1Password is unique as it is designed for personal use, offering centralized service management with both UI and API access. It focuses on securely storing and managing sensitive information for individual users across multiple devices.
x??

---

#### Storing Secrets Directly in Code
Background context: The provided text emphasizes the importance of not storing secrets directly in the code, especially for providers like AWS. This method is insecure and impractical as it hardcodes credentials across all users and environments.

:p How should you store secrets when working with Terraform, specifically for authentication to a provider?
??x
Storing secrets directly in plain text within the code is not secure or practical. Instead, consider using more secure methods such as environment variables, secret management tools (e.g., HashiCorp Vault), or Terraform backend configurations.

Example of storing credentials securely:
```sh
export TF_VAR_aws_access_key=$(cat ~/.aws/credentials | grep aws_access_key_id | awk '{print $2}' | sed 's/"//g')
export TF_VAR_aws_secret_key=$(cat ~/.aws/credentials | grep aws_secret_access_key | awk '{print $2}' | sed 's/"//g')
```
This script reads the AWS credentials from a file and stores them as environment variables.

```sh
provider "aws" {
  region = "us-east-2"
  access_key = "${var.aws_access_key}"
  secret_key = "${var.aws_secret_key}"
}
```

x??

---

#### Differentiating Human Users vs. Machine Users
Background context: The text discusses the differences in handling secrets for human users (developers) and machine users (CI servers). Each group requires different approaches to securely manage credentials.

:p How do you handle secret storage for developers running Terraform on their own computers?
??x
For developers running Terraform on their local machines, consider using environment variables or a tool like HashiCorp Vault. This approach keeps secrets out of the source code and ensures that each developer can use different credentials as needed.

Example: Using an `.env` file for storing sensitive information.
```sh
aws_access_key=your-access-key-here
aws_secret_key=your-secret-key-here
```

Then, in your Terraform configuration:
```hcl
provider "aws" {
  region = "us-east-2"
  access_key = "${var.aws_access_key}"
  secret_key = "${var.aws_secret_key}"
}
```
x??

---

#### Secure Secrets for Machine Users
Background context: The text highlights the need for secure storage of credentials in environments where no human interaction is present, such as CI servers. This often involves more robust and centralized solutions.

:p How do you handle secrets management for automated systems (CI servers) running Terraform?
??x
For machine users like CI servers, use a more robust secret management solution such as HashiCorp Vault or another centralized service designed to securely store and manage credentials at scale.

Example: Using HashiCorp Vault with Terraform.
```hcl
provider "aws" {
  region = "us-east-2"
  access_key = "${lookup(var.aws_credentials, "access_key", null)}"
  secret_key = "${lookup(var.aws_credentials, "secret_key", null)}"
}

variable "aws_credentials" {
  type = map(string)
}
```

You can then use a Vault plugin or direct API calls to fetch these secrets:
```sh
vault read -field=access_key aws/creds/example
```
x??

---

---
#### Storing AWS Credentials Securely Using Environment Variables
Background context: When working with AWS, it is crucial to securely store your access keys and secret keys. Storing these credentials directly in code or plain text files can expose sensitive information. Instead, environment variables provide a safer alternative by keeping the credentials out of your source code.

:p How do you set up environment variables for AWS authentication?
??x
To set up environment variables for AWS authentication, follow these steps:

1. Open your terminal.
2. Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as environment variables using the `export` command:
   ```sh
   export AWS_ACCESS_KEY_ID=(YOUR_ACCESS_KEY_ID)
   export AWS_SECRET_ACCESS_KEY=(YOUR_SECRET_ACCESS_KEY)
   ```

This method ensures that your credentials are not stored in plain text within your codebase, thereby reducing the risk of exposure.

??x
The answer with detailed explanations.
To securely authenticate to AWS using environment variables, you need to explicitly set these values before running any AWS CLI commands or Terraform scripts. This approach leverages the security principle of least privilege by requiring that each user sets their own credentials rather than having a shared secret stored in source control.

```sh
# Set up the access key and secret key as environment variables
export AWS_ACCESS_KEY_ID=(YOUR_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=(YOUR_SECRET_ACCESS_KEY)

# You can now use these environment variables to authenticate with AWS CLI commands or Terraform scripts.
```

This setup ensures that credentials are only stored in memory, providing an additional layer of security compared to storing them on disk.

---
#### Using Secret Managers for Personal Credentials
Background context: While using environment variables is a good practice, it still poses the challenge of securely managing your access keys and secret keys. Secret managers can help address this by securely storing and retrieving sensitive information, such as AWS credentials, from a centralized and encrypted repository.

:p What are some benefits of using secret managers for personal secrets?
??x
Some key benefits of using secret managers like 1Password or LastPass include:

- **Secure Storage:** Secrets are stored in an encrypted format, reducing the risk of unauthorized access.
- **Centralized Management:** You can manage multiple credentials from a single application, making it easier to rotate and revoke access keys as needed.
- **Convenience:** Tools like `op` (1Password) provide CLI interfaces that simplify the process of retrieving secrets.

:p How do you use 1Password or LastPass for AWS authentication?
??x
To use 1Password or LastPass for AWS authentication, follow these steps:

1. Install and configure the secret manager tool on your computer.
2. Store your access keys in a secure vault within 1Password or LastPass.
3. Use the CLI to retrieve and set environment variables.

For example, with 1Password, you can use the `op` command-line tool to authenticate and retrieve secrets:

```sh
# Authenticate to 1Password using the "my" profile
eval $(op signin my)

# Retrieve the AWS access key ID and set it as an environment variable
export AWS_ACCESS_KEY_ID=$(op get item 'aws-dev' --fields 'id')

# Retrieve the AWS secret access key and set it as an environment variable
export AWS_SECRET_ACCESS_KEY=$(op get item 'aws-dev' --fields 'secret')
```

This method ensures that your credentials are never exposed in plain text, providing a secure way to manage sensitive information.

??x
The answer with detailed explanations.
Using 1Password or LastPass for AWS authentication involves several steps:

1. **Authentication:** Use the `op signin` command to authenticate to 1Password or LastPass. This step ensures that your credentials are securely fetched from the vault.
2. **Retrieve and Set Environment Variables:** Use the `op get item` command to retrieve specific fields (such as the access key ID and secret access key) and set them as environment variables using the `export` command.

```sh
# Authenticate to 1Password
eval $(op signin my)

# Retrieve and set AWS credentials
export AWS_ACCESS_KEY_ID=$(op get item 'aws-dev' --fields 'id')
export AWS_SECRET_ACCESS_KEY=$(op get item 'aws-dev' --fields 'secret')
```

This approach ensures that your credentials are securely managed and only temporarily exposed to the environment, reducing the risk of exposure.

---
#### Using aws-vault for Simplified AWS Authentication
Background context: `aws-vault` is a dedicated CLI tool designed specifically for managing AWS credentials. It integrates seamlessly with 1Password or other secret managers and provides an easy way to authenticate to AWS services.

:p What are the advantages of using aws-vault over manual environment variable setup?
??x
The primary advantages of using `aws-vault` include:

- **Secure Storage:** Credentials are securely stored in your operating system’s native password manager, such as Keychain on macOS or Credential Manager on Windows.
- **Convenience:** It simplifies the process of authenticating to AWS services by leveraging a secure vault that stores and retrieves credentials automatically.
- **Fine-grained Access Control:** You can manage multiple profiles (e.g., `dev`, `prod`) and easily switch between them.

:p How do you set up aws-vault for AWS authentication?
??x
To set up `aws-vault` for AWS authentication, follow these steps:

1. Install the `aws-vault` tool.
2. Store your access keys in 1Password or another secret manager under a specific profile (e.g., `dev`).
3. Use the `aws-vault add` command to securely store credentials.

Here’s how you can set up and use `aws-vault`:

```sh
# Install aws-vault
# For macOS:
brew install aws-vault

# Add your AWS profile using 1Password as the secret manager
aws-vault add dev
Enter Access Key Id: (YOUR_ACCESS_KEY_ID)
Enter Secret Key: (YOUR_SECRET_ACCESS_KEY)

# Authenticate and execute a command
aws-vault exec dev -- terraform apply
```

This method ensures that your credentials are securely stored in a password manager, and you can easily switch between different profiles using the `aws-vault exec` command.

??x
The answer with detailed explanations.
Using `aws-vault` for AWS authentication provides several benefits:

1. **Secure Storage:** Credentials are securely stored in your operating system’s native password manager (e.g., Keychain on macOS, Credential Manager on Windows).
2. **Convenience:** The tool simplifies the process of authenticating to AWS services by automatically handling credential retrieval.
3. **Fine-grained Access Control:** You can manage multiple profiles and easily switch between them using the `aws-vault exec` command.

```sh
# Install aws-vault (for macOS)
brew install aws-vault

# Add a new profile named "dev"
aws-vault add dev
Enter Access Key Id: (YOUR_ACCESS_KEY_ID)
Enter Secret Key: (YOUR_SECRET_ACCESS_KEY)

# Authenticate and execute a Terraform apply command
aws-vault exec dev -- terraform apply
```

This setup ensures that your credentials are securely managed, and you can easily switch between different AWS profiles as needed.

---

#### Machine Users and Authentication in CI/CD Pipelines
Background context explaining that machine users, as opposed to human users, require secure authentication mechanisms for automation tasks. This is particularly important in CI/CD pipelines where automated processes need to authenticate to cloud services without storing secrets in plain text.

:p What are the challenges of authenticating a machine user in a CI/CD pipeline?
??x
The main challenge is ensuring that the machine can securely authenticate itself to another machine or service (like AWS API servers) without storing any secrets in plain text. This is crucial for maintaining security and compliance, especially when dealing with automated processes.

Example: In CircleCI, you might need to store IAM access keys as environment variables within a CircleCI context.
```shell
circleci context set my-context --key "AWS_ACCESS_KEY_ID" --value "my-access-key"
circleci context set my-context --key "AWS_SECRET_ACCESS_KEY" --value "my-secret-key"
```
x??

---

#### CircleCI with Stored Secrets
Background context explaining how CircleCI can be used to run Terraform code, and the process of using stored secrets for authentication. Highlight that creating a machine user in CircleCI involves storing its credentials as environment variables.

:p How do you set up an IAM user in AWS specifically for CircleCI automation?
??x
You create an IAM user with specific permissions needed by CircleCI for running Terraform commands. Then, copy the generated access keys and secret access key to your CircleCI context under a unique name (like `my-context`).

Example: Setting up environment variables in CircleCI.
```shell
circleci context set my-context --key "AWS_ACCESS_KEY_ID" --value "my-access-key"
circleci context set my-context --key "AWS_SECRET_ACCESS_KEY" --value "my-secret-key"
```
x??

---

#### EC2 Instance Running Jenkins as a CI/CD Server with IAM Roles
Background context explaining the use of IAM roles for EC2 instances to authenticate to AWS services. This approach involves attaching an IAM role to the instance that allows it to assume temporary credentials.

:p How does IAM role-based authentication work in an EC2 instance running Jenkins?
??x
IAM role-based authentication works by attaching a policy- and trust-policy-defined IAM role to your EC2 instance. The instance can then assume these roles to get temporary security credentials. These credentials are automatically managed by AWS, reducing the risk of exposing long-term secrets.

Example: Attaching an IAM role to an EC2 instance.
```shell
# Example using AWS CLI
aws ec2 attach-launch-template-ssm-policy \
    --instance-id i-1234567890abcdef0 \
    --policy-arn arn:aws:iam::123456789012:role/JenkinsRole
```
x??

---

#### GitHub Actions as a CI/CD Server with OIDC (OpenID Connect)
Background context explaining the integration of OpenID Connect for authentication between GitHub and AWS. This approach leverages OAuth 2.0 to securely authenticate users or services.

:p How does OIDC-based authentication work in GitHub Actions?
??x
OIDC-based authentication works by configuring an OIDC identity provider in GitHub, which issues tokens that can be used to assume IAM roles in AWS. The workflow involves setting up a trust relationship between the OIDC provider and your AWS account. When GitHub Actions trigger Terraform code execution, it uses these tokens to gain temporary access to AWS services.

Example: Setting up an OIDC connection in AWS.
```shell
# Example using AWS CLI
aws iam create-open-id-connect-provider \
    --url "https://token.actions.githubusercontent.com" \
    --tags Key=Name,Value=GitHubActions
```
x??

---

#### CircleCI Context for AWS Credentials
Background context: This section explains how to use a CircleCI Context to manage AWS credentials securely within your CI/CD pipeline. A CircleCI Context allows you to define and share secrets, such as AWS access keys, across different jobs within your `.circleci/config.yml` file.

:p How does using a CircleCI Context help in managing AWS credentials?
??x
Using a CircleCI Context helps manage AWS credentials securely by storing sensitive information like `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` outside of the main configuration files. These secrets are stored within the CircleCI environment, making them accessible to your build jobs via environment variables defined in the context.

Example: 
```yaml
workflows :  
  # Create a workflow to run the 'terraform apply' job defined above  
  deploy:    
    jobs:      
      - terraform_apply    
    # Only run this workflow on commits to the main branch    
    filters:      
      branches :        
        only:          
          - main    
    # Expose secrets in the CircleCI context as environment variables     
    context:       
      - example-context
```
x??

---

#### EC2 Instance with IAM Roles for CI/CD Authentication
Background context: In this approach, you use an EC2 instance running Jenkins (a CI/CD server) and leverage AWS Identity and Access Management (IAM) roles to authenticate Terraform code. IAM roles provide temporary credentials, which are more secure than permanent access keys.

:p How does an IAM role differ from an IAM user?
??x
An IAM role differs from an IAM user in that it is not associated with any one person and has no permanent credentials like a password or access keys. Instead, roles can be assumed by other IAM entities such as services (like EC2) to grant them temporary permissions within the AWS account.

:p What steps are involved in creating an IAM role for an EC2 instance?
??x
To create an IAM role for an EC2 instance, you need to define an `assume role policy` and then use it when creating the IAM role. Here’s a step-by-step breakdown:
1. Define the `assume_role_policy` that allows the EC2 service to assume the role.
2. Use Terraform's `aws_iam_policy_document` data source to generate JSON for the `assume_role_policy`.
3. Create an `aws_iam_role` resource and set its `assume_role_policy` attribute to use the generated policy.

Example:
```hcl
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect   = "Allow"
    actions  = ["sts:AssumeRole"]
    principals {
      type         = "Service"
      identifiers  = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name_prefix = var.name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}
```
x??

---

#### IAM Policy for EC2 Instance Role
Background context: Once you have created an IAM role, the next step is to attach one or more policies that grant specific permissions to this role. This ensures that the instance running Terraform has the necessary access to perform its tasks.

:p How do you define and attach a policy to an IAM role in Terraform?
??x
To define and attach a policy to an IAM role in Terraform, follow these steps:
1. Use `aws_iam_policy_document` data source or manually write JSON to create the policy document.
2. Attach this policy to your IAM role using the `policies` attribute of the `aws_iam_role` resource.

Example:
```hcl
data "aws_iam_policy_document" "example_policy" {
  statement {
    effect   = "Allow"
    actions  = ["ec2:DescribeInstances"]
    resources = ["*"]
  }
}

resource "aws_iam_role" "instance" {
  name_prefix = var.name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json

  # Attach the policy to the IAM role
  policies {
    name        = "example-policy"
    policy      = data.aws_iam_policy_document.example_policy.json
  }
}
```
x??

---

#### Terraform and Jenkins for EC2 Deployment
Background context: This section discusses how to use Terraform with Jenkins to deploy EC2 instances, focusing on IAM policies and instance profiles. The goal is to ensure that EC2 instances have necessary permissions while enhancing security by limiting access to metadata endpoints.

:p What is the purpose of defining an `aws_iam_policy_document` for administering EC2 Instances in Terraform?
??x
The purpose of defining an `aws_iam_policy_document` with admin permissions over EC2 Instances is to create a policy that allows all actions on EC2 resources. This policy can then be attached to an IAM role, ensuring the EC2 instance has broad administrative capabilities when running Terraform code.

```hcl
data "aws_iam_policy_document" "ec2_admin_permissions" {
   statement  {
     effect     = "Allow"
     actions    = ["ec2:*"]
     resources  = ["*"]
   }
}
```
x??

---

#### Attaching IAM Policies to EC2 Instances via Instance Profiles
Background context: Once you have the policy defined, you need to attach it to an instance profile and then assign that instance profile to your EC2 instances. This ensures that only specific OS users can access metadata endpoints.

:p How do you attach a custom IAM policy to an IAM role using Terraform?
??x
You use the `aws_iam_role_policy` resource in Terraform to attach a custom IAM policy document to an existing IAM role. The policy document is obtained from the `data "aws_iam_policy_document"` block.

```hcl
resource "aws_iam_role_policy" "example" {
   role   = aws_iam_role.instance.id
   policy  = data.aws_iam_policy_document.ec2_admin_permissions.json
}
```
x??

---

#### Creating an Instance Profile for EC2 Instances
Background context: An instance profile is a container that holds a trust relationship and one or more IAM roles. It associates these roles with the instances created in your AWS environment.

:p How do you create an instance profile to be used by your EC2 instances?
??x
To create an instance profile, you use the `aws_iam_instance_profile` resource in Terraform. This resource associates a role with an instance profile, which can then be assigned to EC2 instances.

```hcl
resource "aws_iam_instance_profile" "instance" {
   role = aws_iam_role.instance.name
}
```
x??

---

#### Disabling Instance Metadata Endpoint After Boot
Background context: By default, the instance metadata endpoint is open and accessible by all OS users. To enhance security, you can restrict access to this endpoint or disable it entirely after boot if not required.

:p How do you restrict access to the instance metadata endpoint in Linux using `iptables`?
??x
You can use `iptables` to allow only specific OS users to access the instance metadata endpoint. For example, if your app runs as user `app`, you could configure `iptables` rules to only allow this user to make HTTP requests to the metadata service.

```bash
sudo iptables -A INPUT -p tcp --dport 169254 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 169254 -m state --state NEW -m recent --update --seconds 300 --hitcount 1 -j DROP
```
x??

---

#### IAM Role Permissions During Boot and Later Access
Background context: IAM roles provide temporary credentials that can be used to authenticate with AWS services. These credentials are embedded in the instance metadata endpoint, which is accessible by processes running on the EC2 instance.

:p What happens when you run `terraform apply` on an EC2 instance using an attached IAM role?
??x
When you run `terraform apply` on an EC2 instance that has been configured to use a specific IAM role via an instance profile, Terraform will automatically use the temporary AWS credentials provided by the metadata endpoint. These credentials grant your Terraform code the necessary permissions (as defined in the IAM policy) to execute successfully.

```hcl
resource "aws_instance" "example" {
   ami           = "ami-0fb653ca2d3203ac1"
   instance_type  = "t2.micro"
   # Attach the instance profile
   iam_instance_profile  = aws_iam_instance_profile.instance.name
}
```
x??

#### GitHub Actions and OIDC for CI/CD
GitHub Actions is a managed Continuous Integration/Continuous Deployment (CI/CD) platform that can be used to run Terraform. Traditionally, workflows required manual management of credentials, but now it supports OpenID Connect (OIDC). Using OIDC, you can authenticate to cloud providers like AWS without needing to manage permanent credentials.
:p How does GitHub Actions use OIDC for CI/CD?
??x
GitHub Actions uses OIDC to establish a secure connection between the CI system and your cloud provider. Specifically, it fetches an OpenID Connect token from GitHub Actions and uses this token to authenticate with AWS services like IAM roles. This eliminates the need for manual credential management.
??x

---

#### Creating an IAM OIDC Identity Provider
To enable OIDC authentication in AWS using GitHub Actions, you first need to create an IAM OIDC identity provider. This involves specifying the URL of the OIDC provider and a list of trusted client IDs and thumbprints.
:p How do you create an IAM OIDC identity provider for GitHub Actions?
??x
You create an IAM OIDC identity provider by defining the `aws_iam_openid_connect_provider` resource in Terraform with the necessary attributes. Here’s how:
```hcl
resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[0].sha1_fingerprint]
}
```
This configuration points to the GitHub Actions OIDC provider URL and trusts the `sts.amazonaws.com` client ID.
??x

---

#### Fetching the Thumbprint for GitHub Actions
The thumbprint is crucial for establishing trust between AWS and GitHub Actions. You need to fetch this value using a data source in Terraform, which will be used in the IAM OIDC identity provider configuration.
:p How do you fetch the thumbprint for the GitHub Actions OIDC provider?
??x
You use the `tls_certificate` data source to fetch the thumbprint of the certificate issued by GitHub Actions. Here’s how it works:
```hcl
data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com"
}
```
This data source retrieves the necessary information, which is then used in the `thumbprint_list` attribute.
??x

---

#### IAM Role with OIDC for Terraform
Once you have set up the IAM OIDC identity provider, you can create an IAM role that allows specific GitHub repositories and branches to assume this role. The assumption policy is crucial for defining who can use these credentials.
:p How do you define a policy for assuming an IAM role via OIDC?
??x
You define a policy using `data "aws_iam_policy_document"` where the principal is set as the ARN of the OIDC provider and include conditions to restrict which repositories and branches are allowed to assume the role. Here’s an example:
```hcl
data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"
    principals {
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
      type        = "Federated"
    }
    condition {
      test      = "StringEquals"
      variable  = "token.actions.githubusercontent.com:sub"
      values    = [
        for a in var.allowed_repos_branches : 
        "repo:${a.org}/${a.repo}:ref:refs/heads/${a.branch}"
      ]
    }
  }
}
```
This policy ensures that only specific GitHub repositories and branches are allowed to assume the IAM role.
??x

---

#### Using Terraform with GitHub Actions
With the OIDC setup in place, you can use Terraform within a GitHub Actions workflow. This involves defining a workflow file (like `terraform.yml`) where you specify the steps for initializing and applying your Terraform configuration.
:p How do you set up a GitHub Actions workflow to run Terraform?
??x
You define a workflow file in `.github/workflows` directory, specifying the actions and their configurations. Here’s an example of a simple workflow:
```yaml
name: Terraform Apply

on:
  push:
    branches:
      - 'main'

jobs:
  TerraformApply:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        # Run Terraform using HashiCorp's setup-terraform Action
      - uses: hashicorp/setup-terraform@v1
        with:
          terraform_version : 1.1.0
          terraform_wrapper : false
        run: |
          terraform init
          terraform apply -auto-approve
```
This workflow will initialize Terraform, apply changes to your infrastructure, and automatically approve them.
??x


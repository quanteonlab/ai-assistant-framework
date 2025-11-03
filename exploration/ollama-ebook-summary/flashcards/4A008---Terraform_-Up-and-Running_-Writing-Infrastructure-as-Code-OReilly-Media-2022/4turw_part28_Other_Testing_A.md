# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 28)

**Starting Chapter:** Other Testing Approaches

---

---
#### Incremental Deployment Strategy for End-to-End Testing
Background context: The provided text discusses an incremental deployment strategy used to reduce resource usage during testing and ensure that both infrastructure changes and their deployment processes are validated. This method closely mimics actual production deployments, ensuring reliability.

:p How does the described approach help in reducing test complexity while maintaining robustness?
??x
The described approach reduces test complexity by applying only incremental changes to a persistent "test" environment. By doing so, it minimizes resource deployment needs (from several hundred to just a handful), making tests faster and less brittle. This method also ensures that the deployment process itself is validated along with infrastructure functionality.

```pseudocode
1. Apply infrastructure change incrementally to test environment.
2. Run validations using Selenium or equivalent tool from end-user perspective.
3. Validate both infrastructure correctness and deployment process effectiveness.
```
x?
---

#### End-to-End Testing Strategy Overview
Background context: The text outlines an end-to-end testing strategy where changes are applied to a running "test" environment, mimicking production deployments. This approach helps in validating both the functionality of the infrastructure and the correctness of the deployment process.

:p What is the primary advantage of using incremental changes for end-to-end tests over full apply/destroy cycles?
??x
The primary advantage of using incremental changes is that it closely mirrors real-world production practices, where changes are applied gradually rather than tearing down and rebuilding the entire environment. This method ensures that both infrastructure functionality and deployment processes are validated without the overhead of frequent full-scale deployments.

```pseudocode
1. Apply change incrementally to test env.
2. Run validations (e.g., Selenium tests).
3. Ensure both infra works correctly AND deployment process is sound.
```
x?
---

#### Static Analysis for Terraform Code
Background context: The text introduces static analysis as a fundamental way to test Terraform code by parsing and analyzing it without execution, covering tools like `terraform validate`, `tfsec`, `tflint`, and `Terrascan`.

:p What is the role of static analysis in testing Terraform code?
??x
Static analysis plays a crucial role in testing Terraform code by focusing on syntax checks and identifying potential security issues or compliance violations without running the actual infrastructure. Tools like `terraform validate` can catch syntax errors, while others such as `tfsec` and `tflint` look for security and compliance issues.

```pseudocode
1. Parse and analyze Terraform code.
2. Identify syntax errors (e.g., missing parameters).
3. Detect potential security or compliance violations.
```
x?
---

#### Static Analysis Tools
Static analysis tools run fast, are easy to use, stable, and don't require real provider authentication or resource deployment. However, they can only catch errors that can be determined from reading the code statically, such as syntax and type errors, but not dynamic business logic errors.
:p What is a key difference between static analysis tools and other policy enforcement tools?
??x
Static analysis tools can only detect issues based on static code analysis, whereas other tools like tfsec and tflint can enforce specific policies, such as security group rules and tagging conventions. This means they require the code to be executed or simulated to check for dynamic errors.
x??

---
#### Plan Testing with Terraform
Plan testing involves running `terraform plan` to analyze the potential changes without fully executing them. It checks the infrastructure state but doesn't create or modify actual resources, making it a middle ground between static analysis and full unit tests.
:p How does plan testing differ from static analysis in terms of execution?
??x
Static analysis tools execute code only by reading it statically, whereas plan testing uses `terraform plan` to simulate the actions Terraform would take without actually executing them. Plan testing can check for potential issues like resource changes but doesn't fully validate functionality or create resources.
x??

---
#### Popular Plan Testing Tools Comparison
Here is a comparison of popular tools designed for plan testing with Terraform, including their features, popularity, and use cases:
- **Terratest**: A Go library for IaC testing. It provides built-in checks for common issues but requires custom checks to be defined in Go.
- **Open Policy Agent (OPA)**: A general-purpose policy engine that can enforce policies across various cloud platforms and applications using Rego, a policy specification language.
- **HashiCorp Sentinel**: Part of the HashiCorp ecosystem, it uses Sentinel for defining policies but requires more setup compared to OPA. It's commercial with a proprietary license.
- **Checkov**: A static analysis tool that supports multiple providers like AWS, Azure, and GCP. Policies can be defined in Python or YAML and enforced during plan testing.
- **terraform-compliance**: A BDD test framework for Terraform that uses Go to define checks but requires custom policies to be written in Go.

:p Which tools support defining policies using different languages?
??x
Tools like **Open Policy Agent (OPA)**, which uses Rego, and **Checkov**, where policies can be defined in Python or YAML, allow users to define their policies using multiple programming languages. This flexibility helps in integrating policies seamlessly with existing development workflows.
x??

---
#### Example of Plan Testing Output
Here is an example snippet from a `terraform plan` command output:
```plaintext
Terraform will perform the following actions:

  # module.alb.aws_lb.example will be created
  + resource "aws_lb" "example" {
      + arn                        = (known after apply)
      + load_balancer_type         = "application"
      + name                       = "test-4Ti6CP"
      (...)

  Plan: 5 to add, 0 to change, 0 to destroy.
```
:p What does the `terraform plan` output indicate?
??x
The `terraform plan` output indicates that Terraform will create 5 new resources but won't change or destroy any existing ones. This output helps in understanding the potential changes before applying them fully.

```plaintext
Terraform will perform the following actions:

  # module.alb.aws_lb.example will be created
  + resource "aws_lb" "example" {
      + arn                        = (known after apply)
      + load_balancer_type         = "application"
      + name                       = "test-4Ti6CP"
      (...)

  Plan: 5 to add, 0 to change, 0 to destroy.
```
x??

---

#### Testing Plan Output Using Terratest
Background context explaining how to use Terratest for testing Terraform plans. The `InitAndPlan` helper runs `init` and `plan` automatically, validating syntax and API calls. Additionally, more detailed checks can be performed on the plan output using helpers like `GetResourceCount` and `InitAndPlanAndShowWithStructNoLogTempPlanFile`.
:p How does the `InitAndPlan` function in Terratest help with testing Terraform plans?
??x
The `InitAndPlan` function runs `init` and `plan` automatically, ensuring that your Terraform code can successfully run a plan to check syntax validity and API call functionality. It provides a basic validation step.
```go
func TestAlbExamplePlan(t *testing.T) {
    t.Parallel()
    albName := fmt.Sprintf("test-percents", random.UniqueId())
    opts := &terraform.Options{
        TerraformDir: "../examples/alb",
        Vars: map[string]interface{}{
            "alb_name": albName,
        },
    }
    planString := terraform.InitAndPlan(t, opts)
}
```
x??

---

#### Checking Resource Counts in Plan Output
Background context explaining how to use `GetResourceCount` to validate the number of resources added, changed, or destroyed during a Terraform plan execution.
:p How can you check the add/change/destroy counts from the plan output using Terratest?
??x
You can use the `terraform.GetResourceCount` helper function to parse the plan string and verify that the expected resource changes are made. For example:
```go
resourceCounts := terraform.GetResourceCount(t, planString)
require.Equal(t, 5, resourceCounts.Add)
require.Equal(t, 0, resourceCounts.Change)
require.Equal(t, 0, resourceCounts.Destroy)
```
This ensures that the plan output has the correct number of resources being added without any changes or destruction.
x??

---

#### Programmatic Access to Plan Output
Background context explaining how to use `InitAndPlanAndShowWithStructNoLogTempPlanFile` to parse the plan output into a struct, providing programmatic access to resource values and attributes.
:p How can you check specific values in the plan output using Terratest?
??x
You can use the `terraform.InitAndPlanAndShowWithStructNoLogTempPlanFile` helper function to get detailed information about resources planned for changes. For example:
```go
planStruct := terraform.InitAndPlanAndShowWithStructNoLogTempPlanFile(t, opts)
alb, exists := planStruct.ResourcePlannedValuesMap["module.alb.aws_lb.example"]
require.True(t, exists, "aws_lb resource must exist")
name, exists := alb.AttributeValues["name"]
require.True(t, exists, "missing name parameter")
require.Equal(t, albName, name)
```
This allows you to verify specific attributes of resources in the plan output.
x??

---

#### Open Policy Agent (OPA) for Policy Enforcement
Background context explaining how OPA can be used with Terraform for policy enforcement. The example provided enforces a tagging policy where every resource managed by Terraform must have a `ManagedBy = terraform` tag.
:p How does the OPA policy enforce tagging in Terraform resources?
??x
The OPA policy checks that each resource change has the `ManagedBy` tag set to `terraform`. If not, it sets an allow variable to true or undefined. For example:
```rego
package terraform

allow {
    resource_change := input.resource_changes[_]
    resource_change.change.after.tags["ManagedBy"] == "terraform"
}
```
This ensures that every managed resource in the Terraform codebase has the required tag.
x??

---

#### Testing a Module Without ManagedBy Tag
Background context explaining an example where a Terraform module is missing the `ManagedBy` tag, and how you might test this scenario.
:p How can you identify a module that does not set the `ManagedBy` tag in Terraform code?
??x
You can test for the absence of the `ManagedBy` tag by running a plan and checking resource changes. If no such tags are present, it indicates missing enforcement. For example:
```terraform
resource "aws_instance" "example" {
    ami           = data.aws_ami.ubuntu.id
    instance_type = "t2.micro"
}
```
This module does not set the `ManagedBy` tag and would need to be reviewed or modified.
x??

#### Terraform Plan and OPA Integration for Policy Enforcement
Background context: This concept explains how to integrate policy enforcement using Open Policy Agent (OPA) with Terraform plans. It covers the steps required to convert a Terraform plan into JSON, evaluate it against a policy written in Rego, and handle different outcomes based on the policy evaluation.
:p How can you enforce policies like tagging requirements using OPA with Terraform?
??x
To enforce policies such as tagging requirements using OPA with Terraform, follow these steps:
1. Run `terraform plan -out tfplan.binary` to generate a plan file in binary format.
2. Convert the plan file into JSON: `terraform show -json tfplan.binary > tfplan.json`.
3. Evaluate the JSON against your policy (e.g., `enforce_tagging.rego`) using OPA:
   ```sh
   opa eval \
       --data enforce_tagging.rego \
       --input tfplan.json \
       --format pretty \
       data.terraform.allow undefined
   ```
4. Based on the output, you can determine if the policy has passed or failed.

This process helps ensure that your Terraform configurations meet company-specific requirements before execution.
??x

---

#### Setting ManagedBy Tag in Terraform for Policy Enforcement
Background context: This concept details how setting specific tags (like `ManagedBy`) within a Terraform resource affects policy enforcement using OPA. It provides an example of adding the tag and re-evaluating the plan to check if the policy passes.
:p How can you modify your Terraform configuration to ensure that the ManagedBy tag is set, thus passing a policy check?
??x
To set the `ManagedBy` tag in Terraform for ensuring compliance with tagging requirements using OPA, update your resource block as follows:
```hcl
resource "aws_instance" "example" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  tags = {
    ManagedBy = "terraform"
  }
}
```
After making this change, rerun the necessary steps to evaluate your plan with OPA:
1. Run `terraform plan -out tfplan.binary`.
2. Convert the output into JSON: `terraform show -json tfplan.binary > tfplan.json`.
3. Evaluate using OPA:
   ```sh
   opa eval \
       --data enforce_tagging.rego \
       --input tfplan.json \
       --format pretty \
       data.terraform.allow true
   ```

This will now return `true`, indicating that the policy has passed.
??x

---

#### Plan Testing Tools for Terraform
Background context: This concept discusses the strengths and weaknesses of plan testing tools used in conjunction with OPA to enforce policies against Terraform plans. It highlights how these tools can be integrated into CI/CD pipelines.
:p What are some key benefits and limitations of using plan testing tools like OPA for policy enforcement?
??x
Key benefits of plan testing tools include:
- They run fast, faster than unit or integration tests but not as fast as pure static analysis.
- They are somewhat easy to use, easier than unit or integration tests but not as simple as pure static analysis.
- They are stable with few flaky tests, more stable than unit or integration tests.

Limitations include:
- They can catch a wider range of errors compared to static analysis but fewer than full integration tests.
- Real provider authentication is required for plan validation.
- They do not check functionality; passing checks does not guarantee that the infrastructure works correctly.

These tools are particularly useful in CI/CD pipelines where policies need to be continuously enforced.
??x

---

#### Server Testing Tools for Terraform
Background context: This concept introduces server testing tools, which focus on validating the configuration and operation of servers launched by Terraform. It provides a brief overview of these tools' popularity and maturity based on GitHub statistics from February 2022.
:p What are some key aspects to consider when using server testing tools with Terraform?
??x
Server testing tools for Terraform help validate that the servers launched meet specific configuration requirements. Key aspects include:

- Popularity: The tools have varying levels of popularity, as indicated by GitHub stats.
- Maturity: Some tools may be more mature and stable than others.

These tools are useful for ensuring that servers configured via Terraform are correctly set up and functioning according to the desired state.
??x

---
#### InSpec Overview
InSpec is a framework for auditing and testing infrastructure. It provides a way to define policies that can be run against your deployed servers to ensure they meet certain criteria.

:p What is InSpec, and what does it do?
??x
InSpec is an auditing and testing framework that allows you to define policies to validate the configuration of your servers. You can use it to check if files exist, permissions are set correctly, services are running, and more. The goal is to ensure that infrastructure as code (IAC) matches what is deployed in production.

```ruby
# Example InSpec code to check for a file existence and its mode
describe file('/etc/myapp.conf') do
  it { should exist }
  its('mode') { should cmp 0644 }
end
```
x??

---
#### Serverspec Overview
Serverspec is an RSpec-based tool that tests servers by writing server-specific checks. It's designed to be simple and easy to use, making it ideal for quick validation of server configurations.

:p What is Serverspec, and how does it work?
??x
Serverspec uses RSpec syntax to write tests that can validate the state of a server. These tests are typically written in Ruby and follow the same pattern as RSpec tests. You can check if files exist, services are running, or any other properties of the server.

```ruby
# Example Serverspec code to check for Apache service status
describe package('apache2') do
  it { should be_installed }
end

describe service('apache2') do
  it { should be_enabled }
  it { should be_running }
end
```
x??

---
#### Goss Overview
Goss is a fast, simple, and lightweight tool for server validation. It uses YAML to define checks that can be run against servers.

:p What is Goss, and what makes it unique?
??x
Goss is a server validation tool that focuses on simplicity and speed. It allows you to write checks in YAML, making it easy to use even for non-Ruby developers. This makes it ideal for small projects or teams where ease of use is crucial.

```yaml
# Example Goss configuration in YAML
checks:
  - check: file /etc/myapp.conf exists
  - check: port 8080 open
```
x??

---
#### Comparing InSpec, Serverspec, and Goss
These tools provide a way to validate server configurations but differ in their approach, ease of use, and backing companies. InSpec is part of the Chef ecosystem, while Serverspec uses RSpec syntax, and Goss uses YAML.

:p What are the main differences between InSpec, Serverspec, and Goss?
??x
- **InSpec**: Part of the Chef ecosystem, uses a Ruby-based DSL, and is great for complex policy checks.
- **Serverspec**: Uses RSpec tests in Ruby, making it familiar to developers who use RSpec.
- **Goss**: Focuses on simplicity and speed, using YAML for configuration, which makes it easy for non-Ruby developers.

These tools are all effective but cater to different needs. InSpec is good for complex policies and compliance checks, Serverspec is ideal for those comfortable with Ruby and RSpec, while Goss is perfect for quick, simple validations.
x??

---

#### Weaknesses of Server Testing Tools
Server testing tools have several limitations that make them less than ideal for comprehensive infrastructure testing. They are slow due to needing a full apply cycle, which includes deploying servers and waiting for deployment completion. This can lead to flaky tests because real-world issues may arise during this process. Additionally, these tools require authentication to a real provider (such as AWS) and necessitate the deployment/undeployment of actual resources, which incurs both time and cost.

:p What are some key weaknesses of server testing tools?
??x
The key weaknesses include slowness due to needing a full apply cycle with real servers, flakiness from intermittent issues during real-world deployments, and the necessity for authentication and resource deployment/undeployment which can be costly. 
```pseudocode
# Example pseudocode to illustrate the process of server testing
def testRealServers():
    # Apply changes to infrastructure
    applyChanges()
    # Wait for servers to deploy
    waitForDeploymentCompletion()
    # Run tests on deployed servers
    runTestsOnDeployedServers()
```
x??

---

#### Infrastructure Code Rots Quickly Without Tests
Infrastructure code without automated tests quickly rots, meaning it becomes unreliable and harder to maintain. Manual testing and reviews help initially but eventually fail to catch all bugs. Automated tests are essential because they catch nontrivial bugs that manual testing might miss.

:p Why is infrastructure code with no automated tests considered broken?
??x
Infrastructure code without automated tests is considered broken because real-world changes and evolving tooling can introduce many nontrivial bugs, which manual tests might not detect. Automated tests help in identifying these issues early and maintaining the reliability of the infrastructure.

```pseudocode
# Example pseudocode to illustrate adding an automated test
def addAutomatedTest():
    # Write a test that checks server functionality
    writeFunctionalityTests()
    # Run tests after every commit
    runTestsAfterEveryCommit()
```
x??

---

#### Testing Terraform Code Requires Real Resources
When testing Terraform code, you cannot use localhost for manual or automated testing. This means all testing must involve deploying real resources in isolated sandbox environments to ensure accurate and thorough checks.

:p Why can't you use localhost when testing Terraform code?
??x
You cannot use localhost when testing Terraform code because the environment needs to be a true representation of the production setup, including network configurations and other dependencies. Deploying real resources into isolated sandbox environments ensures that tests accurately reflect how the infrastructure will operate in a live scenario.

```pseudocode
# Example pseudocode for deploying resources for testing
def deployResourcesForTesting():
    # Apply Terraform configuration to create test environment
    applyTerraformConfiguration()
    # Run integration or end-to-end tests on deployed resources
    runIntegrationTestsOnDeployedResources()
```
x??

---

#### Importance of Smaller Modules in Testing
Smaller modules are easier and faster to test because they have fewer moving parts, making it simpler to identify issues. Larger monolithic modules can be complex and harder to manage.

:p Why are smaller modules better for testing?
??x
Smaller modules are better for testing because they contain fewer components, which makes them easier to understand, maintain, and debug. This reduces the complexity of tests and speeds up the development process by allowing quick iterations and validation.

```pseudocode
# Example pseudocode illustrating how to create a smaller module
def createSmallerModule():
    # Define a small, focused Terraform configuration file
    writeSmallTerraformConfig()
    # Write corresponding unit or integration tests for this config
    writeTestsForSmallConfig()
```
x??

---

#### Types of Testing for Infrastructure Code
Infrastructure code can be tested using various approaches such as static analysis, unit testing, integration testing, and end-to-end testing. Each type has its strengths and weaknesses.

:p What are the different types of tests mentioned in the text?
??x
The different types of tests mentioned include:
- Static analysis: Checks syntax and policies.
- Unit tests: Focus on small parts of the codebase.
- Integration tests: Test how components interact with each other.
- End-to-end tests: Validate the entire system from start to finish.

These tests have varying levels of speed, cost, stability, ease of use, and their ability to check different aspects of infrastructure functionality. A mix of all these types is recommended for comprehensive testing.

```pseudocode
# Example pseudocode for setting up a test environment
def setupTestEnvironment():
    # Initialize static analysis tools
    initializeStaticAnalysis()
    # Write unit tests for Terraform configurations
    writeUnitTestsForTerraformConfig()
    # Set up integration and end-to-end tests
    setUpIntegrationAndEndToEndTests()
```
x??

#### Adopting Infrastructure as Code in Your Team
Background context: In the real world, you will likely work within a team that needs to adopt Terraform and IaC tools. Convincing your team of its benefits is crucial for successful implementation.

:p How do you convince your team to use Terraform and other infrastructure-as-code (IaC) tools?
??x
To convince the team, highlight the benefits such as improved reliability, reproducibility, version control, and easier collaboration. Emphasize how IaC can lead to more maintainable and consistent infrastructure.

Example scenario:
- Improving the onboarding process by ensuring new developers have a uniform environment.
- Reducing operational costs through automation and error reduction in manual processes.
x??

---

#### Workflow for Deploying Application Code
Background context: When your team is working with application code, you need to establish a workflow that integrates with Terraform. This involves setting up CI/CD pipelines or manual deployment processes.

:p What are the key steps in creating a workflow for deploying application code using Terraform?
??x
Key steps include:
1. Writing application code.
2. Using `terraform apply` to update infrastructure as needed.
3. Integrating with CI/CD tools (e.g., Jenkins, GitHub Actions) to automate deployments.
4. Testing the deployed application in a staging environment before going live.

Example of integrating with GitHub Actions:
```yaml
name: Terraform and Application Deployment

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2
    - name: Initialize Terraform
      run: |
        terraform init
    - name: Plan changes
      run: |
        terraform plan -out=tfplan
    - name: Apply changes
      run: |
        terraform apply tfplan --auto-approve
```
x??

---

#### Workflow for Deploying Infrastructure Code
Background context: When deploying infrastructure code, you need a workflow that allows multiple team members to understand and modify Terraform scripts safely. This involves version control systems like Git.

:p What are the steps involved in setting up a workflow for deploying infrastructure code?
??x
Steps include:
1. Setting up Terraform configurations.
2. Using Git for version control of these configurations.
3. Creating branches for new features or modifications.
4. Merging changes into main branches after thorough testing.
5. Running `terraform apply` to update the environment.

Example Git branch strategy:
```bash
git checkout -b feature/new-vpc main
# Make necessary Terraform changes
git add .
git commit -m "Add a new VPC"
git push origin feature/new-vpc
```
x??

---

#### Putting It All Together
Background context: Integrating all the above workflows ensures that both application code and infrastructure are managed effectively. This involves aligning with existing tech stacks, integrating CI/CD tools, and maintaining best practices.

:p How do you integrate all the workflows for deploying both application code and infrastructure?
??x
Integration involves:
1. Aligning Terraform configurations with Git repositories.
2. Setting up CI/CD pipelines to automate Terraform deployments.
3. Maintaining a clear separation of concerns between application code and infrastructure.
4. Regularly reviewing and updating processes based on feedback.

Example integration in Jenkins:
```groovy
pipeline {
    agent any
    stages {
        stage('Initialize') { 
            steps { script { 
                sh 'terraform init' 
            }}
        }
        stage('Plan') {
            steps { script { 
                sh 'terraform plan -out=tfplan'
            }}
        }
        stage('Apply') {
            steps { script { 
                sh 'terraform apply tfplan --auto-approve'
            }}
        }
    }
}
```
x??

---


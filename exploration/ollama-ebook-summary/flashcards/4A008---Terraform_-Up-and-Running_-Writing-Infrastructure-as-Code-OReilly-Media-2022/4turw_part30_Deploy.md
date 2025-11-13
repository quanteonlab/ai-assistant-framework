# Flashcards: 4A008---Terraform_-Up-and-Running_-Writing-Infrastructure-as-Code-OReilly-Media-2022_processed (Part 30)

**Starting Chapter:** Deploy

---

#### Running Code Locally
Background context: After checking out a new feature branch, you can run application code locally on your computer. This allows you to test changes and ensure that they work as intended before deploying them.

:p How do you start running the Ruby web server example from Chapter 9?
??x
To start running the Ruby web server example, you need to navigate to the correct directory and then execute the `web-server.rb` file using the Ruby interpreter. Here's an example command:

```bash
$cd code/ruby/10-terraform/team$ ruby web-server.rb
```

This command starts a simple HTTP server that listens on port 8000. You can access it via `http://localhost:8000`.

You can use tools like `curl` to test the output of the server:

```bash
$curl http://localhost:8000
Hello, World
```

Additionally, you can run automated tests by executing another Ruby script:

```bash$ ruby web-server-test.rb
```

The output shows that all tests passed successfully.

x??

---

#### Making Code Changes
Background context: After verifying the application code works as expected locally, you can start making changes to the code. This process involves making a change, running manual or automated tests to ensure everything still works correctly, and repeating these steps iteratively until the desired functionality is achieved.

:p How do you make changes and verify them in the Ruby web server example?
??x
To make changes and verify them in the Ruby web server example, follow these steps:

1. **Make a Change:** Modify the `web-server.rb` file as needed. For instance, change the output message from "Hello, World" to "Hello, World v2":
    ```ruby
    # Before:
    # puts "Hello, World"
    
    # After:
    puts "Hello, World v2"
    ```

2. **Restart the Server:** After making changes, restart the server by stopping it and starting it again.

3. **Test the Change:** Use `curl` to test the new output of the server:
    ```bash
    $curl http://localhost:8000
    Hello, World v2
    ```

4. **Run Automated Tests (Optional):** If you have automated tests for your application, run them to ensure that no other functionality was affected by the change.

:p How do you run a simple HTTP server using Ruby?
??x
To run a simple HTTP server using Ruby, follow these steps:

1. Navigate to the directory containing the `web-server.rb` file:
    ```bash$ cd code/ruby/10-terraform/team
    ```

2. Start the server by running the script with the Ruby interpreter:
    ```bash
    $ruby web-server.rb
    ```

The output should indicate that the server is up and running on port 8000.

:p How do you test a locally run HTTP server using `curl`?
??x
To test a locally run HTTP server using `curl`, use the following command:
```bash$ curl http://localhost:8000
```

This command sends an HTTP GET request to the local server running on port 8000 and displays the response, which should be "Hello, World" by default.

:p How do you run automated tests for a Ruby application?
??x
To run automated tests for a Ruby application, use the `ruby` command followed by the path to the test file. For example:
```bash
$ruby web-server-test.rb
```

The output will indicate whether all tests passed or if any failed.

:x??

---

#### Committing Code Changes
Background context: After making and testing changes locally, you should commit your code with a clear message explaining what was changed. This helps maintain the history of changes made to the project.

:p How do you commit local code changes?
??x
To commit local code changes, use the `git commit` command followed by an appropriate message that describes the changes. For example:
```bash$ git commit -m "Updated Hello, World text"
```

This command commits all staged changes to your current branch with a clear commit message.

:p How do you push your feature branch back to GitHub?
??x
To push your feature branch back to GitHub, use the `git push` command followed by the remote repository and the name of the branch. For example:
```bash
$ git push origin example-feature
```

This command pushes all changes in the current branch (`example-feature`) to the remote repository on GitHub.

:p What happens after pushing your feature branch back to GitHub?
??x
After pushing your feature branch back to GitHub, you can create a pull request from the web interface. The log output will contain a URL that you can visit:
```bash
remote: Create a pull request for 'example-feature' on GitHub by visiting:
      https://github.com/<OWNER>/<REPO>/pull/new/example-feature
```

You can open this URL in your browser and fill out the pull request title and description, then click "Create" to submit the changes for review.

:x??

---

#### Setting Up Commit Hooks for Automated Tests
Continuous integration (CI) servers like Jenkins, CircleCI, or GitHub Actions can be used to set up commit hooks that run automated tests on every push. This ensures code quality and reduces bugs before merging into the main branch.

:p How do you integrate automated testing with a CI server in a version control system?
??x
To integrate automated testing with a CI server such as CircleCI, you configure the server to automatically execute test scripts whenever a new commit is pushed. For example, on GitHub Actions, you add a `.github/workflows` file to your repository that defines jobs and steps for running tests.

For instance, in a CircleCI configuration:
```yaml
version: 2.1
jobs:
  build:
    docker:
      - image: circleci/node:latest
    steps:
      - run: npm install
      - run: npm test

workflows:
  version: 2
  build-and-test:
    jobs:
      - build
```
x??

---

#### Running Automated Tests in GitHub Pull Requests
Automated tests can be run and their results displayed directly in the pull request. This helps developers review code changes more effectively.

:p How do automated test results appear in a GitHub pull request?
??x
Automated test results are shown in the pull request itself when using CI servers integrated with platforms like GitHub. For example, if you use CircleCI, it can run unit tests, integration tests, end-to-end tests, and static analysis checks against the code changes.

In the GitHub interface, a badge or a section will display the status of these tests:
- Green: Tests passed
- Red: Tests failed

This information helps reviewers quickly assess the state of the pull request.
??x
Example output in the GitHub pull request could look like this:
```
![CircleCI](https://circleci.com/gh/user/repo/tree/main)
<img src="https://example.com/circleci.png" alt="Test Results">
```

The output integrates seamlessly with the pull request, providing instant feedback to the reviewers.
x??

---

#### Immutable Infrastructure Practices for Releasing Code
Using immutable infrastructure practices ensures that once a code artifact is created, it cannot be changed. Each release should have a unique version number or tag.

:p How do you ensure your application code follows immutable infrastructure practices?
??x
To follow immutable infrastructure practices, you package the new version of the application into an unchangeable artifact with a unique identifier. For Docker images, this can be done by tagging the image with a commit ID or a custom version number.

For example:
```bash
# Using commit hash as tag
commit_id=$(git rev-parse HEAD)
docker build -t brikis98/ruby-web-server:$commit_id .

# Alternatively using semantic versioning
git tag -a "v0.0.4" -m "Update Hello, World text"
git push --follow-tags
```

This ensures that the deployed artifact is immutable and can be traced back to its exact codebase.
??x

Example Docker build command:
```bash
commit_id=$(git rev-parse HEAD)
docker build -t brikis98/ruby-web-server:$commit_id .
```
And pushing a tag:
```bash
git tag -a "v0.0.4" -m "Update Hello, World text"
git push --follow-tags
```

These commands help maintain the integrity of the deployed application by ensuring that each version is unique and unchangeable.
x??

---

#### Tagging Git Commits for Readability
Git tags can be used to give human-readable names to specific commits, making it easier to identify code versions.

:p How do you create a Git tag?
??x
To create a Git tag, you use the `git tag` command followed by an optional message. For example:
```bash
git tag -a "v0.0.4" -m "Update Hello, World text"
```
This creates a new annotated tag named `v0.0.4` with the commit message: "Update Hello, World text". 

After tagging, you need to push the tag to the remote repository:
```bash
git push --follow-tags
```

Using tags makes it easier for developers and operations teams to reference specific versions of code.
??x

Example command sequence:
```bash
git tag -a "v0.0.4" -m "Update Hello, World text"
git push --follow-tags
```
This will create a named version that can be pushed to the remote repository for tracking and referencing.
x??

---

#### Git Tagging and Docker Deployment

Background context: When deploying application code, it is crucial to version your artifacts. Using Git tags can help you manage different versions of your application effectively. The provided snippet demonstrates how to create a tag from the current commit hash using `git describe --tags` and apply this tag in a Docker build process.

:p How do you use Git tags for versioning in a Docker image?
??x
To use Git tags for versioning in a Docker image, first, generate a tag based on the latest commit hash using `git describe --tags`. Then, use this tag to label your Docker image during the build process. This helps in tracking specific versions of your application and facilitates easy debugging by allowing you to check out the code at a particular tag.

```bash
# Generate Git tag from current commit
git_tag=$(git describe --tags)

# Build Docker image with the generated tag
docker build -t brikis98/ruby-web-server:$git_tag .
```
x??

---

#### Deployment Tooling

Background context: The choice of deployment tool depends on how you package and run your application, your infrastructure architecture, and the desired deployment strategies. This section discusses different tools such as Terraform for managing infrastructure-as-code and orchestration tools like Kubernetes.

:p What are some examples of deployment tooling mentioned in the text?
??x
Some examples of deployment tooling mentioned include:

- **Terraform**: A tool used to manage infrastructure as code, allowing you to define infrastructure resources using configuration files. It can be used for deploying applications by updating parameters and running `terraform apply`.

- **Orchestration tools**: These include Kubernetes (Docker orchestration), Amazon ECS, HashiCorp Nomad, and Apache Mesos, which are designed to deploy and manage application containers.

- **Scripts**: Custom scripts might be necessary when more complex requirements cannot be met by Terraform or other orchestration tools.
x??

---

#### Deployment Strategies

Background context: Different deployment strategies can be employed based on your application's needs. The text outlines three common strategies: rolling deployment with replacement, rolling deployment without replacement, and blue-green deployment.

:p What is a rolling deployment with replacement?
??x
A rolling deployment with replacement involves gradually replacing old copies of the application with new ones while ensuring that at least one version of the application remains available to users. This method ensures zero downtime by replacing one instance at a time after it passes health checks and starts receiving live traffic.

During this process, both the old and new versions coexist temporarily, allowing for monitoring and rollbacks if necessary.
x??

---

#### Blue-Green Deployment

Background context: Blue-green deployment is another strategy where you deploy two identical sets of application instances (blue and green) in parallel. Traffic can be switched to either set based on health checks.

:p What does blue-green deployment involve?
??x
Blue-green deployment involves deploying an identical copy (green) of the existing application (blue). After both copies are healthy, all traffic is redirected to the new version (green), and the old version (blue) is undeployed. This method ensures that users always see a consistent version of the application.

This strategy is useful in environments with flexible capacity where multiple instances can be managed without downtime.
x??

---

#### Deployment Server

Background context: The deployment server plays a role in managing the rollout process, but it's not explicitly discussed in detail within this snippet. Typically, a deployment server would handle tasks like rolling updates, health checks, and traffic redirection.

:p What is the role of a deployment server?
??x
The role of a deployment server is to manage the application deployment process, including tasks such as:

- Rolling out new versions of applications to servers.
- Monitoring the health of deployed instances.
- Redirecting traffic between different versions of an application.
- Managing rollback processes if necessary.

While not explicitly detailed in the provided text, these functions are crucial for smooth and efficient deployments.
x??

---

#### Canary Deployment Overview
In software development, a canary deployment is a method used to test new versions of an application in production before rolling it out fully. The process involves deploying a small number of instances (the "canary") and comparing them against existing instances ("control"). This helps identify potential issues early on.
:p What is the purpose of a canary deployment?
??x
The primary purpose of a canary deployment is to test new versions of an application in production without affecting all users. It allows you to compare the behavior of the new version (canary) against the existing version (control). If no issues are found, the full rollout can proceed; otherwise, you can revert changes or troubleshoot.
x??

---
#### Difference Between Canary and Control
During a canary deployment, the "canary" is one or more instances of an application that have been newly deployed. These instances run alongside existing ("control") instances to allow for detailed comparisons. The goal is to ensure both versions perform similarly in all relevant metrics.
:p What are some dimensions used to compare canaries and controls?
??x
Dimensions commonly compared include CPU usage, memory usage, latency, throughput, error rates in logs, HTTP response codes, etc. These metrics help identify any discrepancies between the new ("canary") and existing ("control") versions of the application.
x??

---
#### Rolling Deployment Strategies
After a canary deployment confirms no issues with the new version, you can proceed with a rolling deployment. This strategy gradually deploys the updated code to all instances, allowing time for monitoring and ensuring smooth operation.
:p What are some common rolling deployment strategies?
??x
Common rolling deployment strategies include:
- Gradual increase: Deploying the update to 10% of users initially, then incrementally increasing that percentage over time.
- Canary groups: Similar to canary deployments but applied across multiple servers or environments.
- Blue-green deployment: Using two identical sets of infrastructure; switch traffic from old (blue) to new (green) without downtime.
x??

---
#### Feature Toggles
Feature toggles, also known as feature flags, are mechanisms that allow developers to turn features on and off at runtime. They help in safely deploying new features by controlling access based on user or environment.
:p How do feature toggles work?
??x
Feature toggles work by wrapping the implementation of a new feature inside an if-statement. By default, this statement is set to false, meaning the feature is disabled until manually enabled through configuration. This allows for controlled rollout and easy rollback in case of issues.
x??

---
#### CI Server Deployment
Deploying from a CI (Continuous Integration) server ensures that all deployment steps are automated. This practice promotes consistency, reduces human error, and streamlines the process by capturing workflows as scripts or commands.
:p Why is it important to run deployments from a CI server?
??x
Running deployments from a CI server is crucial because it forces the automation of all deployment processes. Automation ensures consistent behavior across multiple environments, reduces potential human errors, and makes the entire workflow more predictable and repeatable.
x??

---

---
#### Consistent Environment Across Developer Machines
Background context: When developers run deployments from their own computers, it can lead to inconsistencies due to different operating systems, dependency versions, configurations, and deployed changes. Using a CI server ensures that all environments are consistent.

:p What is the problem with running deployments from developer machines?
??x
The main problems include differences in operating systems, dependency versions (such as Terraform), configurations, and what's actually being deployed (e.g., accidentally deploying uncommitted changes). By using a CI server, these issues can be eliminated.
x??

---
#### Better Permissions Management Through CI Server
Background context: Giving developers direct access to the production environment can lead to security risks. Using a single CI server for deployments simplifies permission management and enforces good security practices.

:p How does using a CI server help with permissions management?
??x
Using a CI server helps by centralizing deployment permissions, making it easier to enforce security policies since only the CI server has access to production environments. This reduces the risk of unauthorized changes or accidental deployments.
x??

---
#### Promotion Across Environments in Immutable Infrastructure
Background context: In immutable infrastructure practices, rolling out new versions involves promoting the exact same versioned artifact from one environment to another (e.g., dev to staging and then to prod). This ensures consistency across environments.

:p What is the process for promoting an artifact through different environments?
??x
The process typically includes:
1. Deploying the artifact in the first environment (e.g., dev).
2. Running tests (manual and automated) in that environment.
3. If successful, deploying to the next environment (staging), then running tests there as well.
4. Finally, promoting it to production only if everything works correctly.

This ensures consistency because the exact same version of the artifact is used across environments.
x??

---
#### Workflow for Deploying Infrastructure Code
Background context: The workflow for deploying infrastructure code mirrors that of application code but with additional complexities due to the nature of infrastructure changes. Version control, testing, and deployment are key steps.

:p What are the main steps in the workflow for deploying infrastructure code?
??x
The main steps include:
1. Using version control.
2. Running the code locally.
3. Making code changes.
4. Submitting changes for review.
5. Running automated tests.
6. Merging and releasing the changes.
7. Deploying the infrastructure code.

These steps ensure that changes are thoroughly tested before being deployed, maintaining consistency across environments.
x??

---

#### Live Repo and Modules Repo
Live repo and modules repo are separate version control repositories used for managing Terraform infrastructure code. Typically, one is dedicated to reusable, versioned modules, while another focuses on live infrastructure deployments.

The repository for **modules** contains reusable components that can be shared across projects. These include:
- `cluster/asg-rolling-deploy`
- `data-stores/mysql`
- `networking/alb`
- `services/hello-world-app`

The **live infrastructure repo** defines the actual deployed infrastructure in different environments like dev, stage, and prod.

This separation helps maintain consistency across projects and simplifies maintenance by centralizing common modules.
:p What is the purpose of having separate repositories for Terraform modules and live infrastructure?
??x
Having separate repositories allows for better management and reusability of code. The module repository contains reusable, versioned components that can be shared across multiple projects. Meanwhile, the live infrastructure repository defines the actual deployed configurations in various environments (e.g., dev, stage, prod). This separation ensures consistency and makes maintenance easier.
x??

---

#### Golden Rule of Terraform
The Golden Rule of Terraform is a way to ensure your Terraform code accurately reflects the current state of your deployed infrastructure. To check this rule:
1. Go into the live repository.
2. Randomly select several folders.
3. Run `terraform plan` in each one.

If `terraform plan` always outputs "no changes," then your infrastructure code matches what's actually deployed, indicating everything is up to date and consistent.

:p How can you verify that your Terraform code accurately reflects the current state of your deployed infrastructure?
??x
You can verify this by running `terraform plan` in multiple folders within the live repository. If `terraform plan` consistently shows "no changes," it means your code matches the actual deployment, ensuring consistency and accuracy.
x??

---

#### The Trouble with Branches
Branching in Terraform can lead to issues due to the nature of infrastructure as code (IaC). Each branch may have slightly different configurations that are hard to reconcile when merging back to master.

:p What are the challenges associated with branching in Terraform?
??x
Branching in Terraform can cause inconsistencies because each branch might have its own set of configuration changes. Merging branches back into the main codebase requires careful management to avoid conflicts and ensure consistency across environments.
x??

---

#### Workflow for Deploying Infrastructure Code
A recommended workflow involves having a specialized infrastructure team that builds reusable, robust modules. These teams focus on creating production-grade components that implement best practices like:
- Composable API
- Thorough documentation (including examples)
- Automated tests
- Version control
- Implementation of security, compliance, scalability, high availability, and monitoring requirements

These modules can be shared across the organization as a service catalog, allowing other teams to deploy and manage their infrastructure independently.

:p What is the recommended workflow for deploying infrastructure code?
??x
The recommended workflow includes having an infrastructure team that builds reusable, production-grade modules. These modules are documented, tested, and versioned, ensuring they meet all company requirements (security, compliance, scalability, etc.). Other teams can consume these modules like a service catalog to deploy their infrastructure independently while maintaining consistency.
x??

---

#### The Golden Rule of Terraform: Main Branch Representation
Background context explaining this concept. According to the golden rule, the main branch of the live repository should be a 1:1 representation of what's actually deployed in production. This means that every resource deployed should have corresponding code checked into the live repo.

:p What does "what’s actually deployed" mean in the context of The Golden Rule of Terraform?
??x
"what’s actually deployed" refers to the actual infrastructure resources that are running and in use, which may differ from what is recorded in your version control system if out-of-band changes have been made. This can lead to discrepancies between the live environment and the code in the repository.
x??

---
#### Avoiding Out-of-Band Changes
This concept explains why making manual or out-of-band changes to deployed infrastructure bypasses the version control process, leading to potential inconsistencies.

:p Why should you avoid making out-of-band changes?
??x
Making out-of-band changes can introduce bugs and break the 1:1 relationship between your code and deployed resources. This undermines the benefits of using Infrastructure as Code (IaC) because any changes made outside the version control system cannot be tracked, reverted, or managed effectively.
x??

---
#### Using Terraform Workspaces vs. Separate Environment Folders
This concept differentiates between using workspaces to manage environments and maintaining separate folders for each environment.

:p Why should you avoid using workspaces to manage environments?
??x
Using workspaces can make it difficult to determine what resources are deployed in which environments just by looking at the codebase, because there is only one copy of the main codebase even though multiple environments exist. This can lead to confusion and maintenance issues.
x??

---
#### The Main Branch as a Single Source of Truth
This concept explains why the main branch should be the single source of truth for production deployments.

:p Why does the main branch need to represent what's actually deployed?
??x
The main branch serves as the authoritative version of your infrastructure code. By ensuring that it is an up-to-date and accurate representation, you can avoid confusion, make changes more predictable, and simplify maintenance.
x??

---
#### Workflow for Deploying Infrastructure Code with Locking Mechanisms
This concept discusses the limitations of locking mechanisms provided by Terraform backends.

:p What are the limitations of using Terraform backend locking?
??x
Terraform backend locking only helps prevent overwriting state changes but cannot prevent conflicts in code when two team members are deploying from different branches. This means that even with locking, changes to infrastructure code can still conflict, leading to potential issues.
x??

---
#### Example Scenario: Instance Type Change vs. Tag Addition
This concept provides a concrete example of how out-of-band changes and concurrent modifications can lead to conflicts.

:p What is the example illustrating?
??x
The example illustrates how concurrent modifications by different team members to the same infrastructure code, but in different branches, can lead to conflicts even with locking mechanisms in place. Specifically, Anna’s change from t2.micro to t2.medium conflicts with Bill’s addition of a tag, highlighting the need for careful coordination and management.
x??

---

#### Context of Multiple Branches and Terraform Deployment Conflicts
Background context explaining the scenario where Anna's changes are deployed to staging, but Bill is still using t2.micro. This highlights how branching can lead to conflicts when deploying to shared environments without proper coordination or branch management.
:p How can branching in Terraform lead to deployment conflicts?
??x
Branching in Terraform can lead to deployment conflicts because even though Anna’s changes are on a different branch and deployed to staging, Bill is still using the old configuration (t2.micro). This situation could result in inconsistent states across branches if not managed properly. 
For example:
```terraform
resource "aws_instance" "foo" {
  ami                          = "ami-0fb653ca2d3203ac1"
  instance_type                = "t2.medium" # Anna's change
  tags                         = { Name = "foo" }
}
```
When Bill runs `terraform plan`, he might miss the update to t2.micro, and if not caught, could deploy the old configuration.
x??

---

#### Importance of Consistent Branching for Shared Environments
Background context explaining why consistent branching is crucial for shared environments like staging or production. The example illustrates how Terraform’s implicit mapping from code to infrastructure means only one branch should be used for shared environments.
:p Why is it important to use a single branch for shared environments in Terraform?
??x
Using a single branch for shared environments like staging and production ensures that everyone is working with the same configuration, reducing the risk of conflicts and ensuring consistent states across deployments. For example:
```terraform
# Assuming this is part of the .gitignore or branching strategy to restrict direct changes
# Staging environment should only be modified through a specific branch
```
If multiple branches are used, changes might get out of sync, leading to issues like Bill undeploying Anna's instance type change.
x??

---

#### Manual Testing Basics for Terraform Code
Background context explaining that manual testing in Terraform requires using sandbox environments such as dedicated AWS accounts. This is different from local development due to the nature of deploying infrastructure.
:p How do you manually test Terraform code?
??x
You manually test Terraform code by running `terraform apply` in a sandbox environment, such as an AWS account dedicated for developers or each developer. For instance:
```bash
$terraform apply
Apply complete. Resources: 5 added, 0 changed, 0 destroyed.
Outputs:
alb_dns_name = "hello-world-stage-477699288.us-east-2.elb.amazonaws.com"
```
After applying the changes, you can verify the infrastructure works as expected using tools like `curl` to ensure everything is functioning correctly.
x??

---

#### Workflow for Deploying Infrastructure Code
Background context explaining the iterative process of making code changes, running `terraform apply`, and testing with commands like `curl` or automated tests. The goal is to get feedback quickly and iteratively improve the infrastructure.
:p What are the steps in deploying infrastructure code using Terraform?
??x
The steps for deploying infrastructure code using Terraform include:
1. **Iterative Changes**: Make changes in your code, typically within a sandbox environment.
2. **Apply Changes**: Run `terraform apply` to deploy those changes.
3. **Manual Testing**: Use tools like `curl` or other tests to verify the deployed infrastructure works as expected.
   For example:
   ```bash$ curl hello-world-stage-477699288.us-east-2.elb.amazonaws.com 
   Hello, World v2
   ```
4. **Automated Testing**: Run automated tests with `go test` to ensure the changes don’t break anything.
5. **Commit Changes**: Regularly commit your work to version control.
x??

---

#### Code Review and Clean Coding Practices for Terraform
Background context explaining why code reviews and clean coding practices are essential for writing maintainable, understandable infrastructure code. The example includes documentation requirements like READMEs, API docs, and design documents.
:p What are the key guidelines for clean coding in Terraform?
??x
Key guidelines for clean coding in Terraform include:
- **Documentation**: Write clear READMEs that explain what the module does, how to use it, and how to modify it. Also, include tutorials, API documentation, and design documents.
- **Automated Tests**: Ensure all changes are tested both manually (e.g., `terraform apply`, `curl`) and through automated tests (`go test`).
- **File Layout**: Define conventions for file storage layout that help in providing isolation guarantees between environments.
- **Code Documentation**: Use comments to provide context beyond the code, and leverage Terraform’s description parameters for inputs/outputs.
x??

---

#### Isolation via File Layout
Background context explaining how the file layout of Terraform code can affect isolation guarantees between different environments (e.g., staging vs. production). This ensures that changes in one environment do not inadvertently affect another.
:p How does file layout affect isolation in Terraform?
??x
File layout in Terraform impacts isolation by determining where and how state is stored, which can help prevent accidental modifications from one environment affecting another. For example:
```bash
# Example of a file structure that promotes isolation
├── main.tf
├── staging/
│   └── terraform.tfstate
└── production/
    └── terraform.tfstate
```
This layout ensures that changes in the `staging` directory do not affect the `production` state, maintaining clear separation and preventing unintended side effects.
x??

---


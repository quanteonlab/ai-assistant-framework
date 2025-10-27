# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 38)

**Starting Chapter:** Serving Data in Notebooks

---

#### Federated Queries
Background context: Federated queries allow users to perform analysis on data stored across multiple sources without physically moving the data. This is particularly useful when you need to blend data from various systems for ad hoc analyses or when source data needs tight control.

Federated queries can be ideal in scenarios where setting up complex ETL processes might not be necessary, and flexibility in querying different datasets is required.

:p What are federated queries used for?
??x
Federated queries are used to perform analysis on data stored across multiple sources without moving the data. They provide a way to blend data from various systems for ad hoc analyses and ensure that source data remains tightly controlled.
x??

---

#### Data Sharing in Cloud Environments
Background context: Data sharing through massively multitenant storage systems in cloud environments allows organizations to serve data securely and control access. This is especially useful when serving data internally within an organization, providing public data access, or sharing with partner businesses.

:p How does data sharing work in a cloud environment?
??x
Data sharing works by allowing consumers (analysts and data scientists) to query data hosted on storage systems without the engineers who source the data needing to handle actual queries. This is facilitated through secure mechanisms that control access and ensure compliance.
x??

---

#### Metrics Layer Tools
Background context: A metrics layer maintains and computes business logic, ensuring quality in analytics. Tools like Looker and dbt help define standard metrics and reference them in downstream queries.

:p What are metrics layer tools?
??x
Metrics layer tools maintain and compute business logic. They help ensure the quality of data used for analytics by defining and maintaining standard metrics across multiple queries. Examples include Looker and dbt.
x??

---

#### Serving Data in Notebooks
Background context: Data scientists often use notebooks to explore data, engineer features, or train models. These notebooks can access data from various sources like APIs, databases, data warehouses, or data lakes.

:p How do data scientists access data in Jupyter notebooks?
??x
Data scientists programmatically connect to a data source using appropriate libraries (e.g., pandas for Python) within Jupyter notebooks. They load files, connect to API endpoints, or make ODBC connections to databases.
x??

---

#### Credential Handling
Background context: Incorrectly handled credentials pose significant security risks in data science projects. Credentials should never be embedded in code but managed through credential managers or CLI tools.

:p How can data engineers help manage credentials?
??x
Data engineers can set standards for handling credentials, ensuring they are not embedded in code and using secure methods like credential managers or CLI tools to handle access.
x??

---

#### Handling Large Datasets Beyond Local Memory
Background context: When working with datasets that exceed the available memory on a local machine, data scientists face significant limitations. This can lead to performance bottlenecks and computational inefficiencies. To address this issue, various scalable options are available.

:p What is one solution when your dataset exceeds local machine memory?
??x
One solution is to move to a cloud-based notebook where the underlying storage and memory for the notebook can be flexibly scaled.
x??

---
#### Distributed Execution Systems
Background context: As datasets grow beyond what can be managed on a single machine, distributed execution systems such as Dask, Ray, and Spark become essential tools. These systems allow for parallel processing across multiple machines.

:p What are some popular Python-based options for handling large-scale data processing?
??x
Popular Python-based options for handling large-scale data processing include Dask, Ray, and Apache Spark.
x??

---
#### Cloud-Based Data Science Workflows
Background context: For projects that outgrow cloud notebooks, setting up a full-fledged cloud-managed offering using services like Amazon SageMaker, Google Cloud Vertex AI, or Microsoft Azure Machine Learning becomes necessary. These platforms provide scalable environments for data science workflows.

:p What are some advantages of using cloud-based ML workflow options like Amazon SageMaker?
??x
Some advantages include access to powerful computing resources, automatic scaling capabilities, and the ability to manage larger datasets efficiently without local hardware constraints.
x??

---
#### Data Science Ops (DSOps)
Background context: As data science operations become more complex, DSOps becomes crucial. It involves managing versions, updates, access control, and maintaining service level agreements (SLAs) in cloud environments.

:p What is the role of a data engineer or ML engineer in facilitating scalable cloud infrastructure?
??x
Data engineers and ML engineers play a key role by setting up cloud infrastructure, overseeing environment management, and training data scientists on cloud-based tools. They are responsible for operational tasks such as managing versions, updates, access control, and maintaining SLAs.
x??

---
#### Reverse ETL (Bidirectional Load and Transform)
Background context: Reverse ETL involves loading processed data back into source systems from OLAP databases, enhancing data accessibility and reducing friction for end-users.

:p How does reverse ETL benefit the sales team in a lead scoring model scenario?
??x
Reverse ETL benefits the sales team by loading scored leads directly back into their CRM system. This ensures that the sales team has immediate access to relevant information without needing to rely on separate dashboards or Excel files.
x??

---
#### Implementation of Reverse ETL
Background context: To implement reverse ETL, you can use off-the-shelf solutions or roll your own. Open-source and commercial managed services are available.

:p What is the main challenge in implementing reverse ETL?
??x
The main challenge is selecting a robust solution from the rapidly changing landscape of reverse ETL products.
x??

---
#### Conclusion on Reverse ETL
Background context: The term "reverse ETL" might change, but the practice of loading data back into source systems remains important.

:p Why should you consider using reverse ETL for serving data in your organization?
??x
You should consider reverse ETL to reduce friction with end-users and ensure that processed data is readily available where it’s needed most.
x??

---

#### Reverse ETL Feedback Loops

Background context: Reverse ETL involves using data from a data warehouse to inform decisions in real-time systems, such as updating ad bids. However, this process can create feedback loops that can lead to unintended consequences, like an increase in ad spend without control.

:p What are the potential risks of reverse ETL feedback loops?
??x
The primary risk is that errors or biases in the models used for decision-making can cause the data loop to amplify these issues, leading to rapid and uncontrollable growth in costs. For instance, if a bid model incorrectly increases bids over time due to misinterpretation of data patterns, it could result in excessive ad spending.

To mitigate this risk, implementing robust monitoring and guardrails is essential. This includes setting up alerts when anomalies are detected, ensuring the models can be paused or adjusted, and regularly reviewing the decision-making logic.
??x
Implementing monitoring and guardrails involves several steps:
1. **Alerts**: Setting up real-time alerts for unusual activity in bid trends.
2. **Model Pausing**: Ability to pause model execution if issues are detected.
3. **Regular Reviews**: Periodic review of the models to ensure they are still accurate.

```python
# Example pseudo-code for implementing a guardrail
def check_bid_trend(bid_data):
    # Logic to detect unusual trends in bids
    if bid_data['trend'] > THRESHOLD:
        send_alert()
        pause_model()

def send_alert():
    print("Anomaly detected: Bid trend is unusually high. Alerting stakeholders.")

def pause_model():
    print("Model execution paused due to anomaly.")
```
x??

---

#### Data Engineer Stakeholders

Background context: In the serving stage, data engineers interface with various stakeholders including data analysts, scientists, and business managers. Their role is to provide high-quality data products while not being responsible for their end uses.

:p Who are the key stakeholders a data engineer interacts with during the serving stage?
??x
The key stakeholders include:
- Data Analysts: Interpret reports using data.
- Data Scientists: Develop models and analyze data.
- MLOps/ML Engineers: Implement machine learning workflows.
- Business Stakeholders (Non-technical): Managers, executives who make decisions based on insights.

The data engineer operates in a support role, ensuring that the data provided is of high quality but not responsible for interpreting or using it.
??x
Example interactions with stakeholders:
```python
# Example pseudo-code for interfacing with stakeholders
class DataEngineer:
    def serve_data(self):
        print("Serving data to analysts.")
    
    def support_science(self):
        print("Providing datasets and metadata to scientists.")
    
    def collaborate_with_business(self):
        print("Collaborating with managers on reporting needs.")

data_engineer = DataEngineer()
data_engineer.serve_data()
data_engineer.support_science()
data_engineer.collaborate_with_business()
```
x??

---

#### Security in the Serving Stage

Background context: In the serving stage, data engineers must ensure that security principles are followed to prevent data breaches and unauthorized access. The risk is high due to the exposure of sensitive information.

:p What is a critical aspect of security when serving data?
??x
A critical aspect of security is ensuring **least privilege** for both people and systems accessing the data. This means providing only the minimum necessary permissions based on roles and responsibilities, such as read-only access for most users and write or update access for specific roles that require it.

Additionally, using fine-grained access controls and revoking access when no longer needed is essential.
??x
Example of applying least privilege in a data mesh environment:
```python
# Example pseudo-code for implementing least privilege
def grant_access(user, role):
    if role == 'analyst':
        user.grant_read_only()
    elif role == 'data_scientist':
        user.grant_write_access()
    else:
        user.grant_read_only()

user = User('JohnDoe')
grant_access(user, 'analyst')

# Ensure access revocation
def revoke_access(user):
    user.revoke_all_access()

revoke_access(user)
```
x??

---

#### Data Mesh and Team Responsibilities

Background context: As a company grows, the roles of data engineers need to evolve. In a data mesh architecture, each domain team takes on aspects of serving and must collaborate effectively.

:p How does adopting a data mesh affect responsibilities within a data team?
??x
Adopting a data mesh reorganizes responsibilities such that each domain team is responsible for its own data serving needs. This requires clear division of duties among team members to ensure effective collaboration and organizational success.

For instance, in an early-stage company, the same person might handle multiple roles like ML engineer or data scientist. However, as the company grows, these roles should be separated.
??x
Example of dividing responsibilities in a data mesh:
```python
# Example pseudo-code for organizing teams in a data mesh
class DataMesh:
    def assign_roles(self):
        print("Assigning data engineering and MLOps roles to domain teams.")
    
    def ensure_collaboration(self):
        print("Ensuring effective collaboration among domain teams.")

data_mesh = DataMesh()
data_mesh.assign_roles()
data_mesh.ensure_collaboration()
```
x??

---

#### Monitoring and Guardrails

Background context: To prevent feedback loops in reverse ETL, it is crucial to implement monitoring and guardrails. This includes setting up alerts for unusual data patterns, pausing model execution if issues are detected, and regularly reviewing the decision-making logic.

:p How can data engineers set up guardrails to mitigate risks in reverse ETL?
??x
Data engineers can set up guardrails by implementing real-time alerts when anomalies are detected, allowing models to be paused or adjusted if issues arise, and conducting regular reviews of the models to ensure they remain accurate and unbiased.

This involves setting thresholds for bid trends, sending alerts, and pausing model execution.
??x
Example pseudo-code for implementing guardrails:
```python
# Example pseudo-code for implementing a guardrail system
class GuardRailSystem:
    def check_bid_trend(self, bid_data):
        if bid_data['trend'] > THRESHOLD:
            self.send_alert()
            self.pause_model()

    def send_alert(self):
        print("Anomaly detected: Bid trend is unusually high. Alerting stakeholders.")

    def pause_model(self):
        print("Model execution paused due to anomaly.")

    def resume_model(self):
        print("Resuming model execution after review.")

guard_rail_system = GuardRailSystem()
bid_data = {'trend': 150}
guard_rail_system.check_bid_trend(bid_data)
```
x??

---


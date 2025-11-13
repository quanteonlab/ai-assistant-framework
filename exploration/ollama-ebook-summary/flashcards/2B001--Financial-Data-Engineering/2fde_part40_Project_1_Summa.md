# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 40)

**Starting Chapter:** Project 1 Summary

---

#### Loan Payment Process Overview
This section details a complex database operation for recording loan payments. The process involves inserting records into the `Transaction` table, `LoanPayment` table, and updating the `account` balance through SQL commands.

:p Describe the steps involved in processing a loan payment as illustrated by the example.
??x
The steps involve:
1. Inserting a transaction record with negative `amount` (indicating a deduction) into the `Transaction` table.
2. Using a Common Table Expression (CTE) to get the ID of the newly inserted transaction, which is then used in the `LoanPayment` table insertion.
3. Inserting data into the `LoanPayment` table using the CTE's returned ID and setting relevant fields like `loan_id`, `payment_amount`, `scheduled_payment_date`, etc.
4. Updating the customer’s account balance by subtracting the payment amount from the current balance.

```sql
DO $$
DECLARE 
    payment_amount  DECIMAL := 1000.00;
BEGIN 
WITH inserted_transaction AS (
    INSERT INTO Transaction (account_id, type, currency, amount)
    VALUES (1, 1, 'USD', -payment_amount) 
    RETURNING id
)
INSERT INTO LoanPayment (loan_id, transaction_id, payment_amount, scheduled_payment_date, payment_date, principal_amount, interest_amount, paid_amount)
SELECT 1, id, payment_amount, '2022-04-01', '2022-04-01', 900.00, 100.00, payment_amount
FROM inserted_transaction;

UPDATE account 
SET balance = balance - payment_amount 
WHERE id = 1;
END$$;
```
x??

---

#### PostgreSQL Transaction Management for Loan Payment
This example uses a `DO` block in PostgreSQL to manage transactions and operations within a single database session.

:p What is the purpose of using a `BEGIN; DO $$...$$; COMMIT;` structure in this context?
??x
The `BEGIN; DO $$...$$; COMMIT;` structure ensures that all actions are executed as part of a single transaction. This means if any step fails, none of the steps will be committed, ensuring data integrity and consistency.

```sql
BEGIN;
DO $$
DECLARE 
    payment_amount  DECIMAL := 1000.00;
BEGIN 
...
END$$;
COMMIT;
```
x??

---

#### Update Account Balance During Loan Payment
In this process, the account balance is updated after inserting a new transaction and loan payment record.

:p What SQL command is used to update the customer's account balance during the loan payment?
??x
The `UPDATE` statement is used to decrease the account balance by the amount of the loan payment:

```sql
UPDATE account 
SET balance = balance - payment_amount 
WHERE id = 1;
```
This ensures that the account's financial status is accurately reflected after the transaction.

x??

---

#### Common Table Expression (CTE) in Loan Payment Process
A CTE named `inserted_transaction` is used to insert a new transaction and return its ID, which is then reused for inserting into the `LoanPayment` table.

:p What role does the CTE (`WITH inserted_transaction AS ...`) play in this loan payment process?
??x
The CTE `inserted_transaction` plays a crucial role by inserting a new record into the `Transaction` table and returning its ID. This ID is then used as a foreign key when inserting data into the `LoanPayment` table, ensuring referential integrity.

```sql
WITH inserted_transaction AS (
    INSERT INTO Transaction (account_id, type, currency, amount)
    VALUES (1, 1, 'USD', -payment_amount) 
    RETURNING id
)
```
x??

---

#### Loan Payment Example in PostgreSQL
The provided example uses specific values and tables to illustrate a loan payment process.

:p What is the purpose of running the given query for creating a loan payment?
??x
The purpose of running this query is to demonstrate how to create a record for a loan payment, including updating related tables (`Transaction`, `LoanPayment`, and `account`). The example uses specific values such as an account ID, transaction amount, dates, and amounts split between principal and interest.

```sql
DO $$
DECLARE 
    payment_amount  DECIMAL := 1000.00;
BEGIN 
WITH inserted_transaction AS (
    INSERT INTO Transaction (account_id, type, currency, amount)
    VALUES (1, 1, 'USD', -payment_amount) 
    RETURNING id
)
INSERT INTO LoanPayment (loan_id, transaction_id, payment_amount, scheduled_payment_date, payment_date, principal_amount, interest_amount, paid_amount)
SELECT 1, id, payment_amount, '2022-04-01', '2022-04-01', 900.00, 100.00, payment_amount
FROM inserted_transaction;

UPDATE account 
SET balance = balance - payment_amount 
WHERE id = 1;
END$$;
```
x??

---

#### Data Retrieval from Alpha Vantage API
Background context: In this project, you will fetch historical stock price data using the free version of the Alpha Vantage Stock Market API. The API allows 25 requests per day, and we need to query for four stocks—Google, Amazon, IBM, and Apple. You will use specific parameters in your API request to retrieve adjusted intraday time series history.

:p How do you fetch historical stock price data from the Alpha Vantage API?
??x
To fetch historical stock price data, you can make a GET request using the `TIME_SERIES_INTRADAY` endpoint with the following parameters:
- `function`: `TIME_SERIES_INTRADAY`
- `symbol`: Stock symbol (e.g., `AMZN` for Amazon)
- `interval`: One-minute time interval
- `adjusted`: Set to `true` to adjust the data by historical splits and dividends
- `outputsize`: Set to `full` to get the full intraday history of the specified month
- `datatype`: Set to `csv` to receive the time series as a CSV file

Example code in Python:
```python
import requests
symbol = 'AMZN'  # Example symbol for Amazon
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&adjusted=true&outputsize=full&datatype=csv"
response = requests.get(url)
data = response.text
```
x??

---

#### Storing Raw Data in Database
Background context: The raw data fetched from the Alpha Vantage API needs to be stored in a PostgreSQL database. Each record should include an ingestion timestamp.

:p How do you store the raw data retrieved from the Alpha Vantage API into a PostgreSQL database?
??x
To store the raw data, you can use SQL commands such as `INSERT` to insert each row of data along with its ingestion timestamp. Here is an example using Python and the `psycopg2` library:

```python
import psycopg2

# Database connection setup
conn = psycopg2.connect(
    dbname="your_dbname",
    user="your_username",
    password="your_password",
    host="localhost"
)
cur = conn.cursor()

data = [
    {'timestamp': '2023-10-05 14:00:00', 'open': 150.5, 'close': 151.0, 'high': 151.5, 'low': 150.0, 'volume': 1000},
    # Add more data as necessary
]

# SQL query to insert data
sql = "INSERT INTO stock_data (timestamp, open, close, high, low, volume) VALUES (%s, %s, %s, %s, %s, %s)"

for row in data:
    cur.execute(sql, (
        row['timestamp'],
        row['open'],
        row['close'],
        row['high'],
        row['low'],
        row['volume']
    ))

conn.commit()
cur.close()
```
x??

---

#### Aggregating Intraday Values to Daily Averages
Background context: The intraday data needs to be aggregated into daily averages for each stock. This involves selecting and computing the average of open, close, high, low, and volume values, grouped by date and ticker symbol.

:p How do you compute the daily aggregates from the intraday price data?
??x
To compute the daily aggregates, you can use SQL queries to select the necessary columns and group them by date and ticker symbol. Here is an example using Python:

```python
import pandas as pd

# Load raw data into a DataFrame
df = pd.read_csv('path_to_your_data.csv')

# Convert timestamp column to datetime format if not already in that format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the date and ticker symbol for grouping
df_grouped = df.groupby([df['timestamp'].dt.date, 'ticker_symbol']).agg({
    'open': 'mean',
    'close': 'mean',
    'high': 'mean',
    'low': 'mean',
    'volume': 'sum'
}).reset_index()

# Save the aggregated data to a new CSV file
df_grouped.to_csv('daily_aggregates.csv', index=False)
```
x??

---

#### Deduplication of Aggregated Data
Background context: The aggregation step might produce duplicate entries, so it’s important to deduplicate the data before exporting it. This involves selecting columns and retaining only unique rows based on the date and ticker symbol.

:p How do you deduplicate the aggregated daily price data?
??x
To deduplicate the data, you can use a SQL query or a DataFrame operation in Python that removes duplicate entries while keeping the first occurrence of each group (based on date and ticker symbol).

Example using pandas:
```python
# Deduplicate the DataFrame by date and ticker symbol
df_dedup = df_grouped.drop_duplicates(subset=['timestamp', 'ticker_symbol'])

# Save the deduplicated data to a new CSV file
df_dedup.to_csv('deduplicated_daily_aggregates.csv', index=False)
```
x??

---

#### Exporting Daily Averages to Database
Background context: After aggregating and deduplicating the data, you need to export the daily averages back into the database for further analysis.

:p How do you export the daily aggregates from a CSV file to a PostgreSQL database?
??x
To export the daily aggregates from a CSV file to a PostgreSQL database, you can use SQL `INSERT` statements or a Python script that reads the CSV and inserts it into the database. Here is an example using pandas:

```python
# Load deduplicated data into a DataFrame
df = pd.read_csv('deduplicated_daily_aggregates.csv')

# Database connection setup
conn = psycopg2.connect(
    dbname="your_dbname",
    user="your_username",
    password="your_password",
    host="localhost"
)
cur = conn.cursor()

# SQL query to insert data
sql = "INSERT INTO daily_aggregated_data (timestamp, open, close, high, low, volume) VALUES (%s, %s, %s, %s, %s, %s)"

for index, row in df.iterrows():
    cur.execute(sql, (
        row['timestamp'],
        row['open'],
        row['close'],
        row['high'],
        row['low'],
        row['volume']
    ))

conn.commit()
cur.close()
```
x??

---

#### Alpha Vantage API Key Acquisition
Background context: To integrate Alpha Vantage API into your workflow, you need to obtain an API key. This is done by following the instructions on the Alpha Vantage website. The API key should be treated as a secret and not shared with others.
:p How do you acquire an API key from Alpha Vantage?
??x
To acquire an API key from Alpha Vantage, follow these steps:
1. Go to the Alpha Vantage website.
2. Register or log in if already registered.
3. Follow the on-screen instructions to claim your API key.

You will receive a unique API key that you need to keep secret and not share with anyone else.
x??

---

#### Setting Up Environment Variables
Background context: To integrate your local environment with Mage and PostgreSQL, you need to set up environment variables in an `.env` file. This ensures that your project can access the necessary configurations without hardcoding them.
:p How do you assign the Alpha Vantage API key to a variable in the `.env` file?
??x
To assign the Alpha Vantage API key to a variable in the `.env` file, open the file and add or modify the following line:
```plaintext
ALPHAVANTAGE_API_KEY=YOUR_API_KEY
```
Replace `YOUR_API_KEY` with your actual API key. This step is crucial for maintaining security.
x??

---

#### Running Project Containers Using Docker Compose
Background context: Docker Compose allows you to run multiple containers from a single YAML file, making it easier to manage complex applications like Mage and PostgreSQL. The command provided will start the required containers in detached mode.
:p How do you use Docker Compose to run project containers?
??x
To run the project containers using Docker Compose, execute the following command in your terminal:
```bash
docker compose up -d
```
This command starts three containers: Mage running on `http://localhost:6789/`, a PostgreSQL instance where data will be stored, and pgAdmin running on `http://localhost:8080/`.
x??

---

#### Creating Tables in PostgreSQL via pgAdmin
Background context: After setting up the environment, you need to create tables for storing raw intraday data and transformed daily data. This is done using SQL scripts within the pgAdmin interface.
:p How do you create the necessary tables in PostgreSQL using pgAdmin?
??x
To create the necessary tables in PostgreSQL via pgAdmin:
1. Open a tab in your browser and navigate to `http://localhost:8080/`.
2. Log in with the credentials from the `.env` file (PGADMIN_DEFAULT_EMAIL and PGADMIN_DEFAULT_PASSWORD).
3. Create a server, then open the `stock_data` database.
4. Navigate to the `public` schema.
5. Open a query tool and execute the SQL queries found in the `create_tables.sql` file.

This step ensures that your database is properly set up for storing financial data.
x??

---

#### Building an ETL Workflow with Mage
Background context: Mage is used to build an ETL workflow, where data from Alpha Vantage API is transformed and stored. You will create a pipeline in Mage to handle the process.
:p How do you start building an ETL workflow using Mage?
??x
To start building an ETL workflow using Mage:
1. Open another browser tab and navigate to `http://localhost:6789/`.
2. Click on “+ New pipeline” from the top left corner.
3. Select “Standard (batch)” as the type of pipeline.
4. Name your pipeline, for example, "Adjusted Stock Data".
5. After creating the pipeline, you will be redirected to the pipeline edit page.

On this page, configure the necessary settings and start defining the steps in your ETL workflow.
x??

---

#### Configuring IoConfig File
Background context: The `io_config.yaml` file is used to define input/output configurations for your Mage pipeline. This step involves deleting existing content and adding specific configurations from a provided file.
:p How do you configure the `io_config.yaml` file in Mage?
??x
To configure the `io_config.yaml` file in Mage:
1. Open the `io_config.yaml` file from the left sidebar.
2. Delete all existing content.
3. Add the contents of the specified `io_config.yaml` file to this file.

This step sets up the necessary configurations for your pipeline, including input and output paths.
x??

---

#### Create API Data Loader
Background context: In Step 1, you create an API data loader to fetch financial data from Alpha Vantage using Python. This involves defining a script that interacts with the API and processes the incoming data.

:p How do you create an API data loader in Mage?
??x
To create an API data loader, follow these steps:
1. Click on `+Data loader` → `Python` → `API`.
2. Name it "Load Data from Alpha Vantage".
3. In the code editor that appears, replace the default content with a script (e.g., `fetch_intraday_data.py`) to fetch data.
4. Save and add the operator.

Example code in `fetch_intraday_data.py`:
```python
import requests

def fetch_intraday_data(symbol):
    api_key = "YOUR_API_KEY"
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data
```
x??

---

#### Create Raw Data Exporter
Background context: In Step 2, you export the raw financial data fetched from Alpha Vantage to a PostgreSQL database. This is done using Python and the `export_intraday_data.py` script.

:p How do you create a raw data exporter in Mage?
??x
To create a raw data exporter, follow these steps:
1. Click on `+Data exporter` → `Python` → `PostgreSQL`.
2. Name it "Export Raw Data".
3. In the code editor that appears, replace the default content with a script (e.g., `export_intraday_data.py`) to export raw data.
4. Save and add the operator.

Example code in `export_intraday_data.py`:
```python
import psycopg2

def export_to_db(data):
    conn = psycopg2.connect(
        dbname="yourdbname",
        user="youruser",
        password="yourpassword",
        host="localhost"
    )
    cur = conn.cursor()
    
    for timestamp, values in data.items():
        query = f"INSERT INTO raw_data (timestamp, open, high, low, close) VALUES ('{timestamp}', {values['1. open']}, {values['2. high']}, {values['3. low']}, {values['4. close']})"
        cur.execute(query)
    
    conn.commit()
    cur.close()
    conn.close()
```
x??

---

#### Create Aggregation Column Selection Transformer
Background context: In Step 3, you create a transformer to select columns that are needed for aggregation.

:p How do you create an aggregation column selection transformer in Mage?
??x
To create an aggregation column selection transformer, follow these steps:
1. Click on `+Transformer` → `Python` → `Column removal` → `Keep columns`.
2. Name it "Select Aggregation Columns".
3. In the editor that appears, replace the default content with a script (e.g., `select_columns_for_aggregation.py`) to select specific columns for aggregation.
4. Save and add the operator.

Example code in `select_columns_for_aggregation.py`:
```python
import pandas as pd

def select_aggregation_columns(data):
    df = pd.DataFrame(data)
    selected_columns = ['timestamp', 'close']
    return df[selected_columns].to_dict('records')
```
x??

---

#### Create Aggregation Operator to Compute Averages
Background context: In Step 4, you create an aggregation operator that computes the average of a specific column.

:p How do you perform aggregation in Mage?
??x
To perform aggregation, follow these steps:
1. Click on `+Transformer` → `Python` → `Aggregate` → `Aggregate by average value`.
2. Name it "Compute Averages".
3. In the editor that appears, replace the default content with a script (e.g., `compute_daily_aggregates.py`) to compute averages.
4. Save and add the operator.

Example code in `compute_daily_aggregates.py`:
```python
import pandas as pd

def aggregate_by_average(data):
    df = pd.DataFrame(data)
    agg_df = df.groupby(df['timestamp'].dt.date).mean()
    return agg_df.to_dict('records')
```
x??

---

#### Create Deduplication Column Selection Transformer
Background context: In Step 5, you create a transformer to select columns for deduplication.

:p How do you create a deduplication column selection transformer in Mage?
??x
To create a deduplication column selection transformer, follow these steps:
1. Click on `+Transformer` → `Python` → `Column removal` → `Keep columns`.
2. Name it "Select Deduplication Columns".
3. In the editor that appears, replace the default content with a script (e.g., `select_columns_for_deduplication.py`) to select specific columns for deduplication.
4. Save and add the operator.

Example code in `select_columns_for_deduplication.py`:
```python
import pandas as pd

def select_deduplication_columns(data):
    df = pd.DataFrame(data)
    selected_columns = ['timestamp', 'close']
    return df[selected_columns].to_dict('records')
```
x??

---

#### Create Operator to Drop Duplicates
Background context: In Step 6, you drop duplicate rows based on specific columns.

:p How do you create an operator to drop duplicates in Mage?
??x
To create an operator that drops duplicates, follow these steps:
1. Click on `+Transformer` → `Python` → `Rows actions` → `Drop duplicates`.
2. Name it "Drop Duplicates".
3. In the editor that appears, replace the default content with a script (e.g., `drop_duplicates.py`) to handle duplicate rows.
4. Save and add the operator.

Example code in `drop_duplicates.py`:
```python
import pandas as pd

def drop_duplicates(data):
    df = pd.DataFrame(data)
    deduped_df = df.drop_duplicates(subset=['timestamp', 'close'])
    return deduped_df.to_dict('records')
```
x??

---

#### Export Daily Average Data to Database
Background context: In Step 7, you export the computed daily average data back to a PostgreSQL database.

:p How do you create an exporter for daily average data in Mage?
??x
To create an exporter for daily average data, follow these steps:
1. Click on `+Data exporter` → `Python` → `PostgreSQL`.
2. Name it "Export Daily Data".
3. In the editor that appears, replace the default content with a script (e.g., `export_daily_data.py`) to export the averaged data.
4. Save and add the operator.

Example code in `export_daily_data.py`:
```python
import psycopg2

def export_daily_average_to_db(data):
    conn = psycopg2.connect(
        dbname="yourdbname",
        user="youruser",
        password="yourpassword",
        host="localhost"
    )
    cur = conn.cursor()
    
    for index, row in enumerate(data):
        timestamp = row['timestamp']
        close_avg = row['close']
        query = f"INSERT INTO daily_average (timestamp, average_close) VALUES ('{timestamp}', {close_avg})"
        cur.execute(query)
    
    conn.commit()
    cur.close()
    conn.close()
```
x??

---

#### Execute the Workflow
Background context: Once all components are created, you can execute the entire workflow to see if it processes data correctly and updates the database.

:p How do you run a pipeline in Mage?
??x
To run a pipeline in Mage, follow these steps:
1. Navigate to the Pipelines section using Mage’s left sidebar.
2. Locate your pipeline named "adjusted_stock_data".
3. Click on the pipeline name.
4. From the top menu, select `Run@once` and confirm by clicking “Run now”.
5. Observe the execution progress through the seven steps.

Once completed with success, check the database using pgAdmin to verify that both tables are populated with data.

x??

---

#### Project 2 Overview
Background context: In Project 2, you have built a financial ETL (Extract, Transform, Load) workflow using Alpha Vantage API, Mage, Python, and PostgreSQL. This project aimed to gain hands-on experience with these tools in a real-world scenario.
:p What is the main purpose of Project 2?
??x
The main purpose was to build a financial ETL workflow to process data from the Alpha Vantage API, transform it using Python scripts, and store it in a PostgreSQL database. This project also aimed to familiarize you with Mage for workflow management and automation.
x??

---

#### Complex Data Pipeline Considerations
Background context: In real-world scenarios, pipeline design often involves more complex transformers and data quality validations beyond the scope of basic ETL workflows. Additionally, managing pipelines requires addressing challenges like scheduling, variable and secret management, triggers, scaling, concurrency, and data integration.
:p What are some common challenges in managing complex data pipelines?
??x
Common challenges include:
- Scheduling: Ensuring that tasks run at specific times or intervals.
- Variable Management: Handling dynamic inputs and states within the workflow.
- Secret Management: Safely storing sensitive information like API keys and database credentials.
- Triggers: Setting up conditions to start workflows automatically based on events.
- Scaling: Managing resource allocation during peak loads.
- Concurrency: Ensuring that tasks do not interfere with each other, especially in distributed systems.
x??

---

#### Project 3 Overview
Background context: In Project 3, you will design a microservice-based order management system (OMS) to process orders for an online store using Netflix Conductor, PostgreSQL, and Python. This project aims to provide insights into the structure and characteristics of microservices in a real-world scenario.
:p What is the main goal of Project 3?
??x
The main goal is to implement a microservice-based order management system (OMS) that processes orders for an online store using Netflix Conductor, PostgreSQL, and Python. This project will help you understand the structure and characteristics of microservices in practice.
x??

---

#### Microservice Workflow Definition
Background context: Defining the workflow involves outlining the services and their dependencies to ensure a logical sequence of operations. In this case, we need to define five microservices for an OMS: Order acknowledgement, Payment processing, Stock and inventory management, Shipping, and Notification.
:p What are the five microservices needed for the OMS?
??x
The five microservices required for the OMS are:
- Order acknowledgment service
- Payment processing service
- Stock and inventory service
- Shipping service
- Notification service
x??

---

#### Linear Workflow Structure
Background context: A linear workflow executes services one after another. For this OMS, we will design a simple linear workflow where an order first gets acknowledged, then payment is processed, followed by stock booking, and finally, shipping.
:p What is the sequence of operations in the linear workflow for the OMS?
??x
The sequence of operations in the linear workflow for the OMS is as follows:
1. Order acknowledgment service handles customer orders.
2. Payment processing service processes payments.
3. Stock and inventory service manages stock levels.
4. Shipping service schedules shipments.
5. Notification service sends notifications to customers.
x??

---

#### Workflow Diagram
Background context: Figure 12-10 illustrates the workflow structure of the OMS, showing how each microservice interacts in a linear sequence.
:p What does Figure 12-10 illustrate?
??x
Figure 12-10 illustrates the workflow structure of the OMS, detailing the interaction between five microservices: Order acknowledgment, Payment processing, Stock and inventory management, Shipping, and Notification. The diagram shows how each service is executed sequentially to process an order.
x??

---


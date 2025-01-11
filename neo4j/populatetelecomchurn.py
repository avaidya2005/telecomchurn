import pandas as pd
from neo4j import GraphDatabase
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the URI and credentials for your Neo4j database
uri = "neo4j://localhost:7687"
username = "telcochurn"
password = "telcochurn"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to execute batched queries
def execute_batch_queries(batch_queries):
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for q in batch_queries:
                tx.run(q['query'], q['parameters'])

# Read the CSV file into a DataFrame
csv_file_path = 'datafile/orig/telcochurn.csv'  # Update with your actual file path
df = pd.read_csv(csv_file_path)

# Replace NaN values with a default value (e.g., empty string)
df = df.fillna('')
print(df.info())

# Create Customer nodes
customer_queries = []
for index, row in df.iterrows():
    customer_queries.append({
    'query': """
      MERGE (c:Customer {CustomerID: $CustomerID, Gender: $Gender, SeniorCitizen: $SeniorCitizen, Partner: $Partner, Dependents: $Dependents, Tenure: $Tenure, Churn: $Churn})
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'Gender': row['Gender'],
      'SeniorCitizen': row['SeniorCitizen'],
        'Partner': row['Partner'],
        'Dependents': row['Dependents'],
        'Tenure': row['tenure'],
        'Churn': row['Churn']
    }
    })

# Create PhoneService nodes and connect to Customer
phone_service_queries = []
for index, row in df.iterrows():
    if row['PhoneService'] == 'Yes':
        phone_service_queries.append({
            'query': """
                MATCH (c:Customer {CustomerID: $CustomerID})
                MERGE (p:PhoneService {PhoneService: $PhoneService, MultipleLines: $MultipleLines})
                MERGE (c)-[:HAS_PHONE_SERVICE]->(p)
            """,
            'parameters': {
                'CustomerID': row['CustomerID'],
                'PhoneService': row['PhoneService'],
                'MultipleLines': row['MultipleLines']
            }
        })
# Create InternetService nodes and connect to Customer
internet_service_queries = []
for index, row in df.iterrows():
  internet_service_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (i:InternetService {InternetService: $InternetService, OnlineSecurity: $OnlineSecurity, OnlineBackup: $OnlineBackup, 
                               DeviceProtection: $DeviceProtection, TechSupport: $TechSupport})
      WITH c, i
      WHERE i.InternetService <> 'No'
      MERGE (c)-[:HAS_INTERNET_SERVICE]->(i)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'InternetService': row['InternetService'],
      'OnlineSecurity': row['OnlineSecurity'],
      'OnlineBackup': row['OnlineBackup'],
      'DeviceProtection': row['DeviceProtection'],
      'TechSupport': row['TechSupport'],
    }
  })

streaming_tv_queries = []
for index, row in df.iterrows():
  streaming_tv_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (st:StreamingTV {StreamingTV: $StreamingTV})
      WITH c, st
      WHERE st.StreamingTV = 'Yes'
      MERGE (c)-[:HAS_STREAMING_TV]->(st)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'StreamingTV': row['StreamingTV'],
    }
  })

# Create StreamingMovies nodes and connect to Customer
streaming_movies_queries = []
for index, row in df.iterrows():
  streaming_movies_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (sm:StreamingMovies {StreamingMovies: $StreamingMovies})
      WITH c, sm
      WHERE sm.StreamingMovies = 'Yes'
      MERGE (c)-[:HAS_STREAMING_MOVIES]->(sm)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'StreamingMovies': row['StreamingMovies'],
    }
  })

# Create Contract nodes and connect to Customer
contract_queries = []
for index, row in df.iterrows():
  contract_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (ct:Contract {Contract: $Contract})
      MERGE (c)-[:HAS_CONTRACT]->(ct)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'Contract': row['Contract'],
    }
  })

# Create Billing nodes and connect to Customer
billing_queries = []
for index, row in df.iterrows():
  billing_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (b:Billing {PaperlessBilling: $PaperlessBilling, PaymentMethod: $PaymentMethod})
      MERGE (c)-[:HAS_BILLING]->(b)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'PaperlessBilling': row['PaperlessBilling'],
      'PaymentMethod': row['PaymentMethod'],
    }
  })

# Create Charges nodes and connect to Customer
charges_queries = []
for index, row in df.iterrows():
  charges_queries.append({
    'query': """
      MATCH (c:Customer {CustomerID: $CustomerID})
      MERGE (ch:Charges {MonthlyCharges: $MonthlyCharges, TotalCharges: $TotalCharges})
      MERGE (c)-[:HAS_CHARGES]->(ch)
    """,
    'parameters': {
      'CustomerID': row['CustomerID'],
      'MonthlyCharges': row['MonthlyCharges'],
      'TotalCharges': row['TotalCharges'],
    }
  })


# Batch size for executing queries
batch_size = 1000

""" # Execute customer queries in batches
for i in range(0, len(customer_queries), batch_size):
    execute_batch_queries(customer_queries[i:i+batch_size])

# Execute phone service queries in batches
for i in range(0, len(phone_service_queries), batch_size):
    execute_batch_queries(phone_service_queries[i:i+batch_size])

# Execute phone service queries in batches
for i in range(0, len(internet_service_queries), batch_size):
    execute_batch_queries(internet_service_queries[i:i+batch_size])
 """
execute_batch_queries(customer_queries)
execute_batch_queries(phone_service_queries)
execute_batch_queries(internet_service_queries)
execute_batch_queries(streaming_tv_queries)
execute_batch_queries(streaming_movies_queries)
execute_batch_queries(contract_queries)
execute_batch_queries(billing_queries)
execute_batch_queries(charges_queries)

# Close the driver connection
driver.close()

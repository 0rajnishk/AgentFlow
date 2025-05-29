import os
import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from app.core.config import get_settings
from app.core.logging import logger

class DatabaseManager:
    def __init__(self, db_path: str = "project.db"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            self.init_database()

    def init_database(self):
        """Initialize database from CSV if empty"""
        if not os.path.exists(r"G:\Rajnish\Desktop\fetchai\AgentFlow\App\app\DataCoSupplyChainDataset.csv"):
            logger.error("CSV file not found for data population.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create schema (your existing schema remains the same)
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS Roles (
            RoleId INTEGER PRIMARY KEY AUTOINCREMENT,
            RoleName TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS Users (
            UserId INTEGER PRIMARY KEY AUTOINCREMENT,
            Username TEXT NOT NULL UNIQUE,
            Email TEXT,
            PasswordHash TEXT NOT NULL,
            RoleId INTEGER,
            Region TEXT,
            FOREIGN KEY (RoleId) REFERENCES Roles(RoleId)
        );

        CREATE TABLE IF NOT EXISTS Customers (
            CustomerId INTEGER PRIMARY KEY,
            Fname TEXT, Lname TEXT, Email TEXT, City TEXT, State TEXT, Country TEXT,
            Zipcode TEXT, Segment TEXT, Street TEXT, Password TEXT
        );

        CREATE TABLE IF NOT EXISTS Products (
            ProductCardId INTEGER PRIMARY KEY,
            ProductCategoryId INTEGER,
            ProductName TEXT,
            ProductDescription TEXT,
            ProductPrice REAL,
            ProductImage TEXT,
            ProductStatus INTEGER
        );

        CREATE TABLE IF NOT EXISTS Orders (
            OrderId INTEGER PRIMARY KEY,
            OrderDate TEXT,
            ShippingDate TEXT,
            OrderStatus TEXT,
            OrderRegion TEXT,
            OrderCountry TEXT,
            OrderState TEXT,
            OrderCity TEXT,
            OrderZipcode TEXT,
            OrderCustomerId INTEGER,
            ProductCardId INTEGER,
            SalesPerCustomer REAL,
            OrderProfitPerOrder REAL,
            OrderItemTotal REAL,
            ShippingMode TEXT,
            DeliveryStatus TEXT,
            LateDeliveryRisk INTEGER,
            FOREIGN KEY (OrderCustomerId) REFERENCES Customers(CustomerId),
            FOREIGN KEY (ProductCardId) REFERENCES Products(ProductCardId)
        );

        CREATE TABLE IF NOT EXISTS Chats (
            ChatId INTEGER PRIMARY KEY AUTOINCREMENT,
            UserId INTEGER,
            ChatTitle TEXT,
            CreatedAt TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (UserId) REFERENCES Users(UserId)
        );

        CREATE TABLE IF NOT EXISTS Messages (
            MessageId INTEGER PRIMARY KEY AUTOINCREMENT,
            ChatId INTEGER,
            Role TEXT CHECK(Role IN ('user', 'assistant')),
            Content TEXT NOT NULL,
            Timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ChatId) REFERENCES Chats(ChatId)
        );
        """)

        # Load data (your existing data loading remains the same)
        df = pd.read_csv("G:\Rajnish\Desktop\fetchai\AgentFlow\App\app\DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")

        customers_df = df[[
            "Customer Id", "Customer Fname", "Customer Lname", "Customer Email", "Customer City",
            "Customer State", "Customer Country", "Customer Zipcode", "Customer Segment",
            "Customer Street", "Customer Password"
        ]].drop_duplicates().rename(columns=lambda x: x.replace("Customer ", ""))
        customers_df.columns = [col.replace(" ", "") for col in customers_df.columns]
        customers_df.to_sql("Customers", conn, if_exists="replace", index=False)

        products_df = df[[
            "Product Card Id", "Product Category Id", "Product Name", "Product Description",
            "Product Price", "Product Image", "Product Status"
        ]].drop_duplicates()
        products_df.columns = [col.replace(" ", "") for col in products_df.columns]
        products_df.to_sql("Products", conn, if_exists="replace", index=False)

        orders_df = df[[
            "Order Id", "order date (DateOrders)", "shipping date (DateOrders)", "Order Status",
            "Order Region", "Order Country", "Order State", "Order City", "Order Zipcode",
            "Order Customer Id", "Product Card Id", "Sales per customer", "Order Profit Per Order",
            "Order Item Total", "Shipping Mode", "Delivery Status", "Late_delivery_risk"
        ]].drop_duplicates()
        orders_df.columns = [
            "OrderId", "OrderDate", "ShippingDate", "OrderStatus", "OrderRegion", "OrderCountry",
            "OrderState", "OrderCity", "OrderZipcode", "OrderCustomerId", "ProductCardId",
            "SalesPerCustomer", "OrderProfitPerOrder", "OrderItemTotal", "ShippingMode",
            "DeliveryStatus", "LateDeliveryRisk"
        ]
        orders_df.to_sql("Orders", conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()
        logger.info("Database initialized and data imported from CSV.")

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Log the query being executed
            logger.info(f"Executing SQL query: {query}")
            
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            result = [dict(zip(columns, row)) for row in rows]
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            logger.info(f"Query executed successfully, returned {result}") # Added for better debugging
            
            return result
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            logger.error(f"Failed query: {query}")
            raise Exception(f"Database query failed: {str(e)}")

    def get_table_info(self) -> str:
        """Get detailed table information for better SQL generation, including unique values for categorical columns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                column_details = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    
                    column_description = f"{col_name} ({col_type})"
                    
                    # Heuristic to identify potential categorical columns
                    if col_type.upper() == 'TEXT':
                        try:
                            cursor.execute(f"SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\" LIMIT 20;")
                            unique_values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                            if unique_values:
                                column_description += f" - Expected values: {', '.join(unique_values)}"
                        except sqlite3.OperationalError as e:
                            logger.warning(f"Could not get unique values for {table_name}.{col_name}: {e}")
                    elif col_type.upper() == 'INTEGER' and col_name.lower().endswith('id'):
                        column_description += " - (Foreign Key/Identifier)"
                    elif col_type.upper() in ['REAL', 'INTEGER']:
                        column_description += " - (Numeric data)"

                    column_details.append(column_description)
                
                # Get sample data
                cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 3;")
                sample_rows = cursor.fetchall()
                
                schema_info.append(f"""
Table: {table_name}
Columns:
{chr(10).join([f"  - {detail}" for detail in column_details])}
Sample rows: {len(sample_rows)} entries
""")
            
            conn.close()
            return '\n'.join(schema_info)
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return self._get_schema_info()

    def _get_schema_info(self) -> str:
        """Fallback schema information (consider updating this with sample values if dynamic fails often)"""
        return """
Database Schema Information:

Table: Orders
Columns: 
  - OrderId (INTEGER) - (Foreign Key/Identifier)
  - OrderDate (TEXT) - (Date in YYYY-MM-DD format)
  - ShippingDate (TEXT) - (Date in YYYY-MM-DD format)
  - OrderStatus (TEXT) - Expected values: Processing, Complete, Pending, Canceled, Closed, Payment Review, On Hold, Suspected Fraud, Pending Payment
  - OrderRegion (TEXT) - Expected values: East, West, Central, South, North
  - OrderCountry (TEXT) - Expected values: USA, Mexico, Canada, Brazil, UK, Germany, France, etc.
  - OrderState (TEXT) - Expected values: California, Texas, New York, Florida, etc.
  - OrderCity (TEXT) - Expected values: New York, Los Angeles, Chicago, Houston, etc.
  - OrderZipcode (TEXT)
  - OrderCustomerId (INTEGER) - (Foreign Key/Identifier)
  - ProductCardId (INTEGER) - (Foreign Key/Identifier)
  - SalesPerCustomer (REAL) - (Numeric data)
  - OrderProfitPerOrder (REAL) - (Numeric data)
  - OrderItemTotal (REAL) - (Numeric data)
  - ShippingMode (TEXT) - Expected values: Standard Class, Second Class, First Class, Same Day
  - DeliveryStatus (TEXT) - Expected values: Advance shipping, Late delivery, Shipping on time, Shipping canceled
  - LateDeliveryRisk (INTEGER) - Expected values: 0 (No Risk), 1 (High Risk)
Description: Contains order transactions with delivery, shipping, regional, and financial data.

Table: Customers  
Columns: 
  - CustomerId (INTEGER) - (Foreign Key/Identifier)
  - Fname (TEXT) - Expected values: Cally, Irene, Gillian, Tana, Orli, Kimberly, Constance, Erica (sample values)
  - Lname (TEXT) - Expected values: Smith, Johnson, Williams, Jones (sample values)
  - Email (TEXT) - Expected values: XXXXXXXXX (placeholder for masked emails)
  - City (TEXT) - Expected values: Caguas, San Jose, Los Angeles, Tonawanda, Miami (sample values)
  - State (TEXT) - Expected values: CA, TX, NY, FL (sample values)
  - Country (TEXT) - Expected values: EE. UU., Puerto Rico
  - Zipcode (TEXT)
  - Segment (TEXT) - Expected values: Consumer, Corporate, Home Office
  - Street (TEXT)
  - Password (TEXT)
Description: Customer information including contact details, location, and market segmentation.

Table: Products
Columns: 
  - ProductCardId (INTEGER) - (Foreign Key/Identifier)
  - ProductCategoryId (INTEGER)
  - ProductName (TEXT) - Expected values: Smartwatch, Laptop, T-Shirt, Running Shoes (sample values)
  - ProductDescription (TEXT)
  - ProductPrice (REAL) - (Numeric data)
  - ProductImage (TEXT)
  - ProductStatus (INTEGER) - Expected values: 0, 1 (e.g., 1 for active, 0 for inactive)
Description: Product catalog with pricing, categorization, and status information.

Key Relationships:
- Orders.OrderCustomerId → Customers.CustomerId
- Orders.ProductCardId → Products.ProductCardId
"""


class NLToSQLConverter:
    def __init__(self):
        genai.configure(api_key=get_settings().GENAI_API_KEY)
        self.model = genai.GenerativeModel(get_settings().GENAI_MODEL)
        self.db_manager = DatabaseManager()

    def convert_to_sql(self, natural_language_query: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Convert natural language query to SQL with enhanced context awareness"""
        
        # Get dynamic schema information including unique values for categorical columns
        schema_info = self.db_manager.get_table_info()
        
        # Build user context information
        user_info = ""
        region_filter = ""
        
        if user_context:
            user_role = user_context.get("role_name", "User")
            user_region = user_context.get("region", "Global")
            user_id = user_context.get("user_id", "Unknown")
            
            user_info = f"""
    User Context:
    - Role: {user_role}
    - Region: {user_region}
    - User ID: {user_id}
    """
            
            # Add region-specific filtering if not Global access
            if user_region != "Global" and user_region != "global":
                region_filter = f"""
    IMPORTANT: This user has {user_region} region access only. 
    If the query involves regional data, filter by that region. For example, add WHERE OrderRegion = '{user_region}' or OrderCountry = '{user_region}'.
    """

        prompt = f"""You are an expert SQL query generator for supply chain analytics. Convert the natural language query into a precise, optimized SQL query.

    {schema_info}

    {user_info}

    {region_filter}

    Natural Language Query: "{natural_language_query}"

    CRITICAL INSTRUCTIONS:
    1. Generate ONLY the SQL query - no explanations, no markdown formatting, no additional text.
    2. Use exact column names as specified in the schema (case-sensitive).
    3. Use proper SQLite syntax and functions.
    4. IMPORTANT: Work only with the existing tables (Orders, Customers, Products) and their actual columns.
    5. Do NOT assume additional tables or columns that don't exist in the schema.
    6. For categorical filters, use the exact "Expected values" provided in the schema.
    7. Use appropriate JOINs when querying multiple tables.


    SQL Query:"""

        try:
            response = self.model.generate_content(prompt)
            
            # Handle safety filter blocking
            if not hasattr(response, 'text') or not response.text:
                logger.warning("Gemini response was blocked or empty, using fallback SQL generation")
                return self._generate_fallback_sql(natural_language_query)
            
            sql_query = response.text.strip()
            
            # Clean up the response
            sql_query = self._clean_sql_response(sql_query)
            
            logger.info(f"Generated SQL query for user {user_context.get('username', 'Unknown') if user_context else 'Unknown'}: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            logger.info("Falling back to simple SQL generation")
            return self._generate_fallback_sql(natural_language_query)


    def _clean_sql_response(self, sql_query: str) -> str:
        """Clean and validate SQL response"""
        # Remove markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        # Remove extra whitespace and newlines
        sql_query = sql_query.strip()
        
        # Basic validation
        sql_query_lower = sql_query.lower()
        if not sql_query_lower.startswith('select'):
            raise Exception("Generated query must be a SELECT statement")
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create']
        for keyword in dangerous_keywords:
            if keyword in sql_query_lower:
                raise Exception(f"Query contains dangerous operation: {keyword}")
        
        return sql_query


    def _generate_fallback_sql(self, query: str) -> str:
        """Generate simple fallback SQL when Gemini fails or is blocked"""
        query_lower = query.lower()
        print(f"Generating fallback SQL for query: {query_lower} --------- ")
        # Simple heuristic-based SQL generation
        if 'product' in query_lower:
            return "SELECT ProductCardId, ProductName, ProductPrice FROM Products LIMIT 100;"
        elif 'customer' in query_lower:
            return "SELECT CustomerId, Fname, Lname, City, State, Country FROM Customers LIMIT 100;"
        elif 'order' in query_lower:
            return "SELECT OrderId, OrderDate, OrderStatus, OrderRegion, SalesPerCustomer FROM Orders LIMIT 100;"
        else:
            # Default fallback
            return "SELECT COUNT(*) as total_records FROM Orders;"

class SQLAnswerGenerator:
    def __init__(self):
        genai.configure(api_key=get_settings().GENAI_API_KEY)
        self.model = genai.GenerativeModel(get_settings().GENAI_MODEL)


    def generate_answer(self, original_query: str, sql_query: str, data: List[Dict[str, Any]], user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate concise natural language answer from SQL results"""

        # Prepare data summary
        data_summary = self._prepare_data_summary(data)

        prompt = f"""Based on the following data, generate a helpful and natural language answer to the original question.

Original Question: "{original_query}"
SQL Query Executed: "{sql_query}"
Data Retrieved: {data_summary}

ANSWER REQUIREMENTS:
1. Formulate a complete and grammatically correct sentence or paragraph that directly answers the original question.
2. Incorporate the relevant information from 'Data Retrieved' into your answer.
3. If 'Data Retrieved' contains a single value (e.g., "The result is: First Class" for the shipping mode question), integrate that value into a full sentence that answers the original question.
4. If 'Data Retrieved' presents a list or table of results, summarize or list them clearly in a natural way.
5. Do NOT include the SQL query in your answer.
6. Do NOT explicitly state "Based on the provided data" or "According to the results". Just state the answer directly.
7. If 'Data Retrieved' indicates "No data found", simply state "No data found for your query."

Answer:"""


        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()

            logger.info(f"Generated concise answer for query: {original_query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating SQL answer: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"


    def _prepare_data_summary(self, data: List[Dict[str, Any]]) -> str:
        """Prepare a concise summary of the data for the prompt"""
        if not data:
            return "No data found"
        
        # --- NEW LOGIC FOR SINGLE-VALUE RESULTS ---
        if len(data) == 1 and len(data[0]) == 1:
            key = list(data[0].keys())[0]
            value = data[0][key]
            return f"The direct answer value is: {value}"
        # --- END NEW LOGIC ---

        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"Total Records: {len(data)}")
        
        # Sample data (first 5 rows)
        if len(data) <= 5:
            summary_parts.append(f"All Data:\n{json.dumps(data, indent=2)}")
        else:
            summary_parts.append(f"Sample Data (first 5 rows):\n{json.dumps(data[:5], indent=2)}")
            summary_parts.append(f"[... {len(data) - 5} more rows]")
        
        # Column analysis
        if data:
            columns = list(data[0].keys())
            summary_parts.append(f"Columns: {', '.join(columns)}")
            
            # Numeric column summaries
            numeric_summaries = []
            for col in columns:
                try:
                    values = [row[col] for row in data if row[col] is not None and isinstance(row[col], (int, float))]
                    if values:
                        numeric_summaries.append(f"{col}: min={min(values)}, max={max(values)}, avg={sum(values)/len(values):.2f}")
                except:
                    continue
            
            if numeric_summaries:
                summary_parts.append("Numeric Summaries:")
                summary_parts.extend(numeric_summaries)
        
        return '\n'.join(summary_parts)

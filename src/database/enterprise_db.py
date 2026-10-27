"""
Enterprise Database Module for Structured Business Data & Text-to-SQL Analytics.
"""

from pathlib import Path
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.config.settings import DB_PATH

class EnterpriseDatabase:
    """Manages SQLite enterprise database with realistic business records and SQL analytics."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database_initialized()

    def get_connection(self) -> sqlite3.Connection:
        """Return a connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_database_initialized(self):
        """Creates tables and seed data if not already present."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check if employees table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
            )
            table_exists = cursor.fetchone()
            
            if not table_exists:
                self._create_schema(cursor)
                self._seed_data(cursor)
                conn.commit()

    def _create_schema(self, cursor: sqlite3.Cursor):
        """Defines enterprise SQL schemas."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                salary INTEGER NOT NULL,
                hire_date TEXT NOT NULL,
                remote_days_per_week INTEGER NOT NULL,
                performance_rating REAL NOT NULL,
                retention_risk TEXT NOT NULL,
                status TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales_transactions (
                transaction_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                product TEXT NOT NULL,
                region TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                customer_tier TEXT NOT NULL,
                amount REAL NOT NULL,
                units INTEGER NOT NULL,
                discount_pct REAL NOT NULL,
                sales_rep TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                created_date TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                resolution_time_hours REAL NOT NULL,
                status TEXT NOT NULL,
                customer_satisfaction INTEGER NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                quarter TEXT PRIMARY KEY,
                revenue REAL NOT NULL,
                gross_profit REAL NOT NULL,
                operating_expenses REAL NOT NULL,
                net_income REAL NOT NULL,
                active_customers INTEGER NOT NULL,
                churn_rate_pct REAL NOT NULL
            );
        """)

    def _seed_data(self, cursor: sqlite3.Cursor):
        """Populate realistic enterprise records."""
        # 1. Employees (35 key records across engineering, sales, ops, security, HR)
        employees_data = [
            (101, 'Alex Rivera', 'Engineering', 'Staff AI Architect', 178000, '2022-03-15', 2, 4.8, 'Low', 'Active'),
            (102, 'Priya Sharma', 'Engineering', 'Senior Distributed Systems Engineer', 152000, '2023-01-10', 2, 4.7, 'Low', 'Active'),
            (103, 'David Chen', 'Engineering', 'DevOps & Cloud Lead', 145000, '2021-08-20', 1, 4.5, 'Medium', 'Active'),
            (104, 'Sophia Martinez', 'Product', 'VP of Enterprise Products', 185000, '2021-04-01', 2, 4.9, 'Low', 'Active'),
            (105, 'Marcus Vance', 'Security', 'Principal Security Architect', 168000, '2022-06-18', 1, 4.6, 'Low', 'Active'),
            (106, 'Elena Rostova', 'Sales', 'Strategic Enterprise AE', 135000, '2023-04-12', 2, 4.6, 'Low', 'Active'),
            (107, 'James Wilson', 'Sales', 'Regional Sales Director - EMEA', 162000, '2021-11-05', 2, 4.4, 'Low', 'Active'),
            (108, 'Tara Patel', 'Customer Support', 'Director of Customer Experience', 128000, '2022-02-01', 2, 4.7, 'Low', 'Active'),
            (109, 'Liam O\'Connor', 'Customer Support', 'Tier-3 Escalations Engineer', 96000, '2023-07-15', 2, 4.3, 'Medium', 'Active'),
            (110, 'Amina Yusuf', 'Engineering', 'Full Stack Frontend Engineer', 115000, '2024-02-01', 2, 4.2, 'High', 'Active'),
            (111, 'Ethan Walker', 'Marketing', 'Head of Enterprise Growth', 138000, '2022-09-10', 1, 4.4, 'Low', 'Active'),
            (112, 'Zoe Takahashi', 'Finance', 'Director of Strategic Finance', 155000, '2021-06-15', 1, 4.8, 'Low', 'Active'),
            (113, 'Lucas Bennett', 'Sales', 'Senior Account Executive - APAC', 125000, '2023-08-25', 2, 4.1, 'High', 'Active'),
            (114, 'Chloe Dubois', 'Human Resources', 'People Operations Partner', 98000, '2022-10-01', 2, 4.5, 'Low', 'Active'),
            (115, 'Siddharth Rao', 'Engineering', 'ML Platform Engineer', 142000, '2023-03-20', 2, 4.6, 'Low', 'Active'),
            (116, 'Hannah Kim', 'Product', 'Senior Technical Product Manager', 139000, '2022-12-05', 2, 4.7, 'Low', 'Active'),
            (117, 'Gabriel Silva', 'Sales', 'Enterprise AE - LATAM', 118000, '2024-01-15', 2, 4.3, 'Medium', 'Active'),
            (118, 'Rachel Green', 'Customer Support', 'Customer Support Specialist', 74000, '2024-05-10', 2, 3.9, 'High', 'Active'),
            (119, 'Oscar Lindqvist', 'Engineering', 'Data Platform Engineer', 136000, '2023-09-01', 2, 4.4, 'Medium', 'Active'),
            (120, 'Nadia Belkacem', 'Security', 'Compliance & SOC2 Officer', 132000, '2023-05-15', 1, 4.6, 'Low', 'Active')
        ]
        cursor.executemany("""
            INSERT INTO employees (id, name, department, role, salary, hire_date, remote_days_per_week, performance_rating, retention_risk, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, employees_data)

        # 2. Sales Transactions
        sales_data = [
            ('TX-901', '2025-01-14', 'EnterpriseAI', 'North America', 'FinTech Global Corp', 'Enterprise', 145000, 1, 0.10, 'Elena Rostova'),
            ('TX-902', '2025-01-22', 'CloudPlatform', 'EMEA', 'Nordic Health Systems', 'Enterprise', 210000, 1, 0.05, 'James Wilson'),
            ('TX-903', '2025-02-05', 'SecuritySuite', 'North America', 'Apex Aerospace', 'Enterprise', 95000, 1, 0.00, 'Elena Rostova'),
            ('TX-904', '2025-02-18', 'EnterpriseAI', 'APAC', 'Tokyo Retail Holdings', 'Enterprise', 180000, 1, 0.15, 'Lucas Bennett'),
            ('TX-905', '2025-03-02', 'DataHub', 'North America', 'Beacon Logistics', 'Mid-Market', 42000, 1, 0.00, 'Elena Rostova'),
            ('TX-906', '2025-03-20', 'CloudPlatform', 'LATAM', 'Banco Paulista S.A.', 'Enterprise', 160000, 1, 0.12, 'Gabriel Silva'),
            ('TX-907', '2025-04-10', 'EnterpriseAI', 'North America', 'OmniMedia Network', 'Enterprise', 230000, 1, 0.08, 'Elena Rostova'),
            ('TX-908', '2025-04-28', 'SecuritySuite', 'EMEA', 'Zurich Insurance Labs', 'Enterprise', 115000, 1, 0.05, 'James Wilson'),
            ('TX-909', '2025-05-15', 'CloudPlatform', 'APAC', 'Singapore Tech Venture', 'Mid-Market', 68000, 1, 0.00, 'Lucas Bennett'),
            ('TX-910', '2025-06-02', 'DataHub', 'EMEA', 'Berlin Automotive KG', 'Enterprise', 84000, 1, 0.05, 'James Wilson'),
            ('TX-911', '2025-06-25', 'EnterpriseAI', 'North America', 'Cascade Health Cloud', 'Enterprise', 320000, 1, 0.10, 'Elena Rostova'),
            ('TX-912', '2025-07-12', 'SecuritySuite', 'LATAM', 'Monterrey Telecom', 'Enterprise', 78000, 1, 0.05, 'Gabriel Silva'),
            ('TX-913', '2025-08-04', 'CloudPlatform', 'North America', 'Summit Financial', 'Enterprise', 260000, 1, 0.15, 'Elena Rostova'),
            ('TX-914', '2025-08-22', 'EnterpriseAI', 'EMEA', 'Royal Logistics UK', 'Enterprise', 195000, 1, 0.05, 'James Wilson'),
            ('TX-915', '2025-09-18', 'DataHub', 'APAC', 'Seoul Analytics Co', 'Mid-Market', 52000, 1, 0.00, 'Lucas Bennett'),
            ('TX-916', '2025-10-05', 'EnterpriseAI', 'North America', 'Vanguard BioTech', 'Enterprise', 275000, 1, 0.05, 'Elena Rostova'),
            ('TX-917', '2025-10-29', 'SecuritySuite', 'North America', 'Horizon Defense Tech', 'Enterprise', 140000, 1, 0.00, 'Elena Rostova'),
            ('TX-918', '2025-11-14', 'CloudPlatform', 'EMEA', 'Amsterdam Data Corp', 'Enterprise', 185000, 1, 0.08, 'James Wilson'),
            ('TX-919', '2025-12-08', 'EnterpriseAI', 'APAC', 'Sydney Cloud Services', 'Enterprise', 215000, 1, 0.10, 'Lucas Bennett'),
            ('TX-920', '2025-12-22', 'DataHub', 'North America', 'Pioneer Manufacturing', 'Mid-Market', 48000, 1, 0.00, 'Elena Rostova')
        ]
        cursor.executemany("""
            INSERT INTO sales_transactions (transaction_id, date, product, region, customer_name, customer_tier, amount, units, discount_pct, sales_rep)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sales_data)

        # 3. Support Tickets
        tickets_data = [
            ('TCK-101', '2025-11-02', 'FinTech Global Corp', 'Billing/Refund', 'Medium', 3.5, 'Resolved', 5),
            ('TCK-102', '2025-11-04', 'Nordic Health Systems', 'Performance/Latency', 'Critical', 1.2, 'Resolved', 5),
            ('TCK-103', '2025-11-10', 'Apex Aerospace', 'Access/MFA', 'High', 2.0, 'Resolved', 4),
            ('TCK-104', '2025-11-15', 'Tokyo Retail Holdings', 'Integration/API', 'Medium', 6.0, 'Resolved', 4),
            ('TCK-105', '2025-11-20', 'Beacon Logistics', 'Billing/Refund', 'Low', 12.0, 'Resolved', 4),
            ('TCK-106', '2025-11-25', 'Banco Paulista S.A.', 'Product Bug', 'Critical', 2.5, 'Resolved', 4),
            ('TCK-107', '2025-12-01', 'OmniMedia Network', 'Access/MFA', 'High', 1.5, 'Resolved', 5),
            ('TCK-108', '2025-12-05', 'Zurich Insurance Labs', 'Security Incident', 'Critical', 0.8, 'Resolved', 5),
            ('TCK-109', '2025-12-12', 'Singapore Tech Venture', 'Integration/API', 'Medium', 4.5, 'Resolved', 4),
            ('TCK-110', '2025-12-18', 'Cascade Health Cloud', 'Performance/Latency', 'High', 3.0, 'Resolved', 4),
            ('TCK-111', '2026-01-05', 'Royal Logistics UK', 'Billing/Refund', 'Low', 8.5, 'Resolved', 3),
            ('TCK-112', '2026-01-12', 'Vanguard BioTech', 'Integration/API', 'High', 4.0, 'Resolved', 5),
            ('TCK-113', '2026-01-20', 'Horizon Defense Tech', 'Access/MFA', 'Critical', 1.0, 'Resolved', 5),
            ('TCK-114', '2026-02-02', 'Amsterdam Data Corp', 'Product Bug', 'Medium', 5.2, 'In Progress', 4),
            ('TCK-115', '2026-02-15', 'Sydney Cloud Services', 'Billing/Refund', 'Low', 14.0, 'In Progress', 4)
        ]
        cursor.executemany("""
            INSERT INTO support_tickets (ticket_id, created_date, customer_name, issue_type, priority, resolution_time_hours, status, customer_satisfaction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tickets_data)

        # 4. Financial Quarters
        financial_data = [
            ('2024-Q1', 22.4, 16.8, 19.2, 1.2, 280, 2.1),
            ('2024-Q2', 25.1, 19.1, 20.5, 1.8, 310, 1.9),
            ('2024-Q3', 28.5, 21.9, 21.8, 2.4, 345, 1.7),
            ('2024-Q4', 30.4, 23.5, 22.1, 2.7, 380, 1.5),
            ('2025-Q1', 31.8, 24.6, 21.0, 3.6, 415, 1.4),
            ('2025-Q2', 33.9, 26.5, 21.8, 4.7, 460, 1.2),
            ('2025-Q3', 35.9, 28.1, 22.4, 5.7, 505, 1.1),
            ('2025-Q4', 41.2, 32.3, 24.2, 8.5, 560, 0.9)
        ]
        cursor.executemany("""
            INSERT INTO financial_metrics (quarter, revenue, gross_profit, operating_expenses, net_income, active_customers, churn_rate_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, financial_data)

    def execute_query(self, sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Safely execute a SELECT query against the enterprise SQLite database.
        Returns (DataFrame, error_string).
        """
        sanitized = sql_query.strip()
        
        # Guard against destructive queries
        disallowed_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE"]
        first_token = sanitized.split()[0].upper() if sanitized.split() else ""
        
        if first_token != "SELECT" and first_token != "WITH" and first_token != "PRAGMA":
            return None, "Only read-only queries (SELECT) are permitted for security."

        for kw in disallowed_keywords:
            if f" {kw} " in f" {sanitized.upper()} ":
                return None, f"Operation '{kw}' is not allowed in analytics mode."

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                df = pd.read_sql_query(sanitized, conn)
                return df, None
        except Exception as err:
            return None, str(err)

    def get_table_names(self) -> List[str]:
        """Return list of all tables in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Return schema details for a specific table."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "dflt_value": row[4],
                    "pk": bool(row[5])
                })
            return columns

    def get_schema_summary(self) -> str:
        """Returns a formatted schema summary for LLM prompt context."""
        summary_lines = []
        tables = self.get_table_names()
        for tbl in tables:
            cols = self.get_table_schema(tbl)
            col_desc = ", ".join([f"{c['name']} ({c['type']})" for c in cols])
            summary_lines.append(f"TABLE {tbl}: {col_desc}")
            
            # Fetch 2 sample rows
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {tbl} LIMIT 2")
                rows = cursor.fetchall()
                if rows:
                    summary_lines.append(f"  Example rows: {rows}")
        return "\n".join(summary_lines)

    def get_table_df(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """Return table data as Pandas DataFrame."""
        with sqlite3.connect(str(self.db_path)) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)

# Singleton instance
enterprise_db = EnterpriseDatabase()

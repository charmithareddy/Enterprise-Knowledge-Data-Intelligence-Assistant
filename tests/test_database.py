"""
Unit tests for the Enterprise SQLite Database module.
"""

import unittest
from src.database.enterprise_db import enterprise_db

class TestEnterpriseDatabase(unittest.TestCase):

    def test_tables_created(self):
        tables = enterprise_db.get_table_names()
        self.assertIn("employees", tables)
        self.assertIn("sales_transactions", tables)
        self.assertIn("support_tickets", tables)
        self.assertIn("financial_metrics", tables)

    def test_read_employees(self):
        df, err = enterprise_db.execute_query("SELECT COUNT(*) as count FROM employees")
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertGreater(int(df.iloc[0]["count"]), 0)

    def test_safe_query_guard(self):
        # Disallow DROP or DELETE
        df, err = enterprise_db.execute_query("DROP TABLE employees")
        self.assertIsNotNone(err)
        self.assertIn("Only read-only queries", err)

    def test_schema_summary(self):
        summary = enterprise_db.get_schema_summary()
        self.assertIn("TABLE employees", summary)
        self.assertIn("TABLE sales_transactions", summary)

if __name__ == "__main__":
    unittest.main()

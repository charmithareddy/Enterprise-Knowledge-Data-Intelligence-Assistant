"""
Enterprise AI Agent Orchestrator & Multi-Tool Router.
Unifies Document RAG, GraphRAG, Text-to-SQL Analytics, Machine Learning Predictions, and Web Search.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import json
import time
from google import genai
from ddgs import DDGS
import pandas as pd

from src.config.settings import DEFAULT_GEMINI_MODEL, get_gemini_api_key
from src.database.enterprise_db import enterprise_db
from src.graph.knowledge_graph import enterprise_kg
from src.models.ml_engine import enterprise_ml
from src.retrieval.vector_store import vector_store

class EnterpriseAgent:
    """Intelligent coordinator that classifies user queries and routes them to specialized tools."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL):
        self.model_name = model_name

    def classify_intent(self, query: str) -> str:
        """Determine the primary technical capability required for a query."""
        q = query.lower()

        # 1. ML Predictive & Forecasting indicators
        ml_indicators = ["predict", "forecast", "what-if", "retention risk", "churn risk", "simulate", "attrition risk", "expected revenue"]
        if any(term in q for term in ml_indicators):
            return "ML_PREDICTION"

        # 2. SQL Analytics & Structured Database indicators
        sql_indicators = [
            "how many", "count", "average", "total revenue", "highest salary", "lowest salary",
            "list employees", "sales in", "transactions", "support tickets", "csat",
            "table", "sql", "by department", "by region", "database", "who earns", "sales rep"
        ]
        if any(term in q for term in sql_indicators):
            return "SQL_ANALYTICS"

        # 3. GraphRAG & Entity Relationship indicators
        graph_indicators = [
            "who governs", "relationship", "connected to", "hierarchy", "governance",
            "dependencies", "who reports", "which department manages", "entity", "topology", "chain"
        ]
        if any(term in q for term in graph_indicators):
            return "GRAPH_RAG"

        # 4. Web Search indicators
        web_indicators = ["latest", "today", "news", "current price", "search web", "online", "2026", "market news"]
        if any(term in q for term in web_indicators):
            return "WEB_SEARCH"

        # Default to Document RAG
        return "DOCUMENT_RAG"

    def execute_text_to_sql(self, query: str, api_key: str) -> Dict[str, Any]:
        """Convert natural language to SQL, execute against SQLite, and synthesize response."""
        schema = enterprise_db.get_schema_summary()

        prompt = f"""
You are an expert SQLite Business Intelligence data analyst.
Convert the user's natural language request into a single valid, read-only SQLite query.

DATABASE SCHEMA:
{schema}

RULES:
- Return ONLY the raw SQL query. No markdown formatting, no backticks, no comments, no explanation.
- Must be a SELECT query.
- Use valid SQLite syntax (strftime, CASE, etc.).
- Limit results to 50 rows maximum unless explicitly requested.

USER QUERY:
{query}

SQL QUERY:
"""
        generated_sql = ""
        client = None
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                generated_sql = (response.text or "").strip()
                # Clean any markdown code blocks
                generated_sql = re.sub(r"^```(sql)?", "", generated_sql, flags=re.IGNORECASE).strip("`").strip()
            except Exception as err:
                generated_sql = ""

        # Fallback heuristic queries if LLM unavailable or failed
        if not generated_sql:
            q_lower = query.lower()
            if "employee" in q_lower or "salary" in q_lower or "headcount" in q_lower:
                if "department" in q_lower:
                    generated_sql = "SELECT department, COUNT(*) as headcount, AVG(salary) as avg_salary, MAX(salary) as max_salary FROM employees GROUP BY department ORDER BY headcount DESC"
                else:
                    generated_sql = "SELECT id, name, department, role, salary, retention_risk FROM employees ORDER BY salary DESC LIMIT 15"
            elif "sales" in q_lower or "transaction" in q_lower or "revenue" in q_lower:
                if "region" in q_lower:
                    generated_sql = "SELECT region, COUNT(*) as total_deals, SUM(amount) as total_revenue, AVG(amount) as avg_deal FROM sales_transactions GROUP BY region ORDER BY total_revenue DESC"
                elif "product" in q_lower:
                    generated_sql = "SELECT product, COUNT(*) as deals, SUM(amount) as total_revenue FROM sales_transactions GROUP BY product ORDER BY total_revenue DESC"
                else:
                    generated_sql = "SELECT transaction_id, date, product, region, customer_name, amount, sales_rep FROM sales_transactions ORDER BY amount DESC LIMIT 15"
            elif "ticket" in q_lower or "support" in q_lower or "csat" in q_lower:
                generated_sql = "SELECT issue_type, priority, COUNT(*) as count, AVG(resolution_time_hours) as avg_resolution_hrs, AVG(customer_satisfaction) as avg_csat FROM support_tickets GROUP BY issue_type"
            else:
                generated_sql = "SELECT * FROM financial_metrics ORDER BY quarter"

        df, err = enterprise_db.execute_query(generated_sql)

        return {
            "sql": generated_sql,
            "data": df,
            "error": err,
            "row_count": len(df) if df is not None else 0
        }

    def execute_ml_prediction(self, query: str) -> Dict[str, Any]:
        """Parse query intent and trigger Scikit-learn predictive models."""
        q = query.lower()
        if "retention" in q or "attrition" in q or "employee" in q:
            # Predict employee retention risk
            # Extract salary if mentioned (e.g. 95000 or 95k)
            salary_match = re.search(r"(\d{2,3})(?:k|,000)?", q)
            salary = float(salary_match.group(1)) * 1000 if salary_match else 110000.0
            
            res = enterprise_ml.predict_retention_risk(
                salary=salary,
                performance_rating=4.2,
                remote_days=2,
                tenure_years=2.5,
                overtime_hours_monthly=15.0
            )
            return {"type": "retention", "result": res}
        else:
            # Predict revenue
            marketing = 4500.0
            sales_reps = 35
            csat = 4.4
            discount = 0.08
            quarter = 2
            
            res = enterprise_ml.predict_revenue(
                marketing_spend_k=marketing,
                sales_reps=sales_reps,
                customer_csat=csat,
                discount_pct=discount,
                quarter=quarter
            )
            return {"type": "revenue", "result": res}

    def execute_graph_query(self, query: str) -> Dict[str, Any]:
        """Perform GraphRAG retrieval on enterprise knowledge graph."""
        res = enterprise_kg.search_subgraph(query)
        context_str = enterprise_kg.format_graph_context_for_prompt(query)
        return {
            "context_str": context_str,
            "matched_entities": res["matched_entities"],
            "connected_entities": res["connected_entities"],
            "triples": res["triples"]
        }

    def answer_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Omni-channel intelligent agent answering function.
        Routes query to optimal tool, retrieves context, and produces synthesis.
        """
        if not api_key:
            api_key = get_gemini_api_key()

        intent = self.classify_intent(query)
        tool_used = intent
        
        doc_context = ""
        doc_sources = []
        graph_data = None
        sql_data = None
        ml_data = None
        web_context = ""

        # 1. Execute corresponding tool pipelines
        if intent == "SQL_ANALYTICS":
            sql_data = self.execute_text_to_sql(query, api_key)
        elif intent == "ML_PREDICTION":
            ml_data = self.execute_ml_prediction(query)
        elif intent == "GRAPH_RAG":
            graph_data = self.execute_graph_query(query)
        elif intent == "WEB_SEARCH":
            try:
                results = DDGS().text(query, max_results=4)
                snippets = [f"- {r.get('title')}: {r.get('body')}" for r in results]
                web_context = "\n".join(snippets)
            except Exception:
                web_context = "Web search service temporarily unavailable."
        else:
            # DOCUMENT_RAG
            doc_context, doc_sources = vector_store.retrieve(query)

        # 2. Generate grounded LLM synthesis if Gemini is available
        client = None
        answer_text = ""
        
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
            except Exception as err:
                client = None

        if client:
            # Build unified agent prompt
            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation_history[-6:]])

            supplemental_context = ""
            if doc_context:
                supplemental_context += f"\nDOCUMENT RAG SOURCES:\n{doc_context}\n"
            if graph_data:
                supplemental_context += f"\n{graph_data['context_str']}\n"
            if sql_data and sql_data["data"] is not None:
                supplemental_context += f"\nSQL QUERY EXECUTED: {sql_data['sql']}\nSQL DATA RESULTS:\n{sql_data['data'].to_string(index=False)}\n"
            if ml_data:
                supplemental_context += f"\nML PREDICTION ENGINE OUTPUT:\n{json.dumps(ml_data['result'], indent=2)}\n"
            if web_context:
                supplemental_context += f"\nWEB RESEARCH SNIPPETS:\n{web_context}\n"

            system_instructions = """
You are the Enterprise Knowledge & Data Intelligence Assistant for Nexora Technologies.
You have access to corporate documents, SQL business databases, GraphRAG relationship models, and ML predictive tools.

RULES:
- Provide clear, highly professional, executive-ready answers.
- Cite specific policies, numbers, metrics, or relationships when provided in the context.
- If referencing data results or predictions, present them clearly with bullet points or summary tables.
- Do not invent facts that are not present in the provided context.
- If the user asks a question about internal company data that isn't found, state that the information was not located in active corporate records.
"""

            user_prompt = f"""
{system_instructions}

CONVERSATION HISTORY:
{history_text if history_text else "No prior conversation."}

INTELLIGENCE CONTEXT RETRIEVED ({intent}):
{supplemental_context if supplemental_context else "No external context retrieved."}

USER QUESTION:
{query}

COMPREHENSIVE ANSWER:
"""
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config={"temperature": 0.2, "max_output_tokens": 1500}
                )
                answer_text = (response.text or "").strip()
            except Exception as err:
                answer_text = f"Gemini API request note: {err}. Displaying direct tool analysis below."

        # Fallback deterministic answer synthesis if LLM was unavailable
        if not answer_text:
            if sql_data and sql_data["data"] is not None:
                answer_text = f"**Executed SQL Query:** `{sql_data['sql']}`\n\nRetrieved **{len(sql_data['data'])}** records from the enterprise database. View the structured table results below."
            elif ml_data:
                res = ml_data["result"]
                if ml_data["type"] == "revenue":
                    answer_text = f"**Revenue Forecast Simulation:**\n- **Projected Revenue:** **${res['forecasted_revenue_m']}M**\n- **Confidence Range (95%):** ${res['confidence_interval'][0]}M - ${res['confidence_interval'][1]}M\n- **Top Drivers:** {', '.join([f'{k} ({v*100:.1f}%)' for k, v in res['top_drivers'][:3]])}"
                else:
                    answer_text = f"**Employee Retention Risk Analysis:**\n- **Predicted Risk Category:** **{res['predicted_risk']}**\n- **Probabilities:** Low: {res['probabilities'].get('Low', 0)*100:.1f}%, Medium: {res['probabilities'].get('Medium', 0)*100:.1f}%, High: {res['probabilities'].get('High', 0)*100:.1f}%\n- **Top Contributing Factors:** {', '.join([f'{k} ({v*100:.1f}%)' for k, v in res['key_factors'][:3]])}"
            elif graph_data:
                entities_str = ", ".join(graph_data["matched_entities"])
                answer_text = f"**GraphRAG Knowledge Traversal:**\nFound {len(graph_data['triples'])} relational links connecting entities `{entities_str}` across company departments and governance rules.\n\n" + "\n".join([f"- **{t['source']}** *({t['relation']})* ➔ **{t['target']}**: {t['description']}" for t in graph_data['triples'][:6]])
            elif doc_sources:
                top = doc_sources[0]
                answer_text = f"**Document Intelligence Result:**\nFound relevant guidance in **{top['source']}** under **{top['section']}** (Similarity: {top['score']:.4f}):\n\n{top['content']}"
            else:
                answer_text = "I processed your request, but could not locate relevant records in the corporate knowledge base. Please check the query or upload additional documentation."

        return {
            "answer": answer_text,
            "intent": intent,
            "tool_used": tool_used,
            "sources": doc_sources,
            "sql_result": sql_data,
            "ml_result": ml_data,
            "graph_result": graph_data,
            "web_context": web_context
        }

# Singleton instance
enterprise_agent = EnterpriseAgent()

"""
Enterprise Knowledge & Data Intelligence Assistant
Production-Ready Streamlit Application for Enterprise RAG, GraphRAG, SQL Analytics & ML Intelligence.
Author: Charmitha Reddy
"""

import sys
from pathlib import Path
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.config.settings import (
    DEFAULT_GEMINI_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_CHUNKS,
    MAX_UPLOAD_SIZE_MB,
    DOCS_DIR,
    get_gemini_api_key,
)
from src.database.enterprise_db import enterprise_db
from src.graph.knowledge_graph import enterprise_kg
from src.models.ml_engine import enterprise_ml
from src.ingestion.document_manager import DocumentManager
from src.retrieval.vector_store import vector_store
from src.analytics.visualizer import analytics
from src.agent.orchestrator import enterprise_agent


# ============================================================
# 1. PAGE SETUP & STYLING
# ============================================================

st.set_page_config(
    page_title="Enterprise Knowledge & Data Intelligence Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark enterprise theme */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Top Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #1e3a8a 50%, #0f172a 100%);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-banner p {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0;
    }

    /* Metric Card Pills */
    .metric-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-pill .label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-pill .value {
        font-size: 20px;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
    }
    .badge-sql { background: #065f46; color: #6ee7b7; border: 1px solid #059669; }
    .badge-rag { background: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; }
    .badge-graph { background: #581c87; color: #d8b4fe; border: 1px solid #7e22ce; }
    .badge-ml { background: #7c2d12; color: #fdba74; border: 1px solid #c2410c; }
    .badge-web { background: #155e75; color: #67e8f9; border: 1px solid #0891b2; }

    /* Source Citation Card */
    .source-card {
        background-color: #111827;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 13px;
    }

    /* Streamlit overrides */
    div[data-testid="stChatMessage"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. STATE & KNOWLEDGE BASE INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner="Initializing Enterprise Knowledge Base...")
def load_and_index_documents():
    """Load preloaded enterprise documents and build vector index."""
    docs = DocumentManager.load_preloaded_documents(DOCS_DIR)
    all_chunks = []
    for doc in docs:
        chunks = DocumentManager.chunk_document(doc["text"], doc["name"])
        all_chunks.extend(chunks)
    vector_store.build_index(all_chunks)
    return docs, all_chunks

docs_list, chunks_indexed = load_and_index_documents()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the Nexora Enterprise Intelligence Assistant.**\n\n"
                "I am your unified AI copilot across corporate knowledge, structured business data, and predictive intelligence:\n\n"
                "- 📄 **Document RAG:** Search corporate policies, audited Q4 earnings, IT compliance, and SLAs.\n"
                "- 📊 **Text-to-SQL Analytics:** Query live employee records, sales performance, and customer support tickets.\n"
                "- 🕸️ **GraphRAG:** Explore interconnected organizational relationships, governance rules, and incident escalation paths.\n"
                "- 🤖 **Predictive ML:** Forecast quarterly revenue and evaluate employee retention risks with scenario simulation.\n\n"
                "Click any sample question below or type your query to begin!"
            ),
            "tool_used": "SYSTEM",
            "sql_result": None,
            "sources": [],
            "graph_result": None,
            "ml_result": None,
            "web_context": ""
        }
    ]


# ============================================================
# 3. SIDEBAR NAVIGATION & TELEMETRY
# ============================================================

with st.sidebar:
    st.markdown("### 🏢 Nexora Intelligence OS")
    st.caption("Enterprise Knowledge, Graph & Analytics Engine")
    st.divider()

    # System Telemetry Pills
    st.markdown("##### ⚙️ System Telemetry")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Documents", len(docs_list))
        st.metric("SQL Tables", len(enterprise_db.get_table_names()))
    with col_t2:
        st.metric("Vectors", len(chunks_indexed))
        st.metric("Graph Nodes", enterprise_kg.get_summary_metrics()["total_nodes"])

    st.divider()

    # Document Uploader
    st.markdown("##### 📤 Add Custom Knowledge")
    uploaded_files = st.file_uploader(
        "Upload PDF, Word, TXT, or CSV",
        type=["pdf", "docx", "txt", "csv", "md"],
        accept_multiple_files=True,
        help="Upload company documents to index dynamically into the active session."
    )

    if uploaded_files:
        new_chunks = []
        for uf in uploaded_files:
            try:
                text = DocumentManager.read_file_content(uf, uf.name)
                chunks = DocumentManager.chunk_document(text, uf.name)
                new_chunks.extend(chunks)
            except Exception as e:
                st.error(f"Error reading {uf.name}: {e}")

        if new_chunks:
            combined = chunks_indexed + new_chunks
            vector_store.build_index(combined)
            st.success(f"Indexed {len(new_chunks)} new chunks from {len(uploaded_files)} file(s)!")

    st.divider()

    # Quick Demo Prompts
    st.markdown("##### 💡 Quick Demo Prompts")
    sample_queries = [
        "What is our refund policy and SLA for critical customer issues?",
        "Show total revenue by region and top 5 sales deals",
        "What policies govern remote work and zero-trust security for engineering?",
        "Predict Q2 sales revenue if marketing spend increases to $5.5M",
        "Which employees have high retention risk and what are their salaries?"
    ]
    for sq in sample_queries:
        if st.button(f"👉 {sq[:38]}...", key=f"btn_{hash(sq)}", use_container_width=True):
            st.session_state.pending_query = sq
            st.rerun()

    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Author: **Charmitha Reddy**")
    st.caption("Architecture: Python • Streamlit • FastEmbed • NetworkX • SQLite • Scikit-learn • Gemini")


# ============================================================
# 4. MAIN INTERFACE & TABS
# ============================================================

st.markdown("""
<div class="hero-banner">
    <h1><span>⚡</span> Enterprise Knowledge & Data Intelligence Assistant</h1>
    <p>Unified enterprise intelligence: Natural Language Document RAG, GraphRAG Ontologies, Text-to-SQL Analytics & Scikit-learn Predictive Forecasting.</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_rag, tab_graph, tab_sql, tab_ml, tab_arch = st.tabs([
    "💬 Unified AI Assistant",
    "📄 Document RAG Studio",
    "🕸️ Enterprise GraphRAG",
    "📊 SQL & Business Intelligence",
    "🤖 ML Predictive Analytics",
    "📐 System Architecture"
])


# ============================================================
# TAB 1: UNIFIED AI ASSISTANT (OMNI-CHAT)
# ============================================================

with tab_chat:
    st.markdown("#### Natural Language Enterprise Orchestrator")
    st.caption("Intelligently routes questions across Vector RAG, GraphRAG, SQL Database, and ML models.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            tool_used = msg.get("tool_used", "GENERAL")
            
            # Badge Tag
            if tool_used == "SQL_ANALYTICS":
                st.markdown('<span class="badge badge-sql">📊 SQL Analytics Executed</span>', unsafe_allow_html=True)
            elif tool_used == "DOCUMENT_RAG":
                st.markdown('<span class="badge badge-rag">📄 Document RAG Grounded</span>', unsafe_allow_html=True)
            elif tool_used == "GRAPH_RAG":
                st.markdown('<span class="badge badge-graph">🕸️ GraphRAG Traversal</span>', unsafe_allow_html=True)
            elif tool_used == "ML_PREDICTION":
                st.markdown('<span class="badge badge-ml">🤖 ML Model Prediction</span>', unsafe_allow_html=True)
            elif tool_used == "WEB_SEARCH":
                st.markdown('<span class="badge badge-web">🌐 Web Research</span>', unsafe_allow_html=True)

            st.markdown(msg["content"])

            # SQL Data Table View
            sql_res = msg.get("sql_result")
            if sql_res and sql_res.get("data") is not None and not sql_res["data"].empty:
                with st.expander(f"🔍 View Executed SQL & Results ({sql_res['row_count']} rows)", expanded=False):
                    st.code(sql_res["sql"], language="sql")
                    st.dataframe(sql_res["data"], use_container_width=True)

            # Document Sources View
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} Supporting Document Sources", expanded=False):
                    for idx, s in enumerate(sources, 1):
                        st.markdown(f"**[{idx}] {s['source']} ➔ {s['section']}** (Similarity: `{s['score']:.4f}`)")
                        st.caption(s["content"])

            # Graph Triples View
            graph_res = msg.get("graph_result")
            if graph_res and graph_res.get("triples"):
                with st.expander(f"🕸️ View {len(graph_res['triples'])} Knowledge Graph Relationships", expanded=False):
                    for t in graph_res["triples"]:
                        st.markdown(f"- **{t['source']}** `[{t['relation']}]` ➔ **{t['target']}**: *{t['description']}*")

            # ML Output View
            ml_res = msg.get("ml_result")
            if ml_res:
                with st.expander("🤖 View ML Model Output & Feature Weights", expanded=False):
                    st.json(ml_res["result"])

    # Chat Input Box
    user_input = st.chat_input("Ask any question about company documents, business data, policies, or predictions...")

    # Handle pending query from quick prompts
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query and coordinating enterprise tools..."):
                response_dict = enterprise_agent.answer_query(
                    query=user_input,
                    conversation_history=st.session_state.messages
                )

                tool_used = response_dict["tool_used"]
                if tool_used == "SQL_ANALYTICS":
                    st.markdown('<span class="badge badge-sql">📊 SQL Analytics Executed</span>', unsafe_allow_html=True)
                elif tool_used == "DOCUMENT_RAG":
                    st.markdown('<span class="badge badge-rag">📄 Document RAG Grounded</span>', unsafe_allow_html=True)
                elif tool_used == "GRAPH_RAG":
                    st.markdown('<span class="badge badge-graph">🕸️ GraphRAG Traversal</span>', unsafe_allow_html=True)
                elif tool_used == "ML_PREDICTION":
                    st.markdown('<span class="badge badge-ml">🤖 ML Model Prediction</span>', unsafe_allow_html=True)
                elif tool_used == "WEB_SEARCH":
                    st.markdown('<span class="badge badge-web">🌐 Web Research</span>', unsafe_allow_html=True)

                st.markdown(response_dict["answer"])

                # Display SQL results if available
                sql_res = response_dict.get("sql_result")
                if sql_res and sql_res.get("data") is not None and not sql_res["data"].empty:
                    with st.expander(f"🔍 View Executed SQL & Results ({sql_res['row_count']} rows)", expanded=True):
                        st.code(sql_res["sql"], language="sql")
                        st.dataframe(sql_res["data"], use_container_width=True)

                # Display document citations
                sources = response_dict.get("sources", [])
                if sources:
                    with st.expander(f"📚 View {len(sources)} Supporting Document Sources", expanded=True):
                        for idx, s in enumerate(sources, 1):
                            st.markdown(f"**[{idx}] {s['source']} ➔ {s['section']}** (Similarity: `{s['score']:.4f}`)")
                            st.caption(s["content"])

                # Display graph results
                graph_res = response_dict.get("graph_result")
                if graph_res and graph_res.get("triples"):
                    with st.expander(f"🕸️ View {len(graph_res['triples'])} Knowledge Graph Relationships", expanded=True):
                        for t in graph_res["triples"]:
                            st.markdown(f"- **{t['source']}** `[{t['relation']}]` ➔ **{t['target']}**: *{t['description']}*")

                # Display ML results
                ml_res = response_dict.get("ml_result")
                if ml_res:
                    with st.expander("🤖 View ML Model Output & Feature Weights", expanded=True):
                        st.json(ml_res["result"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_dict["answer"],
            "tool_used": response_dict["tool_used"],
            "sql_result": response_dict.get("sql_result"),
            "sources": response_dict.get("sources", []),
            "graph_result": response_dict.get("graph_result"),
            "ml_result": response_dict.get("ml_result"),
            "web_context": response_dict.get("web_context", "")
        })


# ============================================================
# TAB 2: DOCUMENT RAG STUDIO
# ============================================================

with tab_rag:
    st.markdown("#### Document RAG & Semantic Retrieval Studio")
    st.caption("Inspect vector embeddings, semantic chunking, and similarity scores across active enterprise documents.")

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.markdown("##### 📁 Preloaded Documents")
        for d in docs_list:
            with st.container(border=True):
                st.markdown(f"📄 **{d['name']}**")
                st.caption(f"Length: {len(d['text']):,} characters")
                st.download_button(
                    label="Download Document",
                    data=d['text'],
                    file_name=d['name'],
                    mime="text/plain",
                    key=f"dl_{d['name']}"
                )

    with col_r2:
        st.markdown("##### 🔎 Live Semantic Retrieval Inspector")
        test_query = st.text_input("Enter test query for vector similarity retrieval:", value="What are the SLA penalty credits?")
        top_k_slider = st.slider("Select Top-K Chunks:", min_value=1, max_value=8, value=3)
        
        if test_query:
            ctx, chunks = vector_store.retrieve(test_query, top_k=top_k_slider)
            st.markdown(f"**Retrieved {len(chunks)} Chunks:**")
            for i, c in enumerate(chunks, 1):
                with st.container(border=True):
                    col_c1, col_c2 = st.columns([3, 1])
                    with col_c1:
                        st.markdown(f"**[{i}] {c['section']}** (`{c['source']}`)")
                    with col_c2:
                        st.metric("Similarity", f"{c['score']:.4f}")
                    st.text(c["content"])


# ============================================================
# TAB 3: ENTERPRISE GRAPHRAG
# ============================================================

with tab_graph:
    st.markdown("#### Enterprise Knowledge Graph (GraphRAG)")
    st.caption("Topological relationship modeling between Departments, Governance Policies, Products, and Compliance Standards.")

    kg_metrics = enterprise_kg.get_summary_metrics()
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Graph Nodes", kg_metrics["total_nodes"])
    c_m2.metric("Relational Edges", kg_metrics["total_edges"])
    c_m3.metric("Graph Density", f"{kg_metrics['density']:.4f}")

    # Interactive Graph Visualization
    nodes_df, edges_df = enterprise_kg.get_graph_visualization_df()
    graph_chart = analytics.chart_knowledge_graph(nodes_df, edges_df)
    if graph_chart:
        st.altair_chart(graph_chart, use_container_width=True)

    st.divider()

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### 🔍 Subgraph Neighborhood Search")
        entity_query = st.text_input("Enter entity keyword (e.g., Security, Remote Work, Sev-1):", value="Security")
        if entity_query:
            subgraph_res = enterprise_kg.search_subgraph(entity_query)
            st.markdown(f"Found **{len(subgraph_res['triples'])}** connected relationships:")
            for t in subgraph_res["triples"]:
                st.markdown(f"- **{t['source']}** ➔ `{t['relation']}` ➔ **{t['target']}**")
                st.caption(f"*{t['description']}*")

    with col_g2:
        st.markdown("##### 📍 Governance Path Finder")
        all_nodes = sorted(list(enterprise_kg.graph.nodes()))
        source_node = st.selectbox("Source Entity:", all_nodes, index=all_nodes.index("Engineering") if "Engineering" in all_nodes else 0)
        target_node = st.selectbox("Target Entity:", all_nodes, index=all_nodes.index("Incident Response Protocol") if "Incident Response Protocol" in all_nodes else 1)

        if st.button("Calculate Dependency Path"):
            path = enterprise_kg.find_shortest_path(source_node, target_node)
            if path:
                st.success(f"**Path Found ({len(path) - 1} hops):**")
                st.markdown(" ➔ ".join([f"`{node}`" for node in path]))
            else:
                st.warning("No direct path found between the selected entities.")


# ============================================================
# TAB 4: SQL & BUSINESS INTELLIGENCE
# ============================================================

with tab_sql:
    st.markdown("#### Structured Business Data & SQL Analytics")
    st.caption("Interact with enterprise relational data across Employees, Sales Deals, Support Tickets, and Financials.")

    # Executive KPIs
    sales_df = enterprise_db.get_table_df("sales_transactions", limit=500)
    employee_df = enterprise_db.get_table_df("employees", limit=500)
    tickets_df = enterprise_db.get_table_df("support_tickets", limit=500)
    financial_df = enterprise_db.get_table_df("financial_metrics", limit=500)

    kpis = analytics.get_executive_kpis(sales_df, employee_df, tickets_df)

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Total Tracked Sales", kpis["total_revenue"], help="Sum of recorded deals in sales_transactions")
    col_k2.metric("Avg Deal Size", kpis["avg_deal_size"])
    col_k3.metric("Support Resolution Time", kpis["avg_resolution_hrs"])
    col_k4.metric("Avg Customer CSAT", kpis["avg_csat"])

    # Visualizations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        rev_chart = analytics.chart_revenue_and_income(financial_df)
        if rev_chart:
            st.altair_chart(rev_chart, use_container_width=True)
    with col_v2:
        sales_chart = analytics.chart_sales_by_region_and_product(sales_df)
        if sales_chart:
            st.altair_chart(sales_chart, use_container_width=True)

    st.divider()

    # Interactive SQL Query Console
    st.markdown("##### 💻 Interactive Enterprise SQL Console")
    default_sql = "SELECT department, COUNT(*) as headcount, AVG(salary) as avg_salary, MAX(salary) as max_salary FROM employees GROUP BY department ORDER BY avg_salary DESC"
    sql_input = st.text_area("Write SQL SELECT Query:", value=default_sql, height=80)

    if st.button("Run SQL Query", type="primary"):
        df_res, err = enterprise_db.execute_query(sql_input)
        if err:
            st.error(f"SQL Error: {err}")
        else:
            st.success(f"Query returned {len(df_res)} rows successfully.")
            st.dataframe(df_res, use_container_width=True)

    with st.expander("📁 View Database Tables & Schemas"):
        tables = enterprise_db.get_table_names()
        selected_tbl = st.selectbox("Select table to preview:", tables)
        st.dataframe(enterprise_db.get_table_df(selected_tbl, limit=15), use_container_width=True)


# ============================================================
# TAB 5: ML PREDICTIVE ANALYTICS
# ============================================================

with tab_ml:
    st.markdown("#### Scikit-learn Machine Learning Engines")
    st.caption("Interactive scenario simulation for Enterprise Revenue Forecasting and Employee Retention Risk.")

    ml_col1, ml_col2 = st.columns(2)

    # Simulator 1: Revenue Forecast
    with ml_col1:
        with st.container(border=True):
            st.markdown("##### 📈 Quarterly Revenue Forecaster")
            st.caption("Random Forest / Decision Tree trained on marketing, sales reps, CSAT, and seasonality.")

            m_spend = st.slider("Marketing Spend ($ in thousands):", min_value=1500, max_value=9000, value=5000, step=250)
            s_reps = st.slider("Active Enterprise Sales Reps:", min_value=15, max_value=70, value=40)
            c_csat = st.slider("Customer CSAT Score (1-5):", min_value=3.0, max_value=5.0, value=4.5, step=0.1)
            d_pct = st.slider("Average Discount Rate (%):", min_value=0.0, max_value=25.0, value=8.0, step=1.0) / 100.0
            q_num = st.selectbox("Fiscal Quarter:", [1, 2, 3, 4], index=3, format_func=lambda x: f"Q{x} Seasonality")

            rev_pred = enterprise_ml.predict_revenue(m_spend, s_reps, c_csat, d_pct, q_num)

            st.metric(
                label="Forecasted Quarterly Revenue",
                value=f"${rev_pred['forecasted_revenue_m']}M",
                delta=f"Range: ${rev_pred['confidence_interval'][0]}M - ${rev_pred['confidence_interval'][1]}M"
            )

            # Feature Importance
            rev_feat_chart = analytics.chart_feature_importance(
                enterprise_ml.revenue_metrics["feature_importance"],
                title="Revenue Drivers Importance"
            )
            st.altair_chart(rev_feat_chart, use_container_width=True)

    # Simulator 2: Retention Risk
    with ml_col2:
        with st.container(border=True):
            st.markdown("##### 🛡️ Employee Retention Risk Scorer")
            st.caption("Evaluates attrition risk probability and key organizational contributing factors.")

            e_sal = st.slider("Annual Salary ($):", min_value=60000, max_value=200000, value=95000, step=5000)
            e_rate = st.slider("Performance Rating (1.0 - 5.0):", min_value=2.5, max_value=5.0, value=4.2, step=0.1)
            e_rem = st.slider("Remote Days / Week:", min_value=0, max_value=3, value=1)
            e_ten = st.slider("Tenure at Company (Years):", min_value=0.5, max_value=6.0, value=2.0, step=0.5)
            e_ot = st.slider("Monthly Overtime (Hours):", min_value=0, max_value=50, value=22)

            ret_pred = enterprise_ml.predict_retention_risk(e_sal, e_rate, e_rem, e_ten, e_ot)

            risk_class = ret_pred["predicted_risk"]
            color = "#ef4444" if risk_class == "High" else ("#f59e0b" if risk_class == "Medium" else "#10b981")
            
            st.markdown(f"**Predicted Retention Risk:** <span style='font-size:22px; font-weight:700; color:{color}'>{risk_class}</span>", unsafe_allow_html=True)
            st.json(ret_pred["probabilities"])

            ret_feat_chart = analytics.chart_feature_importance(
                enterprise_ml.retention_metrics["feature_importance"],
                title="Retention Risk Contributing Factors"
            )
            st.altair_chart(ret_feat_chart, use_container_width=True)


# ============================================================
# TAB 6: SYSTEM ARCHITECTURE & PORTFOLIO
# ============================================================

with tab_arch:
    st.markdown("#### Enterprise Platform Architecture & Technical Design")
    
    st.markdown("""
    ```mermaid
    flowchart TD
        User([User Natural Language Prompt]) --> Router[AI Agent Orchestrator]
        
        Router -->|Intent: Document Facts| RAG[Vector RAG Engine\nFastEmbed + Cosine Similarity]
        Router -->|Intent: Governance & Relationships| GraphRAG[Enterprise GraphRAG\nNetworkX Ontology]
        Router -->|Intent: Relational Analytics| SQL[Text-to-SQL Analytics\nSQLite Engine]
        Router -->|Intent: Predictive Forecasting| ML[Scikit-learn ML Engine\nDecision Trees & Scenarios]
        Router -->|Intent: Current News| Web[DuckDuckGo Search]
        
        RAG --> Synthesizer[Gemini 2.5 LLM\nSynthesis & Verification]
        GraphRAG --> Synthesizer
        SQL --> Synthesizer
        ML --> Synthesizer
        Web --> Synthesizer
        
        Synthesizer --> Output[Grounded Answer + Citations + Interactive Visualizations]
    ```
    """)

    st.divider()

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("##### 🛠️ Core Technologies")
        st.markdown("""
        - **Language & Runtime:** Python 3.13 / Streamlit
        - **Embedding Model:** `BAAI/bge-small-en-v1.5` via FastEmbed
        - **Knowledge Graph:** NetworkX MultiDiGraph (Topology & GraphRAG)
        - **Relational Data:** SQLite (Text-to-SQL & Schema Introspection)
        - **Predictive ML:** Scikit-learn (DecisionTree Regressors & Classifiers)
        - **Visualizations:** Altair / Vega-Lite Interactive Charts
        - **LLM Reasoning:** Google Gemini API (`gemini-2.5-flash`)
        """)

    with col_a2:
        st.markdown("##### 👨‍💻 Project Attribution")
        st.markdown("""
        - **Author:** Charmitha Reddy
        - **Repository:** [Enterprise-Knowledge-Data-Intelligence-Assistant](https://github.com/charmithareddy/Enterprise-Knowledge-Data-Intelligence-Assistant)
        - **Status:** Complete & Portfolio Ready
        """)

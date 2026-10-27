"""
Knowledge Graph & GraphRAG Module for Enterprise Relationship Intelligence.
"""

from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import pandas as pd
import numpy as np

class EnterpriseKnowledgeGraph:
    """Enterprise GraphRAG engine representing interconnected entities, policies, products, and rules."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._build_default_enterprise_graph()

    def _build_default_enterprise_graph(self):
        """Construct the foundational enterprise ontology for Nexora Technologies."""
        # 1. Add Nodes with entity types and metadata
        nodes = [
            # Departments
            ("Engineering", {"type": "Department", "head": "VP of Engineering", "color": "#3b82f6"}),
            ("Security", {"type": "Department", "head": "CISO Marcus Vance", "color": "#ef4444"}),
            ("Customer Support", {"type": "Department", "head": "Tara Patel", "color": "#10b981"}),
            ("Sales", {"type": "Department", "head": "James Wilson", "color": "#f59e0b"}),
            ("Finance", {"type": "Department", "head": "Zoe Takahashi", "color": "#8b5cf6"}),
            ("Legal & Compliance", {"type": "Department", "head": "General Counsel", "color": "#ec4899"}),

            # Policies & Directives
            ("Remote Work Policy", {"type": "Policy", "rule": "Max 2 days/wk remote, requires manager approval", "color": "#06b6d4"}),
            ("Data Security & SOC2 Policy", {"type": "Policy", "rule": "Zero-trust, AES-256 at rest, TLS 1.3 in transit", "color": "#06b6d4"}),
            ("Refund Policy", {"type": "Policy", "rule": "30 days window, 5-7 business days processing", "color": "#06b6d4"}),
            ("Support SLA Directive", {"type": "Policy", "rule": "Sev-1: <30min, Sev-2: <2hr, Sev-3: <8hr", "color": "#06b6d4"}),
            ("Incident Response Protocol", {"type": "Policy", "rule": "15min initial response, 1hr CISO & Legal notice", "color": "#06b6d4"}),
            ("Annual Leave Policy", {"type": "Policy", "rule": "24 days paid leave/year, 5-day advance notice", "color": "#06b6d4"}),

            # Products
            ("EnterpriseAI", {"type": "Product", "revenue": "$41.5M ARR", "margin": "82.1%", "color": "#6366f1"}),
            ("CloudPlatform", {"type": "Product", "revenue": "$68.2M ARR", "uptime_sla": "99.95%", "color": "#6366f1"}),
            ("SecuritySuite", {"type": "Product", "revenue": "$21.1M ARR", "feature": "Zero-trust posture", "color": "#6366f1"}),
            ("DataHub", {"type": "Product", "revenue": "$12.0M ARR", "connectors": "Multi-cloud ETL", "color": "#6366f1"}),

            # Compliance & Standards
            ("SOC 2 Type II", {"type": "Standard", "frequency": "Annual third-party audit", "color": "#14b8a6"}),
            ("GDPR / CCPA", {"type": "Standard", "requirement": "DSAR & right to forget within 30 days", "color": "#14b8a6"}),
            ("FIDO2 Hardware MFA", {"type": "Standard", "enforcement": "Mandatory for production infrastructure", "color": "#14b8a6"}),
            ("AES-256 Encryption", {"type": "Standard", "scope": "All tier-3 confidential & tier-4 restricted data", "color": "#14b8a6"}),

            # Incident Severities
            ("Sev-1 Critical Incident", {"type": "Incident", "sla": "<30 mins response, 1hr exec alert", "color": "#dc2626"}),
            ("Sev-2 High Incident", {"type": "Incident", "sla": "<2 hours response, 12hr fix", "color": "#ea580c"}),
            ("Sev-3 Medium Incident", {"type": "Incident", "sla": "<8 hours response", "color": "#eab308"})
        ]

        for node_id, attrs in nodes:
            self.graph.add_node(node_id, **attrs)

        # 2. Add Edges with relation types and details
        edges = [
            # Department -> Policy / Standards
            ("Engineering", "Remote Work Policy", {"relation": "APPLIES_TO", "desc": "Employees can work remote up to 2 days/week"}),
            ("Engineering", "FIDO2 Hardware MFA", {"relation": "MANDATES", "desc": "Zero-trust biometric key required for production"}),
            ("Security", "Data Security & SOC2 Policy", {"relation": "GOVERNS", "desc": "Enforces least privilege and audit logging"}),
            ("Security", "Incident Response Protocol", {"relation": "ORCHESTRATES", "desc": "Leads 24/7 security triage and response bridge"}),
            ("Customer Support", "Support SLA Directive", {"relation": "EXECUTES", "desc": "Handles tier-1 to tier-3 customer inquiries"}),
            ("Customer Support", "Refund Policy", {"relation": "MANAGES", "desc": "Validates 30-day return eligibility"}),
            ("Finance", "Refund Policy", {"relation": "AUDITS", "desc": "Authorizes payouts and logs financial chargebacks"}),
            ("Legal & Compliance", "GDPR / CCPA", {"relation": "ENFORCES", "desc": "Ensures DSAR compliance within 30 days"}),

            # Products -> Compliance & Policies
            ("EnterpriseAI", "AES-256 Encryption", {"relation": "ENFORCES", "desc": "Envelope encryption on customer RAG embeddings"}),
            ("CloudPlatform", "Support SLA Directive", {"relation": "BOUND_BY", "desc": "Enterprise tier guarantees 99.95% uptime"}),
            ("SecuritySuite", "SOC 2 Type II", {"relation": "VALIDATES", "desc": "Continuous posture assessment against trust criteria"}),
            ("DataHub", "GDPR / CCPA", {"relation": "COMPLIES_WITH", "desc": "Automated data deletion pipeline"}),

            # Incidents -> Policies & Notification
            ("Sev-1 Critical Incident", "Incident Response Protocol", {"relation": "TRIGGERS", "desc": "Requires 15min response and 1hr CISO & Legal notice"}),
            ("Sev-1 Critical Incident", "Security", {"relation": "ESCALATES_TO", "desc": "Dedicated engineering bridge opened immediately"}),
            ("Sev-1 Critical Incident", "Legal & Compliance", {"relation": "NOTIFIES", "desc": "Mandatory breach reporting within 72 hours"}),
            ("Sev-2 High Incident", "Support SLA Directive", {"relation": "PRIORITIZES", "desc": "Engineering response SLA within 2 hours"}),

            # Cross-policy links
            ("Remote Work Policy", "Data Security & SOC2 Policy", {"relation": "REQUIRES", "desc": "Remote devices must use company VPN and FIDO2 MFA"}),
            ("Refund Policy", "Support SLA Directive", {"relation": "INTEGRATES_WITH", "desc": "Defective product claims prioritized within 4 hours"}),
            ("Annual Leave Policy", "Remote Work Policy", {"relation": "CROSS_REFERENCES", "desc": "Remote work cannot substitute for formal annual leave"})
        ]

        for source, target, attrs in edges:
            self.graph.add_edge(source, target, **attrs)

    def search_subgraph(self, query: str, max_hops: int = 1) -> Dict[str, Any]:
        """
        GraphRAG Retrieval: Given a natural language query, find relevant entities
        and traverse connected relationships.
        """
        query_lower = query.lower()
        matched_nodes = []

        # Find directly relevant seed nodes
        for node, data in self.graph.nodes(data=True):
            node_str = f"{node} {data.get('type', '')} {data.get('rule', '')} {data.get('desc', '')}".lower()
            if any(term in node_str for term in query_lower.split() if len(term) > 3):
                matched_nodes.append(node)

        # If no specific match, default to high-degree central nodes
        if not matched_nodes:
            degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
            matched_nodes = [d[0] for d in degrees[:3]]

        # Subgraph expansion
        subgraph_nodes = set(matched_nodes)
        for seed in matched_nodes:
            # Outgoing neighbors
            subgraph_nodes.update(self.graph.successors(seed))
            # Incoming neighbors
            subgraph_nodes.update(self.graph.predecessors(seed))

        sub_g = self.graph.subgraph(subgraph_nodes)
        
        # Format relational context
        triples = []
        for u, v, k, d in sub_g.edges(keys=True, data=True):
            relation = d.get('relation', 'CONNECTED_TO')
            desc = d.get('desc', '')
            triples.append({
                "source": u,
                "relation": relation,
                "target": v,
                "description": desc
            })

        return {
            "matched_entities": matched_nodes,
            "connected_entities": list(subgraph_nodes),
            "triples": triples,
            "subgraph": sub_g
        }

    def format_graph_context_for_prompt(self, query: str) -> str:
        """Format the retrieved graph triples into an informative context block for Gemini."""
        res = self.search_subgraph(query)
        if not res["triples"]:
            return ""

        lines = ["KNOWLEDGE GRAPH RELATIONSHIPS (GraphRAG Context):"]
        for t in res["triples"][:12]:
            lines.append(f"- [{t['source']}] --({t['relation']})--> [{t['target']}] : {t['description']}")
        return "\n".join(lines)

    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest dependency or governance path between two business concepts."""
        undirected = self.graph.to_undirected()
        try:
            return nx.shortest_path(undirected, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Return topological graph metrics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
            "node_types": list(set([data.get("type", "Unknown") for _, data in self.graph.nodes(data=True)]))
        }

    def get_graph_visualization_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate node and edge DataFrames with 2D coordinates for interactive visualization.
        """
        pos = nx.spring_layout(self.graph, seed=42, k=1.8)
        
        node_records = []
        for node, (x, y) in pos.items():
            data = self.graph.nodes[node]
            node_records.append({
                "entity": node,
                "x": float(x),
                "y": float(y),
                "type": data.get("type", "Entity"),
                "color": data.get("color", "#3b82f6"),
                "degree": self.graph.degree[node]
            })
        nodes_df = pd.DataFrame(node_records)

        edge_records = []
        for u, v, d in self.graph.edges(data=True):
            if u in pos and v in pos:
                edge_records.append({
                    "source": u,
                    "target": v,
                    "x_start": float(pos[u][0]),
                    "y_start": float(pos[u][1]),
                    "x_end": float(pos[v][0]),
                    "y_end": float(pos[v][1]),
                    "relation": d.get("relation", "RELATED"),
                    "desc": d.get("desc", "")
                })
        edges_df = pd.DataFrame(edge_records)

        return nodes_df, edges_df

# Singleton instance
enterprise_kg = EnterpriseKnowledgeGraph()

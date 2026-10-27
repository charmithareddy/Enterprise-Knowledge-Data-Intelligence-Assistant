"""
Enterprise Analytics & Interactive Data Visualization Module.
Generates polished Altair charts compatible with Streamlit.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import altair as alt

class EnterpriseAnalytics:
    """Calculates enterprise KPIs and generates presentation-ready visualizations."""

    @staticmethod
    def get_executive_kpis(sales_df: pd.DataFrame, employee_df: pd.DataFrame, tickets_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute high-level company metrics."""
        total_revenue = sales_df["amount"].sum() if not sales_df.empty else 0
        avg_deal_size = sales_df["amount"].mean() if not sales_df.empty else 0
        top_product = sales_df.groupby("product")["amount"].sum().idxmax() if not sales_df.empty else "N/A"
        
        total_headcount = len(employee_df) if not employee_df.empty else 0
        avg_salary = employee_df["salary"].mean() if not employee_df.empty else 0
        high_risk_count = len(employee_df[employee_df["retention_risk"] == "High"]) if not employee_df.empty else 0
        
        avg_resolution_hrs = tickets_df["resolution_time_hours"].mean() if not tickets_df.empty else 0
        avg_csat = tickets_df["customer_satisfaction"].mean() if not tickets_df.empty else 0
        resolved_pct = (len(tickets_df[tickets_df["status"] == "Resolved"]) / len(tickets_df) * 100) if not tickets_df.empty else 100.0

        return {
            "total_revenue": f"${total_revenue:,.0f}",
            "avg_deal_size": f"${avg_deal_size:,.0f}",
            "top_product": top_product,
            "total_headcount": total_headcount,
            "avg_salary": f"${avg_salary:,.0f}",
            "high_risk_employees": high_risk_count,
            "avg_resolution_hrs": f"{avg_resolution_hrs:.1f} hrs",
            "avg_csat": f"{avg_csat:.2f} / 5.0",
            "sla_resolution_pct": f"{resolved_pct:.1f}%"
        }

    @staticmethod
    def chart_revenue_and_income(financial_df: pd.DataFrame) -> alt.Chart:
        """Quarterly financial trajectory with Revenue and Net Income."""
        if financial_df.empty:
            return None

        melted = financial_df.melt(
            id_vars=["quarter"],
            value_vars=["revenue", "net_income", "operating_expenses"],
            var_name="Metric",
            value_name="Amount_M"
        )
        metric_labels = {
            "revenue": "Revenue ($M)",
            "net_income": "Net Income ($M)",
            "operating_expenses": "Operating Expenses ($M)"
        }
        melted["Metric"] = melted["Metric"].map(metric_labels)

        chart = alt.Chart(melted).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("quarter:N", title="Quarter", sort=None),
            y=alt.Y("Amount_M:Q", title="USD Millions ($M)"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#3b82f6", "#10b981", "#ef4444"])),
            tooltip=["quarter:N", "Metric:N", alt.Tooltip("Amount_M:Q", format="$.1f")]
        ).properties(
            title="Quarterly Enterprise Financial Trend (2024-2025)",
            height=320
        ).interactive()

        return chart

    @staticmethod
    def chart_sales_by_region_and_product(sales_df: pd.DataFrame) -> alt.Chart:
        """Sales volume grouped by Region and Product."""
        if sales_df.empty:
            return None

        chart = alt.Chart(sales_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("region:N", title="Sales Region"),
            y=alt.Y("sum(amount):Q", title="Total Sales ($)"),
            color=alt.Color("product:N", scale=alt.Scale(range=["#6366f1", "#06b6d4", "#ec4899", "#f59e0b"])),
            tooltip=["region:N", "product:N", alt.Tooltip("sum(amount):Q", format="$,.0f")]
        ).properties(
            title="Revenue Distribution by Region & Product Offering",
            height=320
        ).interactive()

        return chart

    @staticmethod
    def chart_department_salaries(employee_df: pd.DataFrame) -> alt.Chart:
        """Department salary distribution and average compensation."""
        if employee_df.empty:
            return None

        chart = alt.Chart(employee_df).mark_boxplot(extent="min-max").encode(
            x=alt.X("department:N", title="Department"),
            y=alt.Y("salary:Q", title="Annual Salary ($)"),
            color=alt.Color("department:N", legend=None),
            tooltip=["department:N", alt.Tooltip("mean(salary):Q", format="$,.0f")]
        ).properties(
            title="Departmental Salary Distributions",
            height=320
        ).interactive()

        return chart

    @staticmethod
    def chart_feature_importance(feature_dict: Dict[str, float], title: str = "Feature Importance") -> alt.Chart:
        """Horizontal bar chart for Scikit-learn model explainability."""
        df = pd.DataFrame([
            {"Feature": k.replace("_", " ").title(), "Importance": v}
            for k, v in feature_dict.items()
        ]).sort_values(by="Importance", ascending=True)

        chart = alt.Chart(df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#6366f1").encode(
            x=alt.X("Importance:Q", title="Relative Weight"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".3f")]
        ).properties(
            title=title,
            height=240
        )
        return chart

    @staticmethod
    def chart_knowledge_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> alt.Chart:
        """Interactive 2D topological visualization of Enterprise Knowledge Graph."""
        if nodes_df.empty:
            return None

        # Edges rule lines
        edges_chart = alt.Chart(edges_df).mark_rule(opacity=0.35, color="#94a3b8").encode(
            x="x_start:Q",
            y="y_start:Q",
            x2="x_end:Q",
            y2="y_end:Q",
            tooltip=["source:N", "relation:N", "target:N", "desc:N"]
        )

        # Nodes points
        nodes_chart = alt.Chart(nodes_df).mark_circle(size=380, opacity=0.9).encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("type:N", scale=alt.Scale(scheme="category10"), title="Entity Type"),
            tooltip=["entity:N", "type:N", "degree:Q"]
        )

        # Node labels
        text_chart = alt.Chart(nodes_df).mark_text(dy=-14, fontSize=11, fontWeight="bold", color="#f8fafc").encode(
            x="x:Q",
            y="y:Q",
            text="entity:N"
        )

        combined = (edges_chart + nodes_chart + text_chart).properties(
            title="Enterprise Knowledge Graph Topology (NetworkX + Vega-Lite)",
            height=460
        ).configure_view(strokeWidth=0)

        return combined

# Singleton instance
analytics = EnterpriseAnalytics()

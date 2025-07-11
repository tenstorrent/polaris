import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
from utils import _PROJECT_ROOT

st.set_page_config(page_title="Review Results", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cutive+Mono&display=swap');
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {
        font-family: 'Cutive Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Review Results")

if 'simulation_results' in st.session_state:
    # Load JSON data
    summary_file_path = os.path.join(_PROJECT_ROOT, "results", "CASE_0", "TEST_0", "SUMMARY", "study-summary.json")
    try:
        with open(summary_file_path, "r") as f:
            simulation_data = json.load(f)
            summary = simulation_data.get("summary", [])
    except FileNotFoundError:
        st.error("Summary file not found.")
        summary = []
    except json.JSONDecodeError:
        st.error("Error decoding JSON file.")
        summary = []

    if summary:
        # Filtering options
        wlcls_options = list(set(item["wlcls"] for item in summary))
        wlname_options = list(set(item["wlname"] for item in summary))

        selected_wlcls = st.selectbox("Filter by Workload Class", options=["All"] + wlcls_options, index=0)
        selected_wlname = st.selectbox("Filter by Workload Name", options=["All"] + wlname_options, index=0)

        filtered_summary = [
            item for item in summary
            if (selected_wlcls == "All" or item["wlcls"] == selected_wlcls) and
               (selected_wlname == "All" or item["wlname"] == selected_wlname)
        ]

        # Summary Table
        with st.expander("Summary Table"):
            st.subheader("Summary Table")
            summary_table = pd.DataFrame(
                [{
                    "Device Name": item["devname"],
                    "Workload Name": item["wlname"],
                    "Ideal Throughput": item["ideal_throughput"],
                    "Performance Projection": item["perf_projection"],
                } for item in filtered_summary]
            )
            st.table(summary_table)

        # Graph Visualization
        st.subheader("Graph Visualization")
        if not summary_table.empty:
            fig = px.bar(
                summary_table,
                x="Device Name",
                y=["Ideal Throughput", "Performance Projection"],
                color="Workload Name",
                barmode="group",
                title="Device Performance Metrics by Workload"
            )
            st.plotly_chart(fig)

        # Detailed View
        st.subheader("Detailed Metrics")
        for item in filtered_summary:
            with st.expander(f"Device: {item['devname']} - Workload: {item['wlname']}"):
                st.json(item)
    else:
        st.warning("No simulation data available.")
else:
    st.write("Results will be shown here after the simulation is complete.")

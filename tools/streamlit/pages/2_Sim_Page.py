import streamlit as st
import os
import subprocess
from utils import _PROJECT_ROOT, initApp

st.set_page_config(page_title="Simulation", layout="wide")

initApp()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cutive+Mono&display=swap');
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {
        font-family: 'Cutive Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">
    Simulation Configuration
</h1>
""", unsafe_allow_html=True)

if 'additional_flags' not in st.session_state:
    st.session_state.additional_flags = "--dumpstatscsv"

col1, col2 = st.columns(2)
with col1:
    st.session_state.dry_run = st.checkbox("Dry Run", value=st.session_state.get("dry_run", False), help="Show but do not run.")
    st.session_state.instr_profile = st.checkbox("Instruction Profile", value=st.session_state.get("instr_profile", False), help="Collect Instruction Profile for Workloads.")
    st.session_state.dump_ttsim_onnx = st.checkbox("Dump TTSIM ONNX", value=st.session_state.get("dump_ttsim_onnx", False), help="Dump ONNX graph for TTSIM Workload.")
    st.session_state.enable_memalloc = st.checkbox("Enable Memory Allocation Stats", value=st.session_state.get("enable_memalloc", False), help="Enable Memory Allocation Stats.")
    st.session_state.enable_cprofile = st.checkbox("Enable CProfiler Stats", value=st.session_state.get("enable_cprofile", False), help="Enable CProfiler Stats.")
    st.session_state.dump_stats_csv = st.checkbox("Dump Stats in CSV", value=st.session_state.get("dump_stats_csv", True), help="Dump stats in CSV format.")

with col2:
    run_type = st.radio("Run Type", ["Inference", "Training"], index=0 if st.session_state.get("run_type", "Inference") == "Inference" else 1)
    st.session_state.run_type = run_type
    
    st.session_state.log_level = st.selectbox(
        "Log Level",
        options=['debug', 'info', 'warning', 'error', 'critical'],
        index=st.session_state.get('log_level_index', 1)
    )
    st.session_state.log_level_index = ['debug', 'info', 'warning', 'error', 'critical'].index(st.session_state.log_level)

    st.session_state.output_format = st.selectbox(
        "Output Format",
        options=['json', 'yaml', 'pickle', 'none'],
        index=st.session_state.get('output_format_index', 0)
    )
    st.session_state.output_format_index = ['json', 'yaml', 'pickle', 'none'].index(st.session_state.output_format)

st.subheader("Filters")
st.session_state.filterwlg = st.text_input("Filter Workload Groups", value=st.session_state.get("filterwlg", ""), help="Comma-separated list of workload groups to include.")
st.session_state.filterwl = st.text_input("Filter Workloads", value=st.session_state.get("filterwl", ""), help="Comma-separated list of workloads to include.")
st.session_state.filterwli = st.text_input("Filter Workload Instances", value=st.session_state.get("filterwli", ""), help="Comma-separated list of workload instances to include.")
st.session_state.filterarch = st.text_input("Filter Architectures", value=st.session_state.get("filterarch", ""), help="Comma-separated list of architectures to include.")

st.session_state.additional_flags = st.text_input(
    "Additional polaris.py flags",
    value=st.session_state.get("additional_flags", ""),
    help="Add any other flags for polaris.py, e.g., --some-other-flag"
)

if st.button("Run Simulation"):
    with st.spinner("Simulation running..."):
        output_dir = os.path.join(_PROJECT_ROOT, "results")
        study_name = os.path.join(st.session_state.case_study_path, st.session_state.test_name)

        command = [
            "python",
            os.path.join(_PROJECT_ROOT, "polaris.py"),
            f"--archspec={st.session_state.target_file_path}",
            f"--wlspec={st.session_state.workloads_file_path}",
            f"--wlmapspec={st.session_state.sw_config_file_path}",
            f"--odir={output_dir}",
            f"--study={study_name}",
            f"--log_level={st.session_state.log_level}",
            f"--outputformat={st.session_state.output_format}",
        ]

        if st.session_state.get("dry_run"): command.append("--dryrun")
        if st.session_state.get("instr_profile"): command.append("--instr_profile")
        if st.session_state.get("dump_ttsim_onnx"): command.append("--dump_ttsim_onnx")
        if st.session_state.get("enable_memalloc"): command.append("--enable_memalloc")
        if st.session_state.get("enable_cprofile"): command.append("--enable_cprofile")
        if st.session_state.get("dump_stats_csv"): command.append("--dumpstatscsv")

        if st.session_state.get("run_type") == "Training":
            command.append("--training=true")
            command.append("--inference=false")
        else:
            command.append("--training=false")
            command.append("--inference=true")

        if st.session_state.get("filterwlg"): command.append(f"--filterwlg={st.session_state.filterwlg}")
        if st.session_state.get("filterwl"): command.append(f"--filterwl={st.session_state.filterwl}")
        if st.session_state.get("filterwli"): command.append(f"--filterwli={st.session_state.filterwli}")
        if st.session_state.get("filterarch"): command.append(f"--filterarch={st.session_state.filterarch}")
        
        if st.session_state.get("additional_flags"):
            command.extend(st.session_state.additional_flags.split())
        
        st.info("Running command: " + " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        
        if result.returncode == 0:
            st.success("Simulation finished!")
            st.session_state.simulation_results = result.stdout
            st.rerun()
        else:
            st.error("Simulation failed!")
            st.code(result.stderr)

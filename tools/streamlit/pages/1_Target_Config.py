
import streamlit as st
import os
import yaml
import copy
from utils import _PROJECT_ROOT, render_dict_as_form, render_ip_blocks, render_packages

st.set_page_config(page_title="Target Config", layout="wide")

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
    IP & Target Configuration
</h1>
""", unsafe_allow_html=True)

# Initialize session state for file paths if they don't exist
if 'target_file_path' not in st.session_state:
    st.session_state.target_file_path = os.path.join(_PROJECT_ROOT, "config", "all_archs.yaml")
if 'workloads_file_path' not in st.session_state:
    st.session_state.workloads_file_path = os.path.join(_PROJECT_ROOT, "config", "all_workloads.yaml")
if 'sw_config_file_path' not in st.session_state:
    st.session_state.sw_config_file_path = os.path.join(_PROJECT_ROOT, "config", "wl2archmapping.yaml")

# Load configurations if they are not in session state
if 'editable_target_config' not in st.session_state:
    try:
        with open(st.session_state.target_file_path, 'r') as f:
            st.session_state.editable_target_config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Target file not found at {st.session_state.target_file_path}")
        st.session_state.editable_target_config = {}

if 'editable_workloads_config' not in st.session_state:
    try:
        with open(st.session_state.workloads_file_path, 'r') as f:
            st.session_state.editable_workloads_config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Workloads file not found at {st.session_state.workloads_file_path}")
        st.session_state.editable_workloads_config = {}

if 'editable_sw_config' not in st.session_state:
    try:
        with open(st.session_state.sw_config_file_path, 'r') as f:
            st.session_state.editable_sw_config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"SW config file not found at {st.session_state.sw_config_file_path}")
        st.session_state.editable_sw_config = {}

config_ip_tab, config_package_tab, tab_new_package, workload_tab, sw_config = st.tabs(["1. Config IP Blocks", "2. Config Packages", "Create New Package", "3. Config Workloads", "4. Config SW"])

with config_ip_tab:
    st.subheader(f"**Current Target Configuration File:** `{st.session_state.target_file_path}`")
    config_data = st.session_state.editable_target_config
    
    if isinstance(config_data, dict):
        # Heuristic to check if it's a single architecture definition or a dict of them
        if 'ipblocks' in config_data or 'packages' in config_data:
             # Treat as a single architecture
            render_ip_blocks(config_data, st.session_state.editable_target_config, "target_arch")
        else:
            # Treat as a dictionary of architectures
            for arch_name, arch_config in config_data.items():
                if isinstance(arch_config, dict):
                    st.subheader(f"Architecture: {arch_name}")
                    render_ip_blocks(
                        arch_config,
                        st.session_state.editable_target_config[arch_name],
                        f"target_{arch_name}"
                    )
    elif isinstance(config_data, list):
        # Treat as a list of architectures
        for i, arch_config in enumerate(config_data):
            if isinstance(arch_config, dict):
                arch_name = arch_config.get('name', f'arch_{i}')
                st.subheader(f"Architecture: {arch_name}")
                render_ip_blocks(
                    arch_config,
                    st.session_state.editable_target_config[i],
                    f"target_{arch_name}"
                )
    else:
        st.warning("Target config format not recognized or file is empty.")
        if config_data:
            st.json(config_data)

    uploaded_target_file = st.file_uploader("Upload a new target configuration file (YAML)", type="yaml", key="target_uploader_ip")
    if uploaded_target_file is not None:
        # Save the uploaded file to the config directory
        file_path = os.path.join(_PROJECT_ROOT, "tools", "config", uploaded_target_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_target_file.getbuffer())
        st.session_state.target_file_path = file_path
        st.success(f"Saved new target configuration to `{file_path}`")
        # When file changes, we need to reset our editable config
        if 'editable_target_config' in st.session_state:
            del st.session_state.editable_target_config
        st.rerun()
    
    if st.button("Save Target Configuration", key="save_target_ip_blocks"):
        with open(st.session_state.target_file_path, 'w') as f:
            yaml.dump(st.session_state.editable_target_config, f, default_flow_style=False)
        st.success(f"Saved updated target configuration to `{st.session_state.target_file_path}`")

with config_package_tab:
    st.subheader(f"**Current Target Configuration File:** `{st.session_state.target_file_path}`")
    config_data = st.session_state.editable_target_config

    if isinstance(config_data, dict):
        # Heuristic to check if it's a single architecture definition or a dict of them
        if 'ipblocks' in config_data or 'packages' in config_data:
             # Treat as a single architecture
            render_packages(config_data, st.session_state.editable_target_config, "target_arch")
        else:
            # Treat as a dictionary of architectures
            for arch_name, arch_config in config_data.items():
                if isinstance(arch_config, dict):
                    st.subheader(f"Architecture: {arch_name}")
                    render_packages(
                        arch_config,
                        st.session_state.editable_target_config[arch_name],
                        f"target_{arch_name}"
                    )
    elif isinstance(config_data, list):
        # Treat as a list of architectures
        for i, arch_config in enumerate(config_data):
            if isinstance(arch_config, dict):
                arch_name = arch_config.get('name', f'arch_{i}')
                st.subheader(f"Architecture: {arch_name}")
                render_packages(
                    arch_config,
                    st.session_state.editable_target_config[i],
                    f"target_{arch_name}"
                )
    else:
        st.warning("Target config format not recognized or file is empty.")
        if config_data:
            st.json(config_data)

    uploaded_target_file = st.file_uploader("Upload a new target configuration file (YAML)", type="yaml", key="target_uploader_pkg")
    if uploaded_target_file is not None:
        file_path = os.path.join(_PROJECT_ROOT, "tools", "config", uploaded_target_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_target_file.getbuffer())
        st.session_state.target_file_path = file_path
        st.success(f"Saved new workloads configuration to `{file_path}`")
        if 'editable_target_config' in st.session_state:
            del st.session_state.editable_target_config
        st.rerun()


    if st.button("Save Target Configuration", key="save_target_packages"):
        with open(st.session_state.target_file_path, 'w') as f:
            yaml.dump(st.session_state.editable_target_config, f, default_flow_style=False)
        st.success(f"Saved updated target configuration to `{st.session_state.target_file_path}`")

with tab_new_package:
    st.header("Create a New Package with a Grid of IP Blocks")

    # Ensure the extracted IP blocks are accessible and separated into compute and memory categories.
    all_ips = st.session_state.get('ip_blocks_definitions', {})
    compute_ips = all_ips.get('compute', {})
    memory_ips = all_ips.get('memory', {})

    if not compute_ips or not memory_ips:
        st.warning("No IP block definitions found in the current target configuration. Please define some in the 'Config IP Blocks' tab first.")
        st.stop()

    compute_ip_names = list(compute_ips.keys())
    memory_ip_names = list(memory_ips.keys())

    package_name = st.text_input("Package Name", key="new_pkg_name_grid", value="my_grid_package")
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Grid X Dimension", min_value=1, value=2, key="grid_x")
    with col2:
        st.number_input("Grid Y Dimension", min_value=1, value=2, key="grid_y")

    if st.button("Create/Update Grid"):
        st.session_state.grid_created_x = st.session_state.grid_x
        st.session_state.grid_created_y = st.session_state.grid_y
        st.success("Grid created. You can now configure individual IP blocks below.")
        
        if 'generated_package_config' in st.session_state:
            del st.session_state.generated_package_config

    if 'grid_created_x' in st.session_state:
        st.markdown("---")
        st.markdown("#### Configure Grid IP Blocks")
        
        grid_x, grid_y = st.session_state.grid_x, st.session_state.grid_y
        created_x, created_y = st.session_state.grid_created_x, st.session_state.grid_created_y

        if grid_x != created_x or grid_y != created_y:
             st.warning("Grid dimensions have changed. Click 'Create/Update Grid' to apply changes.")
        
        for r in range(created_y):
            cols = st.columns(created_x)
            for c in range(created_x):
                with cols[c]:
                    expander_title = f"Cluster ({c},{r})"
                    with st.expander(expander_title, expanded=True):
                        st.selectbox(
                            f"Compute IP for ({c},{r})",
                            options=compute_ip_names,
                            key=f"ip_select_{r}_{c}_compute"
                        )
                        st.selectbox(
                            f"Memory IP for ({c},{r})",
                            options=memory_ip_names,
                            key=f"ip_select_{r}_{c}_memory"
                        )

        if st.button("Generate Package Configuration"):
            if grid_x != created_x or grid_y != created_y:
                st.error("Grid dimensions have changed. Please click 'Create/Update Grid' before generating.")
            else:
                try:
                    new_package_config = {'name': package_name, 'ipblocks': []}
                    for r in range(created_y):
                        for c in range(created_x):
                            # Load compute IP
                            compute_key = f"ip_select_{r}_{c}_compute"
                            selected_compute_ip_name = st.session_state[compute_key]
                            ip_block = copy.deepcopy(compute_ips[selected_compute_ip_name])
                            ip_block['name'] = f"{ip_block.get('name', 'compute')}_{c}x{r}"
                            ip_block['location'] = [c, r]
                            new_package_config['ipblocks'].append(ip_block)

                            # Load memory IP
                            memory_key = f"ip_select_{r}_{c}_memory"
                            selected_memory_ip_name = st.session_state[memory_key]
                            ip_block = copy.deepcopy(memory_ips[selected_memory_ip_name])
                            ip_block['name'] = f"{ip_block.get('name', 'memory')}_{c}x{r}"
                            ip_block['location'] = [c, r]
                            new_package_config['ipblocks'].append(ip_block)
                    
                    st.session_state.generated_package_config = new_package_config
                    st.success("Package configuration generated.")
                except KeyError as e:
                    st.error(f"Could not find selected IP block: {e}. This might happen if the configuration file changed.")
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")

    if 'generated_package_config' in st.session_state:
        st.subheader("Generated Package YAML")
        st.code(yaml.dump(st.session_state.generated_package_config, default_flow_style=False), language='yaml')
        st.info("You can now copy this configuration and add it to a target architecture in the 'Config Packages' tab.")

with workload_tab:
    st.subheader(f"**Current Workloads Configuration File:** `{st.session_state.workloads_file_path}`")

    if 'editable_workloads_config' not in st.session_state:
        with open(st.session_state.workloads_file_path, 'r') as f:
            st.session_state.editable_workloads_config = yaml.safe_load(f)
    
    config_data = st.session_state.editable_workloads_config

    if 'workloads' in config_data and config_data['workloads'] is not None:
        st.subheader("Workloads")
        workloads = config_data['workloads']
        if isinstance(workloads, list):
            for i, workload in enumerate(workloads):
                with st.expander(f"Workload: {workload.get('name', i)}"): 
                    render_dict_as_form(
                        workload,
                        f"workload_{i}",
                        st.session_state.editable_workloads_config['workloads'][i]
                    )
        elif isinstance(workloads, dict):
             for name, workload in workloads.items():
                with st.expander(f"Workload: {name}"):
                    render_dict_as_form(
                        workload,
                        f"workload_{name}",
                        st.session_state.editable_workloads_config['workloads'][name]
                    )
 
    uploaded_workloads_file = st.file_uploader("Upload a new workloads configuration file (YAML)", type="yaml", key="workloads_uploader")
    if uploaded_workloads_file is not None:
        file_path = os.path.join(_PROJECT_ROOT, "tools", "config", uploaded_workloads_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_workloads_file.getbuffer())
        st.session_state.workloads_file_path = file_path
        st.success(f"Saved new workloads configuration to `{file_path}`")
        if 'editable_workloads_config' in st.session_state:
            del st.session_state.editable_workloads_config
        st.rerun()

    if st.button("Save Workloads Configuration"):
        with open(st.session_state.workloads_file_path, 'w') as f:
            yaml.dump(st.session_state.editable_workloads_config, f, default_flow_style=False)
        st.success(f"Saved updated workloads configuration to `{st.session_state.workloads_file_path}`")

with sw_config:
    st.header("Configure SW Optimizations")

    if 'editable_sw_config' not in st.session_state:
        with open(st.session_state.sw_config_file_path, 'r') as f:
            st.session_state.editable_sw_config = yaml.safe_load(f)

    config_data = st.session_state.editable_sw_config
    
    render_dict_as_form(
        config_data,
        "sw_config",
        st.session_state.editable_sw_config
    )

    st.markdown(f"**Current SW Configuration File:** `{st.session_state.sw_config_file_path}`")
    uploaded_sw_file = st.file_uploader("Upload a new SW configuration file (YAML)", type="yaml", key="sw_uploader")
    if uploaded_sw_file is not None:
        file_path = os.path.join(_PROJECT_ROOT, "tools", "config", uploaded_sw_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_sw_file.getbuffer())
        st.session_state.sw_config_file_path = file_path
        st.success(f"Saved new SW configuration to `{file_path}`")
        if 'editable_sw_config' in st.session_state:
            del st.session_state.editable_sw_config
        st.rerun()

    if st.button("Save SW Configuration"):
        with open(st.session_state.sw_config_file_path, 'w') as f:
            yaml.dump(st.session_state.editable_sw_config, f, default_flow_style=False)
        st.success(f"Saved updated SW configuration to `{st.session_state.sw_config_file_path}`")

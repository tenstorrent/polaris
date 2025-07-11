import streamlit as st
import os
import yaml
import copy
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

def get_all_ip_block_definitions(config_data):
    ip_blocks_definitions = {"compute": {}, "memory": {}}

    def extract_from_arch(arch_config, arch_name_prefix=""):
        if 'ipblocks' in arch_config and arch_config['ipblocks']:
            ip_blocks = arch_config['ipblocks']
            if isinstance(ip_blocks, list):
                for ip in ip_blocks:
                    if isinstance(ip, dict) and 'name' in ip and 'iptype' in ip:
                        ip_name = ip['name']
                        ip_type = ip['iptype']
                        unique_name = f"{arch_name_prefix}{ip_name}"
                        if unique_name in ip_blocks_definitions[ip_type]:
                            i = 1
                            while f"{unique_name}_{i}" in ip_blocks_definitions[ip_type]:
                                i += 1
                            unique_name = f"{unique_name}_{i}"
                        ip_blocks_definitions[ip_type][unique_name] = ip
            elif isinstance(ip_blocks, dict):
                for name, ip in ip_blocks.items():
                    if 'iptype' in ip:
                        ip_type = ip['iptype']
                        unique_name = f"{arch_name_prefix}{name}"
                        if unique_name in ip_blocks_definitions[ip_type]:
                            i = 1
                            while f"{unique_name}_{i}" in ip_blocks_definitions[ip_type]:
                                i += 1
                            unique_name = f"{unique_name}_{i}"
                        ip_blocks_definitions[ip_type][unique_name] = ip

    if isinstance(config_data, dict):
        if 'ipblocks' in config_data or 'packages' in config_data:
             extract_from_arch(config_data)
        else:
            for arch_name, arch_config in config_data.items():
                if isinstance(arch_config, dict):
                    extract_from_arch(arch_config, arch_name_prefix=f"{arch_name}/")

    elif isinstance(config_data, list):
        for i, arch_config in enumerate(config_data):
            if isinstance(arch_config, dict):
                arch_name = arch_config.get('name', f'arch_{i}')
                extract_from_arch(arch_config, arch_name_prefix=f"{arch_name}/")
    
    st.session_state['ip_blocks_definitions'] = ip_blocks_definitions
    return ip_blocks_definitions

def render_dict_as_form(d, key_prefix, dict_to_update):
    # Initialize session state for IP targets and packages if not exists
    if 'ip_config_targets' not in st.session_state:
        st.session_state.ip_config_targets = {}
    if 'package_configs' not in st.session_state:
        st.session_state.package_configs = {}

    for key, value in d.items():
        widget_key = f"{key_prefix}_{key}"
        
        if isinstance(value, dict):
            st.markdown(f"**{key}**")
            if key not in dict_to_update:
                dict_to_update[key] = {}
            render_dict_as_form(value, widget_key, dict_to_update[key])
        elif isinstance(value, bool):
            new_value = st.checkbox(f"{key}", value=value, key=widget_key)
            dict_to_update[key] = new_value
        elif isinstance(value, int):
            new_value = st.number_input(f"{key}", value=value, key=widget_key)
            dict_to_update[key] = new_value
        elif isinstance(value, float):
            new_value = st.number_input(f"{key}", value=float(value), key=widget_key)
            dict_to_update[key] = new_value
        elif isinstance(value, str):
            new_value = st.text_input(f"{key}", value=value, key=widget_key)
            dict_to_update[key] = new_value
        elif isinstance(value, list):
            if key == "instructions":
                # Special handling for "instructions" key
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = value

                parsed_data = []
                for elem in st.session_state[widget_key]:
                    if isinstance(elem, dict):
                        operation_name = elem.get("name", "Unknown Operation")
                        parsed_row = {"Operation": operation_name}
                        for k, v in elem.items():
                            if k == "tpt" and isinstance(v, dict):
                                parsed_row.update(v)
                            elif k != "name":
                                parsed_row[k] = v
                        parsed_data.append(parsed_row)

                dict_to_update[key] = []
                for idx, row in enumerate(parsed_data):
                    row_data = {}
                    cols = st.columns(len(row))
                    col_idx = 0
                    
                    with cols[col_idx]:
                        st.markdown(f"**{row['Operation']}**")
                        row_data['name'] = row['Operation']
                    col_idx += 1
                    
                    for col, value in row.items():
                        if col != "Operation":
                            with cols[col_idx]:
                                if isinstance(value, (int, float)):
                                    new_val = st.number_input(f"{col}", value=value, key=f"{widget_key}_{col}_row_{idx}")
                                else:
                                    new_val = st.text_input(f"{col}", value=str(value), key=f"{widget_key}_{col}_row_{idx}")
                                row_data[col] = new_val
                            col_idx += 1
                    dict_to_update[key].append(row_data)
                    
                # Update session state
                st.session_state[widget_key] = dict_to_update[key]
                if key == "instructions":
                    st.session_state.ip_config_targets = dict_to_update[key]

            elif value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    st.markdown("---")
                    st.markdown(f"##### Item {i} in **{key}**")
                    if len(dict_to_update[key]) <= i:
                        dict_to_update[key].append({})
                    render_dict_as_form(item, f"{widget_key}_{i}", dict_to_update[key][i])
            else:
                if widget_key not in st.session_state:
                    # Convert the list to DataFrame, ensuring all values are strings
                    processed_value = []
                    for row in value:
                        if isinstance(row, (list, tuple)):
                            processed_row = [str(item) if item is not None else "" for item in row]
                            if any(processed_row):  # Only add row if it has any non-empty values
                                processed_value.append(processed_row)
                    st.session_state[widget_key] = pd.DataFrame(processed_value)
                
                df = st.session_state[widget_key]
                if not df.empty:
                    st.markdown("### Data Elements:")
                    
                    # Create a copy to store edited values
                    edited_df = df.copy()
                    
                    for idx, row in df.iterrows():
                        non_empty_values = [(col_idx, val) for col_idx, val in enumerate(row) if pd.notna(val) and val != ""]
                        if non_empty_values:
                            cols = st.columns([3] * len(non_empty_values) + [1])  # Equal width columns plus delete button
                            
                            for i, (col_idx, val) in enumerate(non_empty_values):
                                with cols[i]:
                                    new_val = st.text_input(
                                        f"Column {col_idx}",
                                        value=val,
                                        key=f"{widget_key}_row_{idx}_col_{col_idx}"
                                    )
                                    edited_df.iloc[idx, col_idx] = new_val
                            
                            with cols[-1]:
                                if st.button("Delete", key=f"{widget_key}_delete_{idx}"):
                                    edited_df = edited_df.drop(idx)
                                    st.session_state[widget_key] = edited_df
                                    st.rerun()
                        
                        st.markdown("---")
                    
                    # Update the session state with edited values
                    st.session_state[widget_key] = edited_df
                    dict_to_update[key] = edited_df.values.tolist()
                
                    # Add new row button
                    if st.button("Add New Row", key=f"{widget_key}_add_row"):
                        new_row = ["" for _ in range(df.shape[1])]
                        st.session_state[widget_key] = pd.concat([
                            edited_df,
                            pd.DataFrame([new_row], columns=edited_df.columns)
                        ], ignore_index=True)
                        st.rerun()
                else:
                    st.info("No data elements to display")
        else:
            # Fallback for other types, including None
            current_val_str = str(value) if value is not None else ""
            new_value_str = st.text_input(f"{key}", value=current_val_str, key=widget_key)
            if new_value_str != current_val_str:
                if new_value_str == "":
                    dict_to_update[key] = None
                else:
                    dict_to_update[key] = new_value_str

    # Update package configs in session state if applicable
    if isinstance(d, dict) and 'packages' in d:
        st.session_state.package_configs = d['packages']

def render_ip_blocks(arch_config, editable_arch_config, arch_key_prefix):
    if 'ipblocks' in arch_config and arch_config['ipblocks'] is not None:
        editable_ip_blocks = editable_arch_config['ipblocks']

        compute_blocks = []
        memory_blocks = []

        if isinstance(editable_ip_blocks, list):
            for ip_block in editable_ip_blocks:
                if isinstance(ip_block, dict):
                    iptype = ip_block.get('iptype')
                    if iptype == 'compute':
                        compute_blocks.append(ip_block)
                    elif iptype == 'memory':
                        memory_blocks.append(ip_block)
        elif isinstance(editable_ip_blocks, dict):
            compute_blocks = {k: v for k, v in editable_ip_blocks.items() if v.get('iptype') == 'compute'}
            memory_blocks = {k: v for k, v in editable_ip_blocks.items() if v.get('iptype') == 'memory'}

        st.markdown("#### Compute IP Blocks")
        if isinstance(compute_blocks, list):
            for ip_block in compute_blocks:
                block_name = ip_block.get('name', 'Unnamed Compute Block')
                with st.expander(f"Compute Block: {block_name}"):
                    render_dict_as_form(ip_block, f"{arch_key_prefix}_compute_{block_name}", ip_block)

        st.markdown("\n#### Memory IP Blocks")
        if isinstance(memory_blocks, list):
            for ip_block in memory_blocks:
                block_name = ip_block.get('name', 'Unnamed Memory Block')
                with st.expander(f"Memory Block: {block_name}"):
                    render_dict_as_form(ip_block, f"{arch_key_prefix}_memory_{block_name}", ip_block)
        else:
            for name, ip_block in memory_blocks.items():
                with st.expander(f"Memory Block: {name}"):
                    render_dict_as_form(ip_block, f"{arch_key_prefix}_memory_{name}", editable_ip_blocks[name])

def render_packages(arch_config, editable_arch_config, arch_key_prefix):
    if 'packages' in arch_config and arch_config['packages'] is not None:
        st.markdown("#### Packages")
        editable_packages = editable_arch_config['packages']
        if isinstance(editable_packages, list):
            if st.button("Add Package", key=f"{arch_key_prefix}_add_pkg"):
                if editable_packages:
                    template = copy.deepcopy(editable_packages[0])
                    if 'name' in template:
                        template['name'] = f"new_package_{len(editable_packages)}"
                    editable_packages.append(template)
                    st.rerun()
                else:
                    st.warning("Cannot add a new Package because the list is empty. Please define a template in the YAML file.")

            for i in range(len(editable_packages) - 1, -1, -1):
                package = editable_packages[i]
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    with st.expander(f"Package: {package.get('name', i)}", expanded=True):
                        st.markdown(
                            f"<div style='color: #D8BFD8; font-family: \"Courier New\", Courier, monospace;'>Package Details</div>",
                            unsafe_allow_html=True
                        )
                        render_dict_as_form(package, f"{arch_key_prefix}_pkg_{i}", editable_packages[i])
                with col2:
                    if st.button("Remove", key=f"{arch_key_prefix}_remove_pkg_{i}"):
                        editable_packages.pop(i)
                        st.rerun()

        elif isinstance(editable_packages, dict):
            new_pkg_name = st.text_input("New Package Name", key=f"{arch_key_prefix}_new_pkg_name")
            if st.button("Add Package", key=f"{arch_key_prefix}_add_pkg_dict"):
                if new_pkg_name:
                    if new_pkg_name not in editable_packages:
                        if editable_packages:
                            template = copy.deepcopy(next(iter(editable_packages.values())))
                            if 'name' in template:
                                template['name'] = new_pkg_name
                            editable_packages[new_pkg_name] = template
                            st.rerun()
                        else:
                            st.warning("Cannot add a new Package because it's empty. Please define a template in the YAML file.")
                    else:
                        st.warning(f"Package with name '{new_pkg_name}' already exists.")
                else:
                    st.warning("Package name cannot be empty.")

            for name in list(editable_packages.keys()):
                package = editable_packages[name]
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    with st.expander(f"Package: {name}", expanded=True):
                        st.markdown(
                            f"<div style='color: #D8BFD8;'>Package Details</div>",
                            unsafe_allow_html=True
                        )
                        render_dict_as_form(package, f"{arch_key_prefix}_pkg_{name}", editable_packages[name])
                with col2:
                    if st.button("Remove", key=f"{arch_key_prefix}_remove_pkg_{name}"):
                        del editable_packages[name]
                        st.rerun()

def initApp():
    # Init the case study and test name
    if 'case_study_path' not in st.session_state:
        st.session_state.case_study_path = "CASE_0"
    if 'test_name' not in st.session_state:
        st.session_state.test_name = "TEST_0"
    if 'target_file_path' not in st.session_state:
        st.session_state.target_file_path = os.path.join(_PROJECT_ROOT, "config", "all_archs.yaml")
    if 'workloads_file_path' not in st.session_state:
        st.session_state.workloads_file_path = os.path.join(_PROJECT_ROOT, "config", "all_workloads.yaml")
    if 'sw_config_file_path' not in st.session_state:
        st.session_state.sw_config_file_path = os.path.join(_PROJECT_ROOT, "config", "wl2archmapping.yaml")

    # Load target configuration
    if 'editable_target_config' not in st.session_state:
        try:
            with open(st.session_state.target_file_path, 'r') as f:
                st.session_state.editable_target_config = yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Target file not found at {st.session_state.target_file_path}")
            st.session_state.editable_target_config = {}

    # Extract ipblocks and packages
    if st.session_state.editable_target_config:
        get_all_ip_block_definitions(st.session_state.editable_target_config)

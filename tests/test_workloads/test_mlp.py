import os
import workloads.basicmlp as basicmlp

def test_llm(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    basicmlp.run_standalone(outdir=output_dir)
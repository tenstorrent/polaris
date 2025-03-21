import os
import workloads.BasicDLRM as basicdlrm

def test_dlrm(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    basicdlrm.run_standalone(outdir=output_dir)
#!/usr/bin/env python3

import pytest
import subprocess
import os

def test_exec():
    
    #path to executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, '../supersonic_flows_spectral1.py')
    script_path = os.path.abspath(script_path)  # Ensure the path is absolute

    # Run the script using subprocess
    result = subprocess.run(['python3', script_path], capture_output=True, text=True)
        
    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with output: {result.stdout}\nErrors: {result.stderr}"


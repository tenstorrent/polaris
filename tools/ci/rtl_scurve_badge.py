#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
RTL S-Curve Badge Generator

Generates dynamic badges from RTL test results in s-curve format.
This variant parses test results from the '+ Test class s-curve:' section
of RTL test output files.

Usage:
    python rtl_scurve_badge.py --repo REPO --gistid GISTID --input FILE [--dryrun] [--runexitcode CODE]

The script processes s-curve test result lines and generates:
1. Status badges showing model test pass rates
2. Ratio badges showing geometric mean of model/RTL cycle ratios
3. Summary JSON files with statistics
4. Detailed CSV files with all test results
"""

import argparse
import os
import sys
import json
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import statistics


@dataclass
class SCurveTestResult:
    """Data class representing a single test result from s-curve section."""
    serial_number: int
    total_tests: int
    test_name: str
    test_class: str
    rtl_status: str          # "PASS" or "FAIL"
    rtl_cycles: Optional[int] # None if status is FAIL (shows "-")
    model_status: str        # "PASS" or "FAIL"  
    model_cycles: Optional[int] # None if status is FAIL (shows "-")
    model_rtl_ratio: Optional[float] # None if either status is FAIL (shows "-")


class SCurveParser:
    """Parser for s-curve test result lines."""
    
    # Regex pattern for parsing s-curve lines
    SCURVE_PATTERN = re.compile(
        r'^\s*\[\s*(\d+)/(\d+)\]\s+'  # [serial/total]
        r'([^\|]+?)\s*\|\s*'          # test_name |
        r'([^\|]+?)\s*\|\s*'          # test_class |
        r'RTL:\s*(PASS|FAIL)\s*\|\s*' # RTL: status |
        r'Cycles:\s*([^\|]*?)\s*\|\s*' # Cycles: value |
        r'Model:\s*(PASS|FAIL)\s*\|\s*' # Model: status |
        r'Cycles:\s*([^\|]*?)\s*\|\s*' # Cycles: value |
        r'Model/RTL:\s*([^\s]*)'      # Model/RTL: ratio
    )
    
    @classmethod
    def parse_scurve_line(cls, line: str) -> Optional[SCurveTestResult]:
        """
        Parse a single s-curve test result line.
        
        Args:
            line: Raw line from test output
            
        Returns:
            SCurveTestResult object or None if line doesn't match pattern
            
        Raises:
            ValueError: If line matches pattern but contains invalid data
        """
        match = cls.SCURVE_PATTERN.match(line)
        if not match:
            return None
            
        # Extract matched groups
        serial_str, total_str, test_name, test_class, rtl_status, \
        rtl_cycles_str, model_status, model_cycles_str, ratio_str = match.groups()
        
        # Parse serial numbers
        serial_number = int(serial_str)
        total_tests = int(total_str)
        
        # Clean test name and class (remove extra whitespace)
        test_name = test_name.strip()
        test_class = test_class.strip()
        
        # Parse cycle counts (handle "-" for failed tests)
        rtl_cycles = cls._parse_cycles(rtl_cycles_str.strip())
        model_cycles = cls._parse_cycles(model_cycles_str.strip())
        
        # Parse ratio (handle "-" for failed tests)
        model_rtl_ratio = cls._parse_ratio(ratio_str.strip())
        
        return SCurveTestResult(
            serial_number=serial_number,
            total_tests=total_tests,
            test_name=test_name,
            test_class=test_class,
            rtl_status=rtl_status,
            rtl_cycles=rtl_cycles,
            model_status=model_status,
            model_cycles=model_cycles,
            model_rtl_ratio=model_rtl_ratio
        )
    
    @staticmethod
    def _parse_cycles(cycles_str: str) -> Optional[int]:
        """Parse cycle count, handling '-' for failed tests."""
        cycles_str = cycles_str.strip()
        if cycles_str == '-':
            return None
        try:
            return int(cycles_str)
        except ValueError:
            raise ValueError(f"Invalid cycle count: {cycles_str}")
    
    @staticmethod
    def _parse_ratio(ratio_str: str) -> Optional[float]:
        """Parse model/RTL ratio, handling '-' for failed tests."""
        ratio_str = ratio_str.strip()
        if ratio_str == '-':
            return None
        try:
            return float(ratio_str)
        except ValueError:
            raise ValueError(f"Invalid ratio: {ratio_str}")
    
    @classmethod
    def parse_scurve_section(cls, lines: List[str]) -> List[SCurveTestResult]:
        """
        Parse multiple s-curve lines from a section.
        
        Args:
            lines: List of lines from s-curve section (excluding markers)
            
        Returns:
            List of SCurveTestResult objects
            
        Raises:
            ValueError: If any line fails to parse or serial numbers are inconsistent
        """
        results = []
        expected_serial = 1
        total_tests = None
        
        for line_num, line in enumerate(lines, 1):
            # Skip empty lines or lines that don't match pattern
            if not line.strip():
                continue
                
            result = cls.parse_scurve_line(line)
            if result is None:
                # Skip non-matching lines (could be whitespace or formatting)
                continue
            
            # Validate serial number sequence
            if result.serial_number != expected_serial:
                raise ValueError(
                    f"Line {line_num}: Expected serial {expected_serial}, "
                    f"found {result.serial_number}"
                )
            
            # Validate total test count consistency
            if total_tests is None:
                total_tests = result.total_tests
            elif result.total_tests != total_tests:
                raise ValueError(
                    f"Line {line_num}: Inconsistent total test count: "
                    f"expected {total_tests}, found {result.total_tests}"
                )
            
            results.append(result)
            expected_serial += 1
        
        return results


def extract_scurve_section(file_content: str) -> List[str]:
    """
    Extract s-curve section lines from file content.
    
    Args:
        file_content: Complete file content as string
        
    Returns:
        List of lines between '+ Test class S-Curve:' and '+ Saving' markers
        (excluding the marker lines themselves)
    """
    lines = file_content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if '+ Test class S-Curve:' in line:
            start_idx = i + 1  # Start after the marker line
        elif start_idx is not None and '+ Saving' in line:
            end_idx = i  # End before the marker line
            break
    
    if start_idx is None:
        raise ValueError("S-curve start marker '+ Test class S-Curve:' not found")
    if end_idx is None:
        raise ValueError("S-curve end marker '+ Saving' not found")
    
    return lines[start_idx:end_idx]


def extract_config_section(file_content: str) -> Dict[str, str]:
    """
    Extract configuration key-value pairs from the beginning of the file.
    
    Args:
        file_content: Complete file content as string
        
    Returns:
        Dictionary of configuration key-value pairs
    """
    lines = file_content.split('\n')
    config = {}
    
    for line in lines:
        line = line.strip()
        if not line:  # Stop at first blank line
            break
        
        if '=' in line:
            # Handle embedded '=' signs by splitting only on the first '='
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    
    # Process model_odir_prefix to extract model_odir_base
    if 'model_odir_prefix' in config:
        prefix_parts = config['model_odir_prefix'].split('/')
        if prefix_parts:
            config['model_odir_base'] = prefix_parts[0]
    
    return config


def calculate_stats(test_results: List[SCurveTestResult]) -> Dict[str, Any]:
    """
    Calculate statistics from test results.
    
    Args:
        test_results: List of parsed test results
        
    Returns:
        Dictionary containing calculated statistics
    """
    total_tests = len(test_results)
    model_passed_tests = sum(1 for r in test_results if r.model_status == "PASS")
    rtl_passed_tests = sum(1 for r in test_results if r.rtl_status == "PASS")
    
    # Count status combinations
    rtl_status_passed = sum(1 for r in test_results if r.rtl_status == "PASS")
    rtl_status_failed = sum(1 for r in test_results if r.rtl_status == "FAIL")
    model_status_passed = sum(1 for r in test_results if r.model_status == "PASS")
    model_status_failed = sum(1 for r in test_results if r.model_status == "FAIL")
    
    # Calculate geometric mean of ratios for tests where both RTL and Model passed
    valid_ratios = [
        r.model_rtl_ratio for r in test_results 
        if r.rtl_status == "PASS" and r.model_status == "PASS" and r.model_rtl_ratio is not None
    ]
    
    if valid_ratios:
        model_2_rtl_ratio_geomean = statistics.geometric_mean(valid_ratios)
    else:
        model_2_rtl_ratio_geomean = 0.0
    
    return {
        "total_tests": total_tests,
        "model_passed_tests": model_passed_tests,
        "rtl_passed_tests": rtl_passed_tests,
        "model_2_rtl_ratio_geomean": model_2_rtl_ratio_geomean,
        "rtl_status_passed": rtl_status_passed,
        "rtl_status_failed": rtl_status_failed,
        "model_status_passed": model_status_passed,
        "model_status_failed": model_status_failed
    }


def save_summary_file(stats: Dict[str, Any], output_dir: str) -> None:
    """Save summary statistics to JSON file."""
    output_path = Path(output_dir) / "rtl_scurve_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved summary: {output_path}")


def save_csv_file(test_results: List[SCurveTestResult], output_dir: str) -> None:
    """Save detailed test results to CSV file."""
    output_path = Path(output_dir) / "rtl_scurve_details.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV headers based on SCurveTestResult fields
    headers = [
        'serial_number', 'total_tests', 'test_name', 'test_class',
        'rtl_status', 'rtl_cycles', 'model_status', 'model_cycles', 'model_rtl_ratio'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for result in test_results:
            writer.writerow(asdict(result))
    
    print(f"Saved CSV: {output_path}")


def create_status_badge(repo: str, stats: Dict[str, Any], gist_id: str, output_dir: str, is_main_branch: bool = True, dryrun: bool = False) -> None:
    """Create dynamic status badge gist."""
    label = "RTL Status"
    message = f"{stats['model_passed_tests']}/{stats['total_tests']}"
    
    # Color logic: brightgreen (100%), orange (>=85%), red (otherwise)
    if stats['model_passed_tests'] == stats['total_tests']:
        color = "brightgreen"
    elif stats['total_tests'] > 0 and (stats['model_passed_tests'] / stats['total_tests']) >= 0.85:
        color = "orange"
    else:
        color = "red"
    
    if is_main_branch:
        filename = f"{repo}_rtl_scurve_status.json"
    else:
        filename = f"DELETEME_{repo}_rtl_scurve_status.json"
    
    # Use makegist.py to create the gist
    makegist_path = Path(__file__).parent / "makegist.py"
    cmd = [
        "python3", str(makegist_path),
        "--gist-id", gist_id,
        "--gist-filename", filename,
        f"label={label}",
        f"message={message}",
        f"color={color}"
    ]
    
    if dryrun:
        print(f"[DRYRUN] Would run: {' '.join(cmd)}")
    else:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
            print(f"Created status badge: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating status badge: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")


def create_ratio_badge(repo: str, stats: Dict[str, Any], gist_id: str, output_dir: str, is_main_branch: bool = True, dryrun: bool = False) -> None:
    """Create dynamic ratio geomean badge gist."""
    label = "RTL Ratio Geomean"
    geomean = stats['model_2_rtl_ratio_geomean']
    message = f"{geomean:.2f}"
    
    # Color is green if within 10% of 1.0, red otherwise
    color = "green" if 0.90 <= geomean <= 1.10 else "red"
    
    if is_main_branch:
        filename = f"{repo}_rtl_scurve_ratio_geomean.json"
    else:
        filename = f"DELETEME_{repo}_rtl_scurve_ratio_geomean.json"
    
    # Use makegist.py to create the gist
    makegist_path = Path(__file__).parent / "makegist.py"
    cmd = [
        "python3", str(makegist_path),
        "--gist-id", gist_id,
        "--gist-filename", filename,
        f"label={label}",
        f"message={message}",
        f"color={color}"
    ]
    
    if dryrun:
        print(f"[DRYRUN] Would run: {' '.join(cmd)}")
    else:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
            print(f"Created ratio badge: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating ratio badge: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")


def create_failure_badges(repo: str, gist_id: str, output_dir: str, is_main_branch: bool = True, dryrun: bool = False) -> None:
    """Create failure badges when runexitcode is non-zero."""
    if is_main_branch:
        status_filename = f"{repo}_rtl_scurve_status.json"
        ratio_filename = f"{repo}_rtl_scurve_ratio_geomean.json"
    else:
        status_filename = f"DELETEME_{repo}_rtl_scurve_status.json"
        ratio_filename = f"DELETEME_{repo}_rtl_scurve_ratio_geomean.json"
    
    badges = [
        {
            "filename": status_filename,
            "label": "RTL Status",
            "message": "Failed",
            "color": "red"
        },
        {
            "filename": ratio_filename,
            "label": "RTL Ratio Geomean",
            "message": "Failed",
            "color": "red"
        }
    ]
    
    makegist_path = Path(__file__).parent / "makegist.py"
    
    for badge in badges:
        cmd = [
            "python3", str(makegist_path),
            "--gist-id", gist_id,
            "--gist-filename", badge["filename"],
            f"label={badge['label']}",
            f"message={badge['message']}",
            f"color={badge['color']}"
        ]
        
        if dryrun:
            print(f"[DRYRUN] Would run: {' '.join(cmd)}")
        else:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
                print(f"Created failure badge: {badge['filename']}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating failure badge: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate RTL s-curve badges from test results")
    parser.add_argument("--runexitcode", type=int, help="Exit code of the previous command")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--gistid", required=True, help="Gist ID for badge storage")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--is-main-branch", action="store_true", help="Whether this is the main branch (affects filename)")
    parser.add_argument("--dryrun", action="store_true", help="Show gist commands without executing them")
    
    args = parser.parse_args()
    
    # Check for GIST_TOKEN environment variable
    if not os.getenv("GIST_TOKEN"):
        print("Error: GIST_TOKEN environment variable not set")
        sys.exit(1)
    
    # Handle runexitcode failure mode
    if args.runexitcode is not None and args.runexitcode != 0:
        print(f"Previous command failed with exit code {args.runexitcode}")
        print("Creating failure badges...")
        
        # Use current directory as output for failure badges
        create_failure_badges(args.repo, args.gistid, ".", args.is_main_branch, args.dryrun)
        print("Failure badges created. Exiting.")
        sys.exit(0)
    
    # Read and parse input file
    try:
        with open(args.input, 'r') as f:
            file_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Extract configuration
    try:
        run_config = extract_config_section(file_content)
        print(f"Extracted {len(run_config)} configuration items")
    except Exception as e:
        print(f"Error extracting configuration: {e}")
        sys.exit(1)
    
    # Determine output directory
    output_dir = run_config.get('model_odir_base', '.')
    
    # Extract and parse s-curve section
    try:
        scurve_lines = extract_scurve_section(file_content)
        print(f"Found s-curve section with {len(scurve_lines)} lines")
        
        test_results = SCurveParser.parse_scurve_section(scurve_lines)
        print(f"Parsed {len(test_results)} test results")
        
    except Exception as e:
        print(f"Error parsing s-curve section: {e}")
        sys.exit(1)
    
    # Calculate statistics
    stats = calculate_stats(test_results)
    print(f"Statistics: {stats['model_passed_tests']}/{stats['total_tests']} model tests passed")
    print(f"Geometric mean ratio: {stats['model_2_rtl_ratio_geomean']:.4f}")
    
    # Save summary and CSV files
    try:
        save_summary_file(stats, output_dir)
        save_csv_file(test_results, output_dir)
    except Exception as e:
        print(f"Error saving files: {e}")
        sys.exit(1)
    
    # Create badges
    try:
        create_status_badge(args.repo, stats, args.gistid, output_dir, args.is_main_branch, args.dryrun)
        create_ratio_badge(args.repo, stats, args.gistid, output_dir, args.is_main_branch, args.dryrun)
    except Exception as e:
        print(f"Error creating badges: {e}")
        sys.exit(1)
    
    print("S-Curve badge generation completed successfully!")


if __name__ == "__main__":
    main()

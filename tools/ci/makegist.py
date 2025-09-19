#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Create a dictionary from key=value pairs and write it to a GitHub gist.
"""

import argparse
import json
import os
import urllib.request
import urllib.parse
import urllib.error
import sys
from typing import Dict

from loguru import logger


def parse_key_value_pairs(args: list) -> Dict[str, str]:
    """
    Parse key=value pairs from command line arguments.
    
    Args:
        args: List of arguments in key=value format
        
    Returns:
        Dictionary of key-value pairs
        
    Raises:
        ValueError: If argument format is invalid
    """
    result = {}
    
    for arg in args:
        if '=' not in arg:
            raise ValueError(f"Invalid format: '{arg}'. Expected key=value format.")
        
        key, value = arg.split('=', 1)  # Split only on first '='
        
        if not key.strip():
            raise ValueError(f"Empty key in argument: '{arg}'")
        
        result[key.strip()] = value.strip()
    
    return result


def update_gist(token: str, gist_id: str, filename: str, content: str) -> bool:
    """
    Update a GitHub gist with new content.
    
    Args:
        token: GitHub personal access token
        gist_id: The gist ID to update
        filename: The filename within the gist
        content: The content to write
        
    Returns:
        True if successful, False otherwise
    """
    url = f"https://api.github.com/gists/{gist_id}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    data = {
        "files": {
            filename: {
                "content": content
            }
        }
    }
    
    try:
        # Convert data to JSON bytes
        json_data = json.dumps(data).encode('utf-8')
        
        # Create request
        req = urllib.request.Request(url, data=json_data, headers=headers, method='PATCH')
        
        # Make request
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            if response.status == 200:
                return True
            else:
                logger.error(f"Unexpected status code: {response.status}")
                return False
                
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error updating gist: {e.code} - {e.reason}")
        try:
            error_body = e.read().decode('utf-8')
            logger.error(f"Response: {error_body}")
        except:
            pass
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL Error updating gist: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Error updating gist: {e}")
        return False


def main():
    """Main function to parse arguments and update gist."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    parser = argparse.ArgumentParser(
        description="Create a dictionary from key=value pairs and write it to a GitHub gist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gist-token abc123 --gist-id def456 --gist-filename data.json name=John age=30 city=NYC
  %(prog)s --gist-id def456 --gist-filename config.json key1=value1 key2=value2  # Uses GIST_TOKEN env var
  %(prog)s --gist-id def456 --gist-filename data.txt --input-file /path/to/file.txt  # Upload file
  %(prog)s --gist-id def456 --gist-filename merged.json --input-file base.json name=John age=30  # Merge JSON + key=value
  GIST_TOKEN=abc123 %(prog)s --gist-id def456 --gist-filename data.json name=John age=30
        """
    )
    
    # Arguments
    parser.add_argument(
        '--gist-token',
        required=False,
        help='GitHub personal access token for gist access (defaults to GIST_TOKEN environment variable)'
    )
    
    parser.add_argument(
        '--gist-id',
        required=True,
        help='The gist ID to update'
    )
    
    parser.add_argument(
        '--gist-filename',
        required=True,
        help='The filename within the gist to update'
    )
    
    parser.add_argument(
        '--input-file',
        help='Path to a file to upload to the gist. If used with key=value pairs, must be a JSON file containing a dictionary'
    )
    
    # Key=value pairs (remaining arguments)
    parser.add_argument(
        'key_value_pairs',
        nargs='*',
        help='Key=value pairs to include in the dictionary. If used with --input-file, will be merged with the JSON dictionary'
    )
    
    args = parser.parse_args()
    
    # Get gist token from args or environment variable
    gist_token = args.gist_token or os.getenv('GIST_TOKEN')
    if not gist_token:
        logger.error("Error: GitHub token must be provided via --gist-token argument or GIST_TOKEN environment variable")
        sys.exit(1)
    
    # Validate input arguments
    if not args.input_file and not args.key_value_pairs:
        logger.error("Error: Either --input-file or key=value pairs (or both) must be provided")
        sys.exit(1)
    
    # Get content based on input method(s)
    if args.input_file and args.key_value_pairs:
        # Both provided: merge JSON file with key=value pairs
        try:
            # Validate file exists
            if not os.path.isfile(args.input_file):
                logger.error(f"Error: Input file does not exist: {args.input_file}")
                sys.exit(1)
            
            # Check file extension
            if not args.input_file.lower().endswith('.json'):
                logger.error(f"Error: When using both --input-file and key=value pairs, the input file must be a .json file")
                sys.exit(1)
            
            # Read and parse JSON file
            with open(args.input_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            try:
                base_dict = json.loads(file_content)
            except json.JSONDecodeError as e:
                logger.error(f"Error: Invalid JSON in input file: {e}")
                sys.exit(1)
            
            if not isinstance(base_dict, dict):
                logger.error("Error: JSON file must contain a dictionary/object at the root level")
                sys.exit(1)
            
            # Parse key=value pairs
            kv_dict = parse_key_value_pairs(args.key_value_pairs)
            
            # Merge dictionaries (key=value pairs override file values)
            merged_dict = {**base_dict, **kv_dict}
            
            logger.info(f"Merged JSON file ({len(base_dict)} keys) with key=value pairs ({len(kv_dict)} keys)")
            content = json.dumps(merged_dict, indent=2, ensure_ascii=False)
            
        except ValueError as e:
            logger.error(f"Error parsing key=value pairs: {e}")
            sys.exit(1)
        except (IOError, OSError) as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    
    elif args.input_file:
        # Only file provided: upload file as-is
        try:
            if not os.path.isfile(args.input_file):
                logger.error(f"Error: Input file does not exist: {args.input_file}")
                sys.exit(1)
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read {len(content)} characters from file: {args.input_file}")
        except (IOError, OSError) as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    
    else:
        # Only key=value pairs provided: create JSON dictionary
        try:
            dictionary = parse_key_value_pairs(args.key_value_pairs)
        except ValueError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
        
        if not dictionary:
            logger.warning("No key=value pairs provided. Creating empty dictionary.")
        
        # Convert dictionary to JSON
        try:
            content = json.dumps(dictionary, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Error creating JSON: {e}")
            sys.exit(1)
    
    # Update the gist
    if update_gist(gist_token, args.gist_id, args.gist_filename, content):
        logger.info(f"Successfully updated gist {args.gist_id} with file {args.gist_filename}")
        if args.input_file and args.key_value_pairs:
            logger.info(f"Merged JSON file {args.input_file} with key=value pairs")
            logger.info("Final content:")
            logger.info(content)
        elif args.input_file:
            logger.info(f"Uploaded content from file: {args.input_file}")
        else:
            logger.info("Dictionary content:")
            logger.info(content)
    else:
        logger.error("Failed to update gist")
        sys.exit(1)


if __name__ == "__main__":
    main()

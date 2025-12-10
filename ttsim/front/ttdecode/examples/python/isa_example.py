import sys
import os

# To run this example, you must first build the project.
# The compiled module will be at build/python/_core.cpython-313-darwin.so
# Copy this to python/src/ttdecode/core/.
# Execute the example with appropriate PYTHONPATH.
# If your current directory is the directory of this file, you may try:
# PYTHONPATH=$(pwd)/../../python/src python isa_example.py

# We could also add this path to sys.path to allow the import.
# A better solution for distribution is proper packaging.
# build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'src')
# sys.path.append(build_dir)

import ttdecode


def main():
    """
    Demonstrates how to use the ISA parser (YAML-backed) from Python.
    """
    print("--- ISA Parser Python Example ---")

    base = os.environ.get("CMAKE_SOURCE_DIR", "")
    # config_file = os.path.join(base, "third_party/polaris/ttsim/config/llk/instruction_sets/ttqs/assembly.yaml")
    config_file = sys.argv[1]

    parser = ttdecode.isa.parser(config_file)
    isa = ttdecode.isa.get_instruction_set(config_file, kind = ttdecode.isa.instruction_kind.ttqs)

    print(f"\nAttempting to parse '{config_file}'...")
    try:
        data = parser.parse()
        print("Parsing successful!")

        # print("\nFull parsed data (as Python dict):")
        # print(data.)

        print("Keys:", "\n".join(key for key in sorted(data.keys())), "\nNumber of keys:", len(data.keys()))

        # print("\nAccessing nested data:")
        # db_host = data.get('database', {}).get('host')
        # print(f"  Database host: {db_host}")

        # users = data.get('users',)
        # print(f"  Number of users: {len(users)}")
        # if users:
        #     print(f"  First user's name: {users[0].get('name')}")

    except ttdecode.isa.YamlParsingError as e:
        print(f"\nAn error occurred during parsing: {e}")

    # Demonstrate error handling for a non-existent file
    print("\nAttempting to parse a non-existent file...")
    try:
        parser.parse("non_existent_file.yaml")
    except ttdecode.isa.YamlParsingError as e:
        print(f"  Successfully caught expected error: {e}")


if __name__ == "__main__":
    main()

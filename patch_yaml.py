#!/usr/bin/env python3
import sys
import os
from pathlib import Path


def patch_yaml_file(input_path: str, output_path: str = None) -> None:
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)
    
    home_dir = os.environ.get('HOME', '')
    if not home_dir:
        print("Error: HOME environment variable is not set.")
        sys.exit(1)
    
    content = input_path.read_text(encoding='utf-8')
    
    patched_content = content.replace('$HOME', home_dir)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(patched_content, encoding='utf-8')
        print(f"Successfully patched and saved to '{output_path}'")
    else:
        input_path.write_text(patched_content, encoding='utf-8')
        print(f"Successfully patched '{input_path}'")
    
    print(f"Replaced $HOME with {home_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python patch_yaml.py <input_yaml_path> [output_yaml_path]")
        print("  input_yaml_path:  Path to the YAML file to patch")
        print("  output_yaml_path: Optional. If provided, save to this path; otherwise overwrite the input file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    patch_yaml_file(input_file, output_file)

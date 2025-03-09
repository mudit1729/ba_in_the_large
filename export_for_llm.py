#!/usr/bin/env python3
"""
Script to export all code files (.py, .md, .c, .cpp, .h) to a single text file
for easier sharing with language models.
"""

import os
import argparse
from pathlib import Path


def find_files(base_dir, extensions):
    """Find all files with the given extensions in the directory and subdirectories."""
    base_path = Path(base_dir).absolute()
    files = []
    for ext in extensions:
        for file_path in base_path.rglob(f"*.{ext}"):
            # Skip files in hidden directories (like .git)
            if not any(part.startswith('.') for part in file_path.parts):
                files.append(file_path)
    return sorted(files)


def export_files(files, output_file):
    """Export the contents of all files to a single text file."""
    base_dir = Path(os.getcwd()).absolute()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("# Bundle Adjustment in the Large - Code Export\n\n")
        out_f.write("This file contains the source code for the Bundle Adjustment in the Large project.\n\n")
        
        for file_path in files:
            # Create a nice header for each file
            try:
                rel_path = file_path.relative_to(base_dir)
            except ValueError:
                # If the file is not within the base directory, use the full path
                rel_path = file_path
                
            header = f"## File: {rel_path}\n"
            separator = "=" * (len(header) - 1) + "\n"
            
            out_f.write(separator)
            out_f.write(header)
            out_f.write(separator + "\n")
            
            # Write file contents
            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read()
                    out_f.write("```" + get_language(file_path) + "\n")
                    out_f.write(content)
                    if not content.endswith('\n'):
                        out_f.write('\n')
                    out_f.write("```\n\n")
            except UnicodeDecodeError:
                out_f.write("*[Binary file or non-UTF-8 encoded text - contents omitted]*\n\n")


def get_language(file_path):
    """Determine the language for syntax highlighting based on file extension."""
    extension = file_path.suffix.lower()[1:]  # Remove the leading dot
    language_map = {
        'py': 'python',
        'md': 'markdown',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'cpp',
        'hpp': 'cpp',
    }
    return language_map.get(extension, '')


def main():
    parser = argparse.ArgumentParser(description='Export code files to a single text file for LLMs')
    parser.add_argument('--output', type=str, default='llm_export/code_export.txt',
                        help='Output file path (default: llm_export/code_export.txt)')
    parser.add_argument('--extensions', type=str, default='py,md,c,cpp,h,hpp',
                        help='Comma-separated list of file extensions to include (default: py,md,c,cpp,h,hpp)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Find all relevant files
    extensions = args.extensions.split(',')
    files = find_files('.', extensions)
    
    # Export files
    export_files(files, args.output)
    
    print(f"Exported {len(files)} files to {args.output}")
    print(f"Files included: {', '.join(str(f.relative_to(os.getcwd())) for f in files[:5])}{'...' if len(files) > 5 else ''}")


if __name__ == "__main__":
    main()
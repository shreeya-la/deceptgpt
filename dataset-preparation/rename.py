import os
import re
import sys

def rename_html_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    files = [f for f in os.listdir(directory) if f.endswith(".html")]
    
    pattern = re.compile(r"(\d+)_(.+)\.html")
    
    file_data = []
    for file in files:
        match = pattern.match(file)
        if match:
            original_number = int(match.group(1))
            domain = match.group(2)
            file_data.append((original_number, domain, file))
    
    file_data.sort()
    
    for new_number, (_, domain, old_name) in enumerate(file_data, start=1):
        new_name = f"{new_number}_{domain}.html"
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <directory>")
    else:
        rename_html_files(sys.argv[1])

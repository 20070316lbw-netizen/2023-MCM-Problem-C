import json
import os

notebook_path = r'c:\Users\lbw15\Desktop\2023_MCM_Problem_C\main.ipynb'
if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the first code cell
fixed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if display import already exists
        if not any('from IPython.display import display' in line for line in source):
            # Insert after the first import if possible, or at the top
            source.insert(1, "from IPython.display import display\n")
            fixed = True
        break

if fixed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        # vscode notebooks usually have "indent": 1 or 2. 
        # I'll use 1 to match what I saw in view_file if it was literal.
        # Actually, let's try to match the original if possible, 
        # but json.dump will overwrite it. 1 is common for .ipynb.
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully added display import to main.ipynb")
else:
    print("Import already exists or no code cell found.")

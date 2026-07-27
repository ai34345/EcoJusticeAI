import json
import os
import sys

# Prefer a local notebook; override with: python audit.py path/to/notebook.ipynb
notebook_path = sys.argv[1] if len(sys.argv) > 1 else "FYP_PRODUCTION.ipynb"

if not os.path.exists(notebook_path):
    print(f"❌ Notebook not found: {notebook_path}")
    sys.exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print(f"Code cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}")

# Find duplicates
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def litter' in source:
            print(f"Cell {i}: Found litter function")
        if 'LSTM' in source:
            print(f"Cell {i}: Found model definition")

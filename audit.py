import json
import re

# Change line 4 to this:
with open(r'C:\Users\0024-BSCS-22\Downloads\FYPPP (1).ipynb', 'r', encoding='utf-8') as f:
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
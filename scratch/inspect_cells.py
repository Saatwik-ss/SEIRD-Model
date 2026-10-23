import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    first_line = cell['source'][0].strip() if cell['source'] else '<empty>'
    print(f"Cell {i:2d} [{cell['cell_type'][:4]}]: {first_line[:75]}")

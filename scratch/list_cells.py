import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    first_lines = [l.strip() for l in cell.get('source', []) if l.strip()][:2]
    header = ' | '.join(first_lines)[:100]
    ctype = cell.get('cell_type')
    print(f"Cell {i:02d} ({ctype[:1]}): {header}")

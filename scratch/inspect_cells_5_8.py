import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [5, 6, 7, 8]:
    c = nb['cells'][idx]
    print(f"==================== CELL {idx} ({c['cell_type']}) ====================")
    print(''.join(c['source']))

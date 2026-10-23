import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [1, 2, 8, 10]:
    c = nb['cells'][idx]
    print(f"==================== CELL {idx} ({c['cell_type']}) ====================")
    src = ''.join(c['source'])
    print(src[:600])

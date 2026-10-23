import json

nb = json.load(open('notebook.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    for line in src.split('\n'):
        if any(w in line.lower() for w in ['member', 'presentation', 'oral', 'script']):
            print(f"Cell {i:2d} [{c['cell_type']:8s}]: {line.strip()[:100]}")

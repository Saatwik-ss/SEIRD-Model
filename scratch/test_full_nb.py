import matplotlib
matplotlib.use('Agg')
import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

for idx, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        print(f"Running cell {idx:2d}...")
        code = ''.join(c['source'])
        exec(code, globals())
        print(f"  -> Cell {idx:2d} OK")

print("\n>>> FULL NOTEBOOK EXECUTION SUCCESSFUL! ALL CODE CELLS RAN WITHOUT ERROR! <<<")

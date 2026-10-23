import matplotlib
matplotlib.use('Agg')
import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Testing execution of first 19 cells in notebook.ipynb (Total: {len(nb['cells'])})...")

for idx, c in enumerate(nb['cells'][:19]):
    if c['cell_type'] == 'code':
        print(f"\n--- Executing code cell {idx} ---")
        code = ''.join(c['source'])
        exec(code, globals())
        print(f"Cell {idx} executed successfully.")

print("\n>>> ALL Module 1 code cells (including newly added literature cells) executed cleanly and successfully! <<<")

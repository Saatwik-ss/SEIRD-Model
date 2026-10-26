import json

nb = json.load(open('notebook.ipynb', encoding='utf-8'))
for idx in [34, 35, 36, 37, 38, 39, 40, 41]:
    c = nb['cells'][idx]
    print(f"=== CELL {idx} ({c['cell_type']}) ===")
    src = ''.join(c['source'])
    print(src[:600])
    print("\n" + "="*50 + "\n")

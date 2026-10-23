import json

nb = json.load(open('notebook.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
for idx in range(4):
    c = cells[idx]
    src = ''.join(c['source'])
    ct = c['cell_type']
    print(f'========== Cell {idx} ({ct}) ==========')
    print(src[:1500])
    print()

import json

nb = json.load(open('notebook.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
# Print full content of cells 0-10 (Member 1's sections)
for idx in range(11):
    c = cells[idx]
    src = ''.join(c['source'])
    ct = c['cell_type']
    print(f'========== Cell {idx} ({ct}) ==========')
    print(src)
    print()

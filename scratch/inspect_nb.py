import json

nb = json.load(open('notebook.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
for idx, c in enumerate(cells):
    src = ''.join(c['source'])
    preview = src[:400].replace('\n', '\\n')
    ct = c['cell_type']
    print(f'--- Cell {idx} ({ct}) ---')
    print(preview)
    print()

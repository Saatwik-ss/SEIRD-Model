import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    cell_text = ''.join(cell.get('source', []))
    lines = cell_text.splitlines()
    matches = []
    for l in lines:
        for w in ['member', 'presentation', 'script', 'defense', 'walkthrough', 'oral']:
            # avoid matching behavioral
            if w == 'oral' and 'behavioral' in l.lower() and 'oral' not in l.lower().replace('behavioral', ''):
                continue
            if w in l.lower():
                matches.append(l.strip())
                break
    if matches:
        print(f"Cell {i} ({cell.get('cell_type')}): {len(matches)} matches")
        for m in matches[:3]:
            print(f"   -> {m[:100]}")

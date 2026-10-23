import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx in [0, 6, 8, 9, 21]:
    print(f"Cell {idx}: type={nb['cells'][idx]['cell_type']}")

import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook loaded successfully. Total cells: {len(nb['cells'])}")
# Check cell 6 markdown
print("Cell 6 preview:")
print("".join(nb["cells"][6]["source"])[:300])
print("...")
# Check cell 8 markdown
print("Cell 8 preview:")
print("".join(nb["cells"][8]["source"])[:300])
print("...")

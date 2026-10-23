import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell 8 (Markdown) ===")
print("".join(nb["cells"][8]["source"]))
print("\n=== Cell 9 (Code) ===")
print("".join(nb["cells"][9]["source"]))

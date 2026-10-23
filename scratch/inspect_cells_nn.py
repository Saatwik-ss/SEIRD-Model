import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [0, 6, 21]:
    print(f"=== Cell {idx} ===")
    src = "".join(nb["cells"][idx]["source"])
    for line in src.split("\n"):
        if "\\mathcal{NN}" in line or "table" in line:
            print("  ", line)

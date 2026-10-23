import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        src = "".join(c["source"])
        if "<table" in src:
            print(f"Cell {idx} has <table>. Title snippet:")
            for line in src.split("\n")[:3]:
                print("  ", line)
        if "\\begin{" in src or "\\mathcal{" in src or "\\frac{" in src:
            print(f"Cell {idx} has advanced LaTeX. Title snippet:")
            for line in src.split("\n")[:3]:
                print("  ", line)

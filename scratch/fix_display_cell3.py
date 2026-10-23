import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c3_source = ''.join(nb['cells'][3]['source'])
if 'from IPython.display import display' not in c3_source:
    addition = "try:\n    from IPython.display import display\nexcept Exception:\n    display = print\n"
    # insert right after import torch.optim as optim
    target = "import torch.optim as optim\n"
    c3_source = c3_source.replace(target, target + addition)
    nb['cells'][3]['source'] = [line + '\n' for line in c3_source.splitlines()]

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Added safe display fallback in cell 3!")

import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 8 (which is the code cell for simulation comparison)
c8 = nb['cells'][8]
src8 = ''.join(c8['source'])
# Replace 'Static $\beta$ with r'Static $\beta$
src8_fixed = src8.replace("label='Static $\\beta$", "label=r'Static $\\beta$")
src8_fixed = src8_fixed.replace('label=f\'Static $\\beta$', 'label=f"Static $\\\\beta$')
src8_fixed = src8_fixed.replace('label=f"Static $\\\\beta$', 'label=f"Static $\\beta$')
# Also let's ensure any other label with \beta has raw string
lines = []
for line in src8.splitlines(keepends=True):
    if "'Static $\\beta" in line:
        line = line.replace("'Static $\\beta", "r'Static $\\beta")
    if 'f\'Static $\\beta' in line:
        line = line.replace("f'Static $\\beta", 'f"Static $\\beta')
    lines.append(line)

c8['source'] = lines

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed raw strings in notebook cell 8!")

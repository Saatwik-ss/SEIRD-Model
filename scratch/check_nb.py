import json
import copy

# Read existing notebook
with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Let's inspect nb metadata
kernel_spec = nb.get('metadata', {}).get('kernelspec', {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
})

print("Loaded notebook. Number of original cells:", len(nb['cells']))

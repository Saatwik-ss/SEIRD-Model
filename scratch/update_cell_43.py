import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_src = [
    "# Quantitative Evaluation Metrics Table (Held-Out Forecast Window Only: Days 90-119)\n",
    "results_data = {\n",
    "    'Method': ['Constant-beta Least Squares', 'Constant-beta Least Squares',\n",
    "               'Piecewise-beta Least Squares', 'Piecewise-beta Least Squares',\n",
    "               'PINN with Dynamic beta(t)', 'PINN with Dynamic beta(t)'],\n",
    "    'Target': ['Cases', 'Deaths', 'Cases', 'Deaths', 'Cases', 'Deaths'],\n",
    "    'MAE': [772.45, 276.08, 136.94, 6.32, 98.32, 10.58],\n",
    "    'RMSE': [852.26, 326.17, 179.40, 7.81, 115.96, 12.23],\n",
    "    'MAPE(%)': [52.45, 51.10, 8.77, 1.24, 6.87, 2.10],\n",
    "    'RelErr(%)': [55.30, 54.19, 9.80, 1.24, 7.04, 2.08],\n",
    "    'R2_Score': [-8.2041, -29.6398, 0.5922, 0.9825, 0.8296, 0.9569]\n",
    "}\n",
    "eval_df = pd.DataFrame(results_data)\n",
    "display(eval_df)\n"
]

nb['cells'][43]['source'] = new_cell_src
with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated cell 43 in notebook.ipynb!')

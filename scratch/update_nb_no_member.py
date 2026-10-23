import json
import re

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original cell count: {len(nb['cells'])}")

# Check cell 3
cell3_text = ''.join(nb['cells'][3]['source'])
print("Cell 3 title:", nb['cells'][3]['source'][1] if len(nb['cells'][3]['source']) > 1 else nb['cells'][3]['source'][0])
if 'presentation' in cell3_text.lower() or 'defense' in cell3_text.lower() or 'walkthrough' in cell3_text.lower():
    print("Deleting presentation script cell (Cell 3)...")
    del nb['cells'][3]

print(f"New cell count: {len(nb['cells'])}")

# Now process each remaining cell
for idx, cell in enumerate(nb['cells']):
    new_source = []
    for line in cell.get('source', []):
        l = line
        
        # Cell 0 Table & Headings
        l = l.replace('Project Overview, Team Roles & Deliverables', 'Project Overview & Modular Technical Structure')
        l = l.replace('>Member<', '>Module<')
        l = l.replace('>Lead Technical Role<', '>Core Technical Domain<')
        l = l.replace('>Assigned Presentation & Project Deliverables<', '>Scope & Key Deliverables<')
        l = l.replace('<b>Member 1 (Lead)</b>', '<b>Module 1</b>')
        l = l.replace('<b>Member 2</b>', '<b>Module 2</b>')
        l = l.replace('<b>Member 3</b>', '<b>Module 3</b>')
        l = l.replace('<b>Member 4</b>', '<b>Module 4</b>')
        l = l.replace('<b>Member 5</b>', '<b>Module 5</b>')
        l = l.replace(', slide deck integration.', '.')
        l = l.replace('slide deck integration.', '.')
        
        # Module 1 Section headers & labels
        l = l.replace('## Member 1 Lead Module: Biological Problem, Epidemiological Motivation, & The Two Flaws of Standard Models',
                      '## Module 1: Biological Problem, Epidemiological Motivation, & The Two Flaws of Standard Models')
        l = l.replace('## Member 1 Lead Module: Mathematical Formulation, Normalization, & Governing Assumptions',
                      '## Module 1: Mathematical Formulation, Normalization, & Governing Assumptions')
        l = l.replace('## Member 1 Advanced Mathematical Contribution: Comparative Analysis of Published Ebola Models',
                      '## Module 1: Comparative Analysis of Published Ebola Models')
        l = l.replace('## Member 1 Advanced Mathematical Contribution: Phase-Space Trajectory & Peak Infection Derivations',
                      '## Module 1: Phase-Space Trajectory & Peak Infection Derivations')
        l = l.replace('## Member 1 Advanced Mathematical Contribution: Herd Immunity & 2D Parameter Space Bounds',
                      '## Module 1: Herd Immunity & 2D Parameter Space Bounds')
        
        # Code comments & prints
        l = l.replace('# MEMBER 1 SETUP:', '# MODULE 1 SETUP:')
        l = l.replace('# MEMBER 1 VISUALIZATION:', '# MODULE 1 VISUALIZATION:')
        l = l.replace('# MEMBER 1 CODE: SIMULATION', '# MODULE 1 CODE: SIMULATION')
        l = l.replace('# MEMBER 1 CODE: PHASE-SPACE', '# MODULE 1 CODE: PHASE-SPACE')
        l = l.replace('# MEMBER 1 CODE: CRITICAL', '# MODULE 1 CODE: CRITICAL')
        l = l.replace('# MEMBER 1 SIMULATION DEMO 1:', '# MODULE 1 SIMULATION DEMO 1:')
        l = l.replace('# MEMBER 1 SIMULATION DEMO 2:', '# MODULE 1 SIMULATION DEMO 2:')
        l = l.replace("Member 1 Simulation Complete:", "Module 1 Simulation Complete:")
        l = l.replace("Member 1 Verification Table:", "Module 1 Verification Table:")
        l = l.replace("Member 1 2D Parameter Contour Analysis Complete.", "Module 1 2D Parameter Contour Analysis Complete.")
        
        # Interpretations
        l = l.replace('Member 1 Interpretation of the Compartmental Flow Structure',
                      'Module 1: Interpretation of the Compartmental Flow Structure')
        l = l.replace('Member 1 Scientific Interpretation of Flaw 1 Demonstration',
                      'Module 1: Scientific Interpretation of Flaw 1 Demonstration')
        l = l.replace('Member 1 Scientific Interpretation of Flaw 2 Demonstration',
                      'Module 1: Scientific Interpretation of Flaw 2 Demonstration')
        l = l.replace("substantiates Member 1's thesis regarding Flaw 1",
                      "substantiates the Module 1 thesis regarding Flaw 1")
        l = l.replace("(as Member 4 demonstrates in Experiment E4)",
                      "(as demonstrated in Module 4, Experiment E4)")
        l = l.replace("Member 1 evaluated three seminal mathematical modeling frameworks",
                      "Module 1 evaluates three seminal mathematical modeling frameworks")
        l = l.replace("Member 1 derived the exact phase-space trajectory",
                      "Module 1 establishes the exact phase-space trajectory")
        l = l.replace("Member 1 maps the 2D contour parameter space",
                      "Module 1 maps the 2D contour parameter space")
        
        # Modules 2-5 Section Headings
        l = l.replace('## Part 2: Vectorized Numerical Solvers and Benchmark Verification (Experiment E1)',
                      '## Module 3: Vectorized Numerical Solvers and Benchmark Verification (Experiment E1)')
        l = l.replace('## Part 3: Mathematical Analysis: Reproduction Numbers, Stability & Final Size',
                      '## Module 2: Mathematical Analysis — Reproduction Numbers, Stability & Final Size')
        l = l.replace('## Part 4: Physics-Informed Neural Network (PINN) Formulation',
                      '## Module 4: Physics-Informed Neural Network (PINN) Formulation')
        l = l.replace('## Part 5: Experiment E2 - Forward PINN Synthetic SEIRD Solution',
                      '## Module 4: Experiment E2 — Forward PINN Synthetic SEIRD Solution')
        l = l.replace('## Part 6: Experiment E3 - Inverse PINN $\\beta(t)$ Recovery Under Noise',
                      '## Module 4: Experiment E3 — Inverse PINN $\\beta(t)$ Recovery Under Noise')
        l = l.replace('## Part 7: Experiment E4 - Hidden-State Latent Recovery ($s(t)$ and $e(t)$)',
                      '## Module 4: Experiment E4 — Hidden-State Latent Recovery ($s(t)$ and $e(t)$)')
        l = l.replace('## Part 8: Experiment E5 - Real Outbreak Application & Benchmark Forecasting',
                      '## Module 5: Experiment E5 — Real Outbreak Application & Benchmark Forecasting')
        l = l.replace('## Part 9: Scientific Interpretation & Critical Discussion',
                      '## Module 5: Scientific Interpretation & Critical Discussion')
        l = l.replace('## Part 10: Academic Honesty & AI Disclosure Statement',
                      '## Academic Honesty & AI Disclosure Statement')
        
        # AI Statement
        l = l.replace('Independent Verification by Team Members:', 'Independent Verification Across Modules:')
        l = l.replace('vetted by all five project members.', 'vetted across all project modules.')
        
        new_source.append(l)
    cell['source'] = new_source

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook saved successfully.")

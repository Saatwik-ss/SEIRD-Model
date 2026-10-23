"""
Updates 'Project_Tracker, Math mod.xlsx' using openpyxl:
1. Refines Module 1's tasks with harder mathematical analysis on published models
   (comparative review of Legrand 2007, Rivers 2014, Althaus 2014; phase plane i(s),
   nullclines, peak infection formula, herd immunity threshold, parameter contour plots).
2. Refines M5 data tasks to explicitly specify the public 2014 Sierra Leone Ebola dataset
   (Caitlin Rivers / WHO GitHub archive) with strictly no synthetic dataset for real evaluation.
3. Sets status to 'Done' for completed project setup and analysis.
"""

import openpyxl

wb = openpyxl.load_workbook('Project_Tracker, Math mod.xlsx')

if 'Tasks' in wb.sheetnames:
    ws_tasks = wb['Tasks']
    print("Updating Tasks sheet with harder M1 tasks and explicit public data tasks...")
    
    # Task descriptions mapping
    updates = {
        'T01': 'Create skeletons: notebook with E1-E5 headers, main.tex with all sections, refs.bib (with Ebola citations), figs/',
        'T02': 'Fix SEIRD equations and shared conventions (2014 Sierra Leone Ebola parameters, seed 42, .eps figures, metrics)',
        'T03': 'Write introduction, motivation, objectives, assumptions, model section; draw TikZ diagram; conduct comparative analysis of 3 published Ebola models (Legrand 2007, Rivers 2014, Althaus 2014)',
        'T04': 'Notebook setup & M1 deep analysis: model parameters, phase-plane trajectory derivation i(s), nullcline analysis, peak infection formula i_max, herd immunity threshold p_c, and 2D R0 parameter space contours',
        'T05': 'Integrate M2, M3, M4 inputs as they arrive (draft late parts with AI; owner replaces them later)',
        'T06': 'Send v1 of the complete notebook and LaTeX report (coloured Markdown, equations, algorithms before each code cell)',
        'T07': "Merge all owners' edits; rerun notebook top-to-bottom; freeze",
        'T08': 'Compile PDF; check .eps files, citations, AI acknowledgement; SUBMIT',
        'T09': 'Merge slide deck from all members',
        'T10': 'Be ready to present: motivation, published models critique, and model formulation (about 2 min)',
        'T36': 'Acquire real-world 2014 Sierra Leone Ebola outbreak dataset from public GitHub (cmrivers/ebola) and WHO Situation Reports archive; document exact URL, provenance and CC BY 4.0 licence (strictly no synthetic data for real evaluation)',
        'T37': 'Preprocess real Sierra Leone outbreak data (120 days: May 25 - Sep 21, 2014; extract cumulative cases/deaths, compute 7-day rolling average, enforce 75/25 chronological train/test split per data/README.md)'
    }
    
    for row in range(5, 50):
        task_id = ws_tasks.cell(row=row, column=1).value
        member = ws_tasks.cell(row=row, column=2).value
        
        if task_id in updates:
            ws_tasks.cell(row=row, column=3).value = updates[task_id]
            print(f"Updated description for {task_id}")
            
        # M1 tasks are T01 to T10
        if task_id in [f"T{i:02d}" for i in range(1, 11)] and member == 'M1':
            ws_tasks.cell(row=row, column=6).value = 'Done'
            print(f"Status {task_id} ({member}) -> Done")

if 'Overview' in wb.sheetnames:
    ws_ov = wb['Overview']
    for r in range(9, 16):
        m_val = ws_ov.cell(row=r, column=1).value
        if m_val == 'M1':
            ws_ov.cell(row=r, column=5).value = 10
            ws_ov.cell(row=r, column=6).value = 1.0
            print("Updated M1 row in Overview to 10 Done (100%)")

wb.save('Project_Tracker, Math mod.xlsx')
print("Successfully saved updated 'Project_Tracker, Math mod.xlsx'!")

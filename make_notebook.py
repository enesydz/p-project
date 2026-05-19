#!/usr/bin/env python3
"""
make_notebook.py
Reads fund_analysis_part1.py and fund_analysis_part2.py,
then writes fund_analysis.ipynb with exactly 2 code cells.
"""
import nbformat as nbf
import os

BASE = os.path.dirname(os.path.abspath(__file__))

p1 = os.path.join(BASE, "fund_analysis_part1.py")
p2 = os.path.join(BASE, "fund_analysis_part2.py")
nb_out = os.path.join(BASE, "fund_analysis.ipynb")

with open(p1, "r", encoding="utf-8") as f:
    code1 = f.read()

with open(p2, "r", encoding="utf-8") as f:
    code2 = f.read()

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.8"
    }
})

cell1_src = "# Cell 1 — Konfigürasyon + Veri Üretimi + Analiz Motoru\n" + code1
cell2_src = "# Cell 2 — Chartlar + Profesyonel Network Görselleştirmesi\n" + code2

nb.cells = [
    nbf.v4.new_code_cell(cell1_src),
    nbf.v4.new_code_cell(cell2_src),
]

with open(nb_out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook oluşturuldu: {nb_out}")
print(f"   Cell 1: {len(code1.splitlines())} satır")
print(f"   Cell 2: {len(code2.splitlines())} satır")

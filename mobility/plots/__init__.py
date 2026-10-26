"""
Figures. Matplotlib runs headless (Agg).
    diagnostic.py   raw radar maps next to MoCap (alignment, geometry, features)
    agreement.py    radar vs MoCap movement and active / inactive over time, the Category 1 states
    bouts.py        activity bouts of both sensors over time (Category 2)
    transitions.py  postural transitions of both sensors over time (Category 3)
    style.py        colours and figure defaults shared by all figures
"""
import matplotlib

matplotlib.use('Agg')

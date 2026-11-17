"""
Figures. Matplotlib runs headless (Agg).
    diagnostic.py   raw radar maps next to MoCap (alignment, geometry, features)
    agreement.py    radar vs MoCap movement and active / inactive over time, the Category 1 states
    bouts.py        activity bouts of both sensors over time (Category 2)
    transitions.py  postural transitions of both sensors over time (Category 3)
    per_person.py   per-person scores and signed errors (Categories 1-2)
    sensitivity.py  Category 1 active / inactive against the noise-floor parameters
    angle.py        radar positions from the angle against MoCap (orientation check)
    common.py       rows shared by the figures over time (labels, hatching, episodes)
    style.py        colours and figure defaults shared by all figures
"""
import matplotlib

matplotlib.use('Agg')

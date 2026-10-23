"""
Scripted activity labels -> active / inactive, to calibrate the MoCap active threshold (mobility/groundtruth.py).

Active means whole-body movement: walking, postural transitions, moving in bed. Inactive means arm-only tasks while
seated or lying (eating, drinking, pressing the wearable button), and waiting between instructions. Labels that mix both
(turn, then lie still) and the caregiver visit get no class and are not used.
"""
import numpy as np

from .config import LABEL_MARGIN_S

ACTIVE, INACTIVE = 'active', 'inactive'

LABEL_CLASS = {
    # locomotion
    'walk inside the main room': ACTIVE,
    'wander around in the room': ACTIVE,
    'walk to the kitchen and stand behind the counter': ACTIVE,
    'go back inside the main room': ACTIVE,
    'walk towards the chair': ACTIVE,
    'go to the left side of the bed': ACTIVE,
    'go to the right side of the bed': ACTIVE,
    'the person goes outside the room': ACTIVE,
    'wander around in the corridor': ACTIVE,
    # postural transitions
    'sit down on the chair': ACTIVE,
    'stand up from the chair': ACTIVE,
    'sit down on the edge of the bed': ACTIVE,
    'lay down on the bed': ACTIVE,
    'sit upright in the bed': ACTIVE,
    'go from upright to lay on the bed': ACTIVE,
    'get up and sit on the edge of the bed': ACTIVE,
    'stand up from the bed': ACTIVE,
    'act like you pick up a coin from the floor': ACTIVE,
    'lay down on the ground': ACTIVE,
    'get up and stand upright': ACTIVE,
    # moving in bed
    'perform anxious frequent movement in the bed': ACTIVE,
    # arm-only tasks, seated or lying
    'start performing eating movements': INACTIVE,
    'stop eating and put down the plate': INACTIVE,
    'start performing drinking movements': INACTIVE,
    'stop drinking and put down the glass': INACTIVE,
    'press the button on the wearable': INACTIVE,
    # no class: mixed (movement, then still), or about someone else
    'enter the room and stand on the cross on the floor': None,
    'stop performing anxious frequent movement and lay on your back': None,
    'turn on you left side and lay still': None,
    'turn on you right side and lay still': None,
    'now, turn on your left side and lie still': None,
    'now, turn on your right side and lie still': None,
    'lie on your back again and lie still': None,
    'a caregiver enters the room and goes to the patient': None,
    'the caregiver and person go to the kitchen and stand behind the counter': None,
    'the caregiver and person go back inside the main room': None,
    'the caregiver goes outside the room': None,
}


def label_class(label):
    """active / inactive / None for a label text (unknown labels: None)"""
    return LABEL_CLASS.get(label.strip().rstrip('.').lower())


def script_period(labels):
    """(start, end) of the scripted part: first label start to last label end. Outside it the operator sets up or
    clears the room."""
    return labels.start.min(), labels.end.max()


def caregiver_period(labels):
    """(start, end) of the caregiver visit, or None"""
    care = labels[labels.label.str.contains('caregiver', case=False)]
    return (care.start.min(), care.end.max()) if len(care) else None


def time_classes(t, labels, margin=LABEL_MARGIN_S):
    """class of each time t (epoch s): the class of the label it falls in; between labels, inactive (waiting) when at
    least `margin` from any label; None outside the script, during the caregiver visit, or near label edges"""
    out = np.full(len(t), None, object)
    s0, s1 = script_period(labels)
    near = np.zeros(len(t), bool)
    for _, r in labels.iterrows():
        out[(t >= r.start) & (t <= r.end)] = label_class(r.label)
        near |= (t >= r.start - margin) & (t <= r.end + margin)
    out[(t >= s0) & (t <= s1) & ~near] = INACTIVE
    out[(t < s0) | (t > s1)] = None
    care = caregiver_period(labels)
    if care:
        out[(t >= care[0]) & (t <= care[1])] = None
    return out

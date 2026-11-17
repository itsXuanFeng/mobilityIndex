"""
Radar vs MoCap, MoCap as the reference:
    agreement.py   per recording, all categories: the registered metrics of both sensors on the same valid epochs, and
                   their agreement across recordings: error for times and counts, next to a no-sensor baseline; for
                   energy, rank correlation and the error after a conversion to MoCap units fitted leaving the person
                   out
    category1.py   per epoch: AUC, and F1 with radar thresholds calibrated leave-one-subject-out; per person, per
                   scripted activity, per distance
    category2.py   bouts per epoch (inside a bout or not) and per bout (matched between radar and MoCap); the radar exit
                   and confirmation thresholds
    category3.py   postural transitions per event, posture per epoch; the radar height rule, peak Doppler quantile and
                   the comparison methods

Generalisation: when recording r (person p) is evaluated, nothing calibrated on p is used. MoCap active / inactive is
defined by gt_thresholds[p], the MoCap threshold calibrated against the activity labels without p, both for scoring r
and for training the radar threshold on the other persons' recordings.

Everything is importable from mobility.evaluation directly.
"""
from ..metrics.category2 import set_bout_levels
from .agreement import (agreement_with_baseline, energy_conversion, metric_agreement, metric_rows,
                        metric_table, paired_states)
from .category1 import (count_epoch_table, distance_residuals, epoch_pairs, epoch_scores, epoch_table,
                        loso_thresholds, per_label, per_person)
from .category2 import (BOUT_MATCH_IOU, RADAR_CONFIRM_EPS, RADAR_EXIT_DELTAS, bout_events, bout_frame,
                        bout_scores, evaluate_bouts, loso_radar_bout_levels, match_bouts)
from .category3 import (CLASSIFIERS, HAND_RULE_GRID, INNER_FOLDS, burst_classifier_fold,
                        calibrate_height_rule, calibrate_peak, fit_burst_model, hand_rule_baseline, height_epochs,
                        peak_measures, peak_pairs, per_kind_table, posture_scores, prepare_recording, radar_events,
                        transition_events, transition_scores, window_filter)

__all__ = [
    'agreement_with_baseline', 'bout_events', 'bout_frame', 'BOUT_MATCH_IOU', 'bout_scores', 'burst_classifier_fold',
    'calibrate_height_rule', 'calibrate_peak', 'CLASSIFIERS', 'count_epoch_table', 'distance_residuals',
    'energy_conversion', 'epoch_pairs', 'epoch_scores', 'epoch_table', 'evaluate_bouts', 'fit_burst_model',
    'hand_rule_baseline', 'HAND_RULE_GRID', 'height_epochs', 'INNER_FOLDS', 'loso_radar_bout_levels',
    'loso_thresholds', 'match_bouts', 'metric_agreement', 'metric_rows', 'metric_table', 'paired_states',
    'peak_measures', 'peak_pairs', 'per_kind_table', 'per_label', 'per_person', 'posture_scores',
    'prepare_recording', 'RADAR_CONFIRM_EPS', 'radar_events', 'RADAR_EXIT_DELTAS', 'set_bout_levels',
    'transition_events', 'transition_scores', 'window_filter',
]

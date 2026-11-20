"""
Personal calibration on top of leave-one-subject-out (docs/calibration.md; mobility/evaluation/personal.py): the
Category 1-3 metrics of the radar against MoCap under three setups,
    A  LOSO                              as in Categories 1-3
    B  LOSO + clip                       person and room parameters from a short labelled clip of the evaluated
                                         recording (the simulated installation session; labels only)
    C  LOSO + other recordings + clip    person parameters from the person's other recordings (MoCap), room
                                         parameters from the clip
all evaluated on the same epochs: the recording without the clip (and 2 s around it).
Run first: scripts/evaluate_category1.py, evaluate_category2.py, evaluate_category3.py (their LOSO values are setup A
and the base of B and C), and scripts/calibrate_posture.py.

    python scripts/evaluate_personal_calibration.py --workers 16

Writes to OUTPUT_ROOT/personal/:
    shrinkage.csv   per left-out person: the shrinkage weights chosen on the other persons
    params.csv      per recording and setup: every personal parameter used
    metrics.csv     per recording, setup and metric: radar and MoCap value
    scores.csv      per recording and setup: F1 of the active calls, bout F1, posture F1, transition F1
    summary.csv     per metric and setup: MAE, bias, relative MAE; paired against A: mean change of the absolute
                    error, share of recordings better / worse, Wilcoxon signed-rank p
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import BoutRule  # noqa: E402
from mobility.cli import error_summary, parallel_map, wide_tables  # noqa: E402
from mobility.config import CATEGORY1_THRESHOLDS, EPOCH_FRAMES, EPOCH_S, OUTPUT_ROOT, category_dir  # noqa: E402
from mobility.evaluation import personal as pc  # noqa: E402
from mobility.evaluation.category3 import posture_scores  # noqa: E402
from mobility.labels import ACTIVE, INACTIVE  # noqa: E402
from mobility.posture import UNKNOWN, epoch_posture, posture_rules, radar_posture_rules  # noqa: E402
from mobility.utils import fold_of, spread  # noqa: E402

OUT = os.path.join(OUTPUT_ROOT, 'personal')
# filled before the pools fork
DATA, BY_PERSON, LOSO, OWN, LEVELS, HEIGHTS = {}, {}, {}, {}, {}, {}


def prepare(job):
    th, mrule = job
    try:
        return pc.prepare(th, mrule), None
    except Exception:
        return None, f'{th.rec}: {error_summary()}'


def others_of(d):
    return [x for x in BY_PERSON[d['person']] if x['rec'] != d['rec']]


def own_params(rec):
    """setup C, person parameters of one recording from the person's other recordings (the Category 2 levels follow
    later, in own_bout_levels: they need the shrunk threshold)"""
    d = DATA[rec]
    p, L = d['person'], LOSO[d['person']]
    rest = others_of(d)
    thr = pc.person_threshold(rest, L['mocap_thr'])
    return rec, dict(own_thr=thr, own_energy_factor=pc.energy_factor(rest),
                     own_h_stand=pc.person_h_stand(rest, L['rule3'].offaxis_k),
                     own_levels=pc._ref([LEVELS[x['rec']] for x in rest], (ACTIVE, INACTIVE)))


# ---- personal values of one recording, given a shrinkage weight and the persons excluded from the reference ----

def ref_levels(exclude):
    return pc._ref([LEVELS[r] for r, d in DATA.items() if d['person'] not in exclude], (ACTIVE, INACTIVE))


def ref_heights(exclude, k):
    return pc._ref([HEIGHTS[r, k] for r, d in DATA.items() if d['person'] not in exclude], pc.CLIP_POSTURES)


def own_bout_levels(d, w_own):
    """setup C, Category 2: radar exit delta and confirmation eps fitted on the person's other recordings around the
    entry setup C uses for the person there: LOSO + w_own x (the person's own threshold - LOSO), the clip's room shift
    being specific to the evaluated recording"""
    L = LOSO[d['person']]
    entry = L['radar_thr'] + w_own * (OWN[d['rec']]['own_thr'] - L['radar_thr'])
    return pc.person_bout_levels(others_of(d), entry, L['mocap_thr'], L['mocap_factors'], L['bout_rule'])


def threshold(d, setup, w, ref=None, w_own=0.0):
    """the setup's radar active threshold. B: LOSO + w x the clip shift (vs the reference persons' clips). C: LOSO +
    w_own x (the person's other recordings - LOSO), then + w x the clip shift (vs the person's own other clips)"""
    L = LOSO[d['person']]
    if setup == 'A':
        return L['radar_thr']
    if setup == 'B':
        return L['radar_thr'] + w * pc.level_shift(LEVELS[d['rec']], ref)
    o = OWN[d['rec']]
    return (L['radar_thr'] + w_own * (o['own_thr'] - L['radar_thr'])
            + w * pc.level_shift(LEVELS[d['rec']], o['own_levels']))


def energy_factor(d, setup, w_own=0.0):
    """the setup's energy conversion factor: LOSO in A and B; C: LOSO + w_own x (the person's other recordings -
    LOSO)"""
    L = LOSO[d['person']]
    if setup != 'C':
        return L['energy_factor']
    return L['energy_factor'] + w_own * (OWN[d['rec']]['own_energy_factor'] - L['energy_factor'])


def energy_error(d, factor):
    """|radar converted - MoCap| total energy on the evaluated epochs"""
    e = d['eval']
    return abs(factor * np.nansum(d['radar_energy'][e]) - np.nansum(d['mocap_energy'][e]))


def posture_rule(d, setup, w, ref=None, w_own=0.0):
    """the setup's RadarPostureRule: B and C shift h_sit and h_floor (B also h_stand) by w x the clip shifts; C's
    h_stand is LOSO + w_own x (the person's other recordings - LOSO)"""
    base = LOSO[d['person']]['rule3']
    if setup == 'A':
        return base
    shifts = pc.height_shifts(HEIGHTS[d['rec'], base.offaxis_k], ref)
    if setup == 'B':
        return pc.shifted_rule(base, shifts, w)
    own = OWN[d['rec']]['own_h_stand']
    return pc.shifted_rule(base, shifts, w, base.h_stand + (0.0 if np.isnan(own) else w_own * (own - base.h_stand)))


def posture_f1(d, rule):
    E = len(d['t_epoch'])
    rp = pc._fit(epoch_posture(rule.classify(d['height'], d['offaxis']), EPOCH_FRAMES), E, UNKNOWN)
    return posture_scores(rp, d['mocap_posture'], pc.height_epochs(d, rule.offaxis_k), d['eval'])['f1_mean']


def choose_weights(p):
    """per left-out person, on every other person q (reference clips without p and q): the weight with the lowest
    total active-time error (radar threshold, B and C; for C first the weight of the person's other recordings with
    no clip shift, then the clip's weight), the lowest total energy error (energy factor, C) and the highest mean
    posture F1 (height thresholds, B and C; for C first the clip shifts with h_stand at LOSO, then the weight of
    h_stand from the other recordings). Ties: the smallest weight. Each q uses its own LOSO base, which was
    calibrated with p among the training persons."""
    train = [q for q in BY_PERSON if q != p]
    lv = {q: ref_levels({p, q}) for q in train}
    ht = {q: {k: ref_heights({p, q}, k) for k in {LOSO[q]['rule3'].offaxis_k}} for q in train}
    out = dict(person=p)

    def time_error(setup, w, w_own=0.0):
        return sum(pc.active_time_error(d, threshold(d, setup, w, lv[q], w_own), LOSO[q]['mocap_thr'])
                   for q in train for d in BY_PERSON[q])
    err = {w: time_error('B', w) for w in pc.SHRINK_GRID}
    out['w_threshold_B'] = min(pc.SHRINK_GRID, key=lambda w: (round(err[w], 9), w))
    err = {w: time_error('C', 0.0, w) for w in pc.SHRINK_GRID}
    out['w_own_threshold_C'] = min(pc.SHRINK_GRID, key=lambda w: (round(err[w], 9), w))
    err = {w: time_error('C', w, out['w_own_threshold_C']) for w in pc.SHRINK_GRID}
    out['w_threshold_C'] = min(pc.SHRINK_GRID, key=lambda w: (round(err[w], 9), w))
    err = {w: sum(energy_error(d, energy_factor(d, 'C', w)) for q in train for d in BY_PERSON[q])
           for w in pc.SHRINK_GRID}
    out['w_own_energy_C'] = min(pc.SHRINK_GRID, key=lambda w: (round(err[w], 9), w))
    for setup in ('B', 'C'):
        f1 = {w: np.nanmean([posture_f1(d, posture_rule(d, setup, w, ht[q][LOSO[q]['rule3'].offaxis_k]))
                             for q in train for d in BY_PERSON[q]]) for w in pc.SHRINK_GRID}
        out[f'w_height_{setup}'] = max(pc.SHRINK_GRID, key=lambda w: (round(f1[w], 9), -w))
    f1 = {w: np.nanmean([posture_f1(d, posture_rule(d, 'C', out['w_height_C'], ht[q][LOSO[q]['rule3'].offaxis_k],
                                                    w_own=w)) for q in train for d in BY_PERSON[q]])
          for w in pc.SHRINK_GRID}
    out['w_own_h_stand_C'] = max(pc.SHRINK_GRID, key=lambda w: (round(f1[w], 9), -w))
    return out


def evaluate(job):
    """one recording, every setup: the parameters, the metric rows, the scores"""
    rec, wts = job
    d = DATA[rec]
    p, L = d['person'], LOSO[d['person']]
    lv, ht = ref_levels({p}), ref_heights({p}, L['rule3'].offaxis_k)
    params, metrics, scores = [], [], []
    own_levels = own_bout_levels(d, wts.get('w_own_threshold_C', 0))
    for setup in pc.SETUPS:
        thr = threshold(d, setup, wts.get(f'w_threshold_{setup}', 0), lv, wts.get('w_own_threshold_C', 0))
        rule = posture_rule(d, setup, wts.get(f'w_height_{setup}', 0), ht, wts.get('w_own_h_stand_C', 0))
        delta, eps = own_levels if setup == 'C' else (L['delta'], L['eps'])
        factor = energy_factor(d, setup, wts.get('w_own_energy_C', 0))
        params.append(dict(rec=rec, person=p, setup=setup, radar_threshold=thr, energy_factor=factor, delta=delta,
                           eps=eps, h_stand=rule.h_stand, h_sit=rule.h_sit, h_floor=rule.h_floor,
                           offaxis_k=rule.offaxis_k, min_hold_s=rule.min_hold_s))
        e, mthr = d['eval'], L['mocap_thr']
        ra, ma = d['radar_level'][e] > thr, d['mocap_level'][e] > mthr
        rows = [('active_time', ra.sum() / 60, ma.sum() / 60), ('inactive_time', (~ra).sum() / 60,
                                                                 (~ma).sum() / 60),
                ('total_activity_energy', factor * np.nansum(d['radar_energy'][e]) * EPOCH_S,
                 np.nansum(d['mocap_energy'][e]) * EPOCH_S)]
        b_rows, b_sc = pc.bout_results(d, thr, mthr, delta, eps, L['mocap_factors'], L['bout_rule'])
        rows += [(r['metric'], r['radar'], r['mocap']) for r in b_rows]
        ps, ts, times = pc.posture_results(d, rule)
        rows += [(k, *v) for k, v in times.items()]
        metrics += [dict(rec=rec, person=p, setup=setup, metric=k, radar=r, mocap=m) for k, r, m in rows]
        scores.append(dict(rec=rec, person=p, setup=setup, f1_active=f1_score(ma, ra), bout_f1=b_sc['bout_f1'],
                           posture_f1=ps['f1_mean'], transition_f1=ts['f1']))
    return params, metrics, scores


def paired(a, x):
    """mean change, share better / worse, Wilcoxon p, of x against a (paired, lower is better)"""
    diff = x - a
    nz = diff[np.abs(diff) > 1e-12]
    return dict(change=diff.mean(), better=(diff < -1e-12).mean(), worse=(diff > 1e-12).mean(),
                wilcoxon_p=wilcoxon(nz).pvalue if len(nz) > 5 else np.nan)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--persons', nargs='+', help='only these persons (a quick check; at least 3)')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wide_tables(max_columns=30)
    th = pd.read_csv(CATEGORY1_THRESHOLDS)
    if args.persons:
        th = th[th.person.isin(args.persons)]
    mrules, rrules = posture_rules(), radar_posture_rules()
    c2 = pd.read_csv(os.path.join(category_dir(2), 'rules.csv')).groupby('person').first()

    res = parallel_map(prepare, [(r, fold_of(mrules, r.person)) for _, r in th.iterrows()], args.workers)
    failed = [e for _, e in res if e]
    if failed:
        print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
    DATA.update({d['rec']: d for d, e in res if not e})
    for d in DATA.values():
        BY_PERSON.setdefault(d['person'], []).append(d)
    clip_s = [d['clip'].sum() for d in DATA.values()]
    eval_s = [d['eval'].sum() for d in DATA.values()]
    print(f'{len(DATA)} recordings, {len(BY_PERSON)} persons; clip {spread(clip_s, ".0f")} s, evaluated '
          f'{spread(eval_s, ".0f")} s per recording', flush=True)

    per = th.groupby('person').first()
    for p in BY_PERSON:
        r = c2.loc[p]
        LOSO[p] = dict(radar_thr=per.radar_threshold[p], mocap_thr=per.mocap_threshold[p], delta=r.radar_exit_delta,
                       eps=r.radar_confirm_eps, mocap_factors=(r.mocap_exit_factor, r.mocap_confirm_factor),
                       bout_rule=BoutRule(int(round(r.max_gap_s / EPOCH_S)), int(round(r.min_bout_s / EPOCH_S))),
                       rule3=fold_of(rrules, p),
                       energy_factor=pc.energy_factor([d for d in DATA.values() if d['person'] != p]))
    ks = {L['rule3'].offaxis_k for L in LOSO.values()}
    for r, d in DATA.items():
        LEVELS[r] = pc.clip_levels(d)
        for k in ks:
            HEIGHTS[r, k] = pc.clip_heights(d, k)

    OWN.update(dict(parallel_map(own_params, list(DATA), args.workers)))
    print('setup C person parameters done', flush=True)
    wts = pd.DataFrame(parallel_map(choose_weights, sorted(BY_PERSON), args.workers))
    wts.to_csv(f'{OUT}/shrinkage.csv', index=False)
    print(f'shrinkage weights chosen per left-out person:\n'
          f'{wts.drop(columns="person").apply(lambda c: c.value_counts()).fillna(0).astype(int).to_string()}',
          flush=True)
    wof = wts.set_index('person').to_dict('index')

    out = parallel_map(evaluate, [(r, wof[d['person']]) for r, d in DATA.items()], args.workers)
    params = pd.DataFrame([x for o in out for x in o[0]])
    mt = pd.DataFrame([x for o in out for x in o[1]])
    sc = pd.DataFrame([x for o in out for x in o[2]])
    params.to_csv(f'{OUT}/params.csv', index=False)
    mt.to_csv(f'{OUT}/metrics.csv', index=False)
    sc.to_csv(f'{OUT}/scores.csv', index=False)

    rows = []
    mt['abs_err'] = (mt.radar - mt.mocap).abs()
    for k, g in mt.groupby('metric', sort=False):
        a = g[g.setup == 'A'].set_index('rec').abs_err
        for s, gs in g.groupby('setup'):
            gs = gs.set_index('rec')
            row = dict(metric=k, setup=s, n=int(gs.abs_err.notna().sum()), mae=gs.abs_err.mean(),
                       bias=(gs.radar - gs.mocap).mean(), rel_mae=gs.abs_err.mean() / gs.mocap.abs().mean())
            if 'energy' in k:                  # different units on both sides: also the rank agreement
                row['spearman'] = spearmanr(gs.radar, gs.mocap, nan_policy='omit')[0]
            if s != 'A':
                j = a.index.intersection(gs.index[gs.abs_err.notna() & a.reindex(gs.index).notna()])
                row.update({f'{c}_vs_A': v for c, v in paired(a[j], gs.abs_err[j]).items()})
            rows.append(row)
    for k in ('f1_active', 'bout_f1', 'posture_f1', 'transition_f1'):
        a = sc[sc.setup == 'A'].set_index('rec')[k]
        for s, gs in sc.groupby('setup'):
            x = gs.set_index('rec')[k]
            row = dict(metric=k, setup=s, n=int(x.notna().sum()), median=x.median(), mean=x.mean())
            if s != 'A':
                j = a.index[a.notna() & x.reindex(a.index).notna()]
                row.update({f'{c}_vs_A': v for c, v in paired(-a[j], -x[j]).items()})   # higher is better
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(f'{OUT}/summary.csv', index=False)
    print(f'\n{summary.round(4).to_string(index=False)}')
    print(f'\npersonal parameters, median [10th-90th percentile] over recordings:')
    for s, g in params.groupby('setup'):
        print(f'  {s}: ' + ', '.join(f'{c} {spread(g[c])}' for c in ('radar_threshold', 'energy_factor', 'delta',
                                                                      'eps', 'h_stand', 'h_sit', 'h_floor')))

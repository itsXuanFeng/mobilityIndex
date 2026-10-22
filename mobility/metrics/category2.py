"""
Category 2: activity bout structure. NOT IMPLEMENTED YET.

Planned metrics (SCOPE.md), all from the active / inactive sequence of mobility.states.States, so radar and MoCap get
the same rules:
    n_bouts               number of active bouts
    bout_duration_mean    s
    bout_duration_var     s^2
    fragmentation_index   e.g. bouts per active hour, or the active -> inactive transition probability

Shared bout rules (hysteresis, minimum bout length, merging short gaps) belong in one helper here, e.g.
bouts(states) -> list of (start, end) epochs, used by all four metrics. Then:

    @metric('n_bouts', category=2, unit='count')
    def n_bouts(s):
        return float(len(bouts(s)))

Caveat: the recordings follow a script, so the bout structure is imposed; validate mainly per epoch.
"""

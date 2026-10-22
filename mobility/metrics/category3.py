"""
Category 3: postural transitions. NOT IMPLEMENTED YET.

Planned metrics (SCOPE.md):
    peak_transition_doppler   max |radial velocity| during a transition (m/s)
    sit_to_stand_count, stand_to_sit_count, lie_to_sit_count

Needs, in addition to Categories 1-2:
    - States fields for posture and transition events (mobility/states.py), filled from MoCap (head / pelvis height,
      trunk and knee angles; labels only for lie-to-sit) and from the radar;
    - radar features already available: v_centroid (direction: + = away = going down), r_near (height of the top of the
      body = radar height - r_near on axis); still to add: angle / 3-D position (AOP array) for off-centre positions;
    - the radar position in config.RADAR_XYZ (fitted, scripts/fit_radar_position.py).
"""

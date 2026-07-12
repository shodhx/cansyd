"""Status fusion: the one real feedback path - physics verdict can override
network confidence."""


def fuse(verdict, cnn_conf, conf_thresh=0.90):
    if verdict == 'CONFLICT':
        return 'MANUAL_REVIEW'
    if verdict == 'CONFIRMED' and cnn_conf >= conf_thresh:
        return 'HIGH_CONFIDENCE'
    if verdict == 'CONFIRMED':
        return 'RELIABLE'
    return 'UNCERTAIN'

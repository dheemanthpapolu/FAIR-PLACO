import numpy as np


def posterior_estimation(hcm_list, mpv):
    n_cls = len(mpv)
    y_h_star = []
    for hcm in hcm_list:
        plist = []
        for y_h in range(n_cls):
            tot = 0.0
            for y in range(n_cls):
                tot += hcm[y_h][y] * mpv[y]
            plist.append(tot)
        y_h_star.append(int(np.argmax(plist)))
    return None, y_h_star

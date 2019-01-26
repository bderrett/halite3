# Lookups for converting actions between different representations
ACTION_CHR_IDX = {"o": 0, "n": 1, "s": 2, "e": 3, "w": 4, "d": 5}
ACTION_IDX_CHR = {idx: chr for chr, idx in ACTION_CHR_IDX.items()}
ACTION_CHR_DIR = {"o": (0, 0), "n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}
ACTION_IDX_DIR = {
    idx: ACTION_CHR_DIR[ACTION_IDX_CHR[idx]]
    for idx in ACTION_IDX_CHR
    if ACTION_IDX_CHR[idx] in ACTION_CHR_DIR
}
ACTION_DIR_IDX = {v: k for k, v in ACTION_IDX_DIR.items()}
ACTION_CHR_MRK = {"o": ".", "n": "^", "e": ">", "w": "<", "s": "v", "d": "X"}
ACTION_IDX_MRK = {idx: ACTION_CHR_MRK[ACTION_IDX_CHR[idx]] for idx in ACTION_IDX_CHR}

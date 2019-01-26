#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd


def calc_normalization(map_width, num_players):
    print(map_width, num_players)
    ddir = f"examples/{map_width}/{num_players}/"
    if os.path.exists(ddir + "vec_input_std.pickle"):
        return
    examples = np.load(ddir + "data.npz")
    print("a")
    mi = examples["map_input"]
    map_means = mi.mean(0).mean(0).mean(0)  # 9 or 17 elems
    pd.to_pickle(map_means, ddir + "map_input_mean.pickle")

    print("b")
    np.square(mi, out=mi)
    map_stds = np.sqrt(mi.mean(0).mean(0).mean(0))
    print("c")
    pd.to_pickle(map_stds, ddir + "map_input_std.pickle")

    vi = examples["vec_input"]
    vec_means = vi.mean(0)
    pd.to_pickle(vec_means, ddir + "vec_input_mean.pickle")
    vec_stds = np.sqrt((vi ** 2).mean(0))
    pd.to_pickle(vec_stds, ddir + "vec_input_std.pickle")


def calc_normalizations():
    for map_width in (32, 40, 48, 56, 64):
        for num_players in (2, 4):
            calc_normalization(map_width, num_players)


if __name__ == "__main__":
    calc_normalizations()

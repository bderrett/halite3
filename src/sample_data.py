#!/usr/bin/env python3
import numpy as np


def sample_data(map_width, num_players):
    ddir = f"examples/{map_width}/{num_players}/"
    examples = np.load(ddir + "data.npz")  # shuffled
    to_save = {k: v[:100] for k, v in examples.items()}
    np.savez_compressed(ddir + "data_sample.npz", **to_save)


def sample_data_all():
    for map_width in (32, 40, 48, 56, 64):
        for num_players in (2, 4):
            print(map_width, num_players)
            sample_data(map_width, num_players)


if __name__ == "__main__":
    sample_data_all()

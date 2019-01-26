#!/usr/bin/env python3
import json
import os
import subprocess
import concurrent.futures
import tqdm


def download(match_id, num_players, map_size):
    filename = f"training_replays/{map_size}/{num_players}/{match_id}.hlt"
    if os.path.exists(filename):
        return
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    command = (
        "wget",
        "-q",
        f"https://api.2018.halite.io/v1/api/user/0/match/{match_id}/replay",
        "-O",
        filename,
    )
    # print(' '.join(command))
    subprocess.check_output(command)


def download_all():
    array = json.load(open("all_replays_valid.json"))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(10) as pool:
        for replay in array:
            future = pool.submit(
                download, replay["game_id"], replay["num_players"], replay["map_width"]
            )
            futures.append(future)
        for future in tqdm.tqdm(futures):
            future.result()


if __name__ == "__main__":
    download_all()

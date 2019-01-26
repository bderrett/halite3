#!/usr/bin/env python3
"""Convert a replay file into states and actions for training.

Usage:
    process

When run, it will convert any unconverted replays in the `training_replays`
dir.

Note
----
The JSON is strangely formatted, so be on the lookout for off-by-one errors!

State
-----
For each player:
  * ship locations
  * ship halite
  * factory location
  * drop-off locations
* Tile Halite amounts

* Turns remaining
* Player scores

Action
------
Ship actions for each player (x, y) -> Action
Spawn indicator for each player

Reward
------
Ranking of the players


"""
# pylint: disable=c-extension-no-member
from collections import defaultdict, namedtuple
from pickle import dump
import glob
import json
import numpy as np
import os
import pandas as pd
import subprocess
import tqdm
import zstd

from shipstate import Actions, State
from wrapped import WrappedPosition, ModularInt, parse_initial_production
import actions

MAX_HALITE = 1000


def process_file(replay_file):
    """Generate states for all players in the game.

    Yields
    ------
    List of GameFrames
    """
    out_path = replay_file.replace("training_replays", "processed_replays")
    if os.path.exists(out_path):
        return

    replay = json.loads(zstd.loads(open(replay_file, "rb").read()))
    production_arr = parse_initial_production(replay["production_map"])
    width, height = production_arr.shape

    factory_locations = {}
    for player in replay["players"]:
        player_id = player["player_id"]
        factory_loc = player["factory_location"]
        factory_locations[player_id] = (factory_loc["x"], factory_loc["y"])

    game_constants = replay["GAME_CONSTANTS"]
    map_width = game_constants["DEFAULT_MAP_WIDTH"]
    num_players = replay["number_of_players"]

    num_turns = game_constants["MAX_TURNS"]

    state_actions = []
    dropoff_locs = {player_id: [] for player_id in range(num_players)}
    for frame_num, frame in enumerate(replay["full_frames"]):
        # Look carefully at the JSON to understand this.
        # To get the state for (0-indexed) turn i, use the energy from frame i
        # and the entities from frame i + 1
        if frame_num == len(replay["full_frames"]) - 1:
            break
        next_frame = replay["full_frames"][frame_num + 1]

        # TODO: see why the max is needed
        turns_remaining = max(num_turns - frame_num, 0)

        # player -> (ship_id -> position)
        ship_ids = {player_id: {} for player_id in range(num_players)}

        # player -> ship_id mutable
        player_ship_set = {player_id: set() for player_id in range(num_players)}

        # player_id -> (position -> energy)
        ship_pos_energy = {player_id: {} for player_id in range(num_players)}
        for player_id, entities in next_frame["entities"].items():
            player_id = int(player_id)
            for ship_id, ship_info in entities.items():
                ship_id = int(ship_id)
                player_ship_set[player_id].update([ship_id])
                x, y = ship_info["x"], ship_info["y"]
                ship_ids[player_id][ship_id] = x, y
                ship_pos_energy[player_id][(x, y)] = ship_info["energy"]

        spawns = []
        moves = defaultdict(dict)
        constructions = {player_id: [] for player_id in range(num_players)}

        scores = {
            int(player_id_str): energy
            for player_id_str, energy in frame["energy"].items()
        }

        # Update the Halite map.
        for cell in frame["cells"]:
            production_arr[cell["x"], cell["y"]] = cell["production"]

        for player_id_str, player_moves in next_frame["moves"].items():
            player_id = int(player_id_str)
            for move in player_moves:
                if move["type"] == "g":
                    assert scores[player_id] >= 1000
                    spawns.append(player_id)
                elif move["type"] == "m":
                    ship_id = move["id"]
                    player_ship_set[player_id].remove(ship_id)
                    position = ship_ids[player_id][ship_id]
                    moves[player_id][position] = actions.ACTION_CHR_DIR[
                        move["direction"]
                    ]
                elif move["type"] == "c":
                    loc = frame["entities"][player_id_str][str(move["id"])]
                    constructions[player_id].append((loc["x"], loc["y"]))

        for event in frame["events"]:
            if event["type"] == "spawn":
                # These events are handled as moves.
                continue
            elif event["type"] == "shipwreck":
                # We don't care about these.
                continue
            player_id = int(event["owner_id"])
            assert event["type"] == "construct"
            location = event["location"]
            dropoff_locs[player_id].append((location["x"], location["y"]))

        if frame_num <= num_turns:
            for player_id, p_ship_ids in player_ship_set.items():
                for ship_id in p_ship_ids:
                    position = ship_ids[player_id][ship_id]
                    # some players rely on the halite engine collecting for
                    # them automatically
                    moves[player_id][position] = (0, 0)

        action = Actions(
            spawns=spawns,
            moves={0: moves[0]},  # only consider moves for p0
            constructions={0: constructions[0]},
            num_players=num_players,
            map_width=map_width,
        )
        state = State(
            halite_map=production_arr,
            ships=ship_pos_energy,
            turns_remaining=turns_remaining,
            factory_locs=factory_locations,
            scores=scores,
            dropoff_locs=dropoff_locs,
            num_players=num_players,
        )

        state_actions.append((state, action))

    # TODO: add rewards
    sar = state_actions
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    dump(sar, open(out_path, "wb"))


def process_state_actions(args):
    map_width = args.map_width
    num_players = args.num_players
    if args.file is None:
        t = glob.glob(f"training_replays/{map_width}/{num_players}/*.hlt")
        for replay_file in tqdm.tqdm(t):
            process_file(replay_file)
    else:
        process_file(args.file)


def examples_iter(files):
    """Load examples. """
    examples_with_metadata = []
    for filename in files:
        state_actions = pd.read_pickle(filename)
        for i, (state, action) in enumerate(state_actions):
            yield (state, action, filename, i)


def process_sa(state, action, filename, turn_number):
    current_map_state, current_vec_state = state.to_arrays()
    map_action, vec_action = action.to_arrays()
    processed_sars = {
        "current_map_state": current_map_state,
        "current_vec_state": current_vec_state,
        "map_action": map_action,
        "vec_action": vec_action,
        "filename": filename,  # metadata
        "turn_number": turn_number,  # metadata
    }
    return processed_sars


def make_example_arr(files, map_width, num_players):
    available_examples = len(list(examples_iter(files)))
    num_planes = 9 if num_players == 2 else 17
    # Memory usage hack. Want full dataset to fit in memory without fiddling.
    # 4 bytes (32 bit float)
    cost_per_example = map_width ** 2 * (num_planes + 1) * 4
    max_memory = 52e9
    num_examples = min(int(max_memory / cost_per_example), available_examples)
    print(f"processing {num_examples} of {available_examples} examples.")
    num_vec_features = 1 + num_players
    result = {
        "map_input": np.full(
            (num_examples, map_width, map_width, num_planes), np.NaN, dtype=np.float32
        ),
        "vec_input": np.full(
            (num_examples, num_vec_features), np.NaN, dtype=np.float32
        ),
        "map_output": np.full(
            (num_examples, map_width, map_width), np.NaN, dtype=np.float32
        ),
        "vec_output": np.full((num_examples, num_players), np.NaN, dtype=np.float32),
        "filename": np.full((num_examples,), "", dtype="U128"),
        "turn_number": np.full((num_examples, 1), np.NaN, dtype=np.float32),
    }
    positions = np.arange(num_examples, dtype=int)
    np.random.shuffle(positions)
    for i, (state, action, filename, turn_number) in enumerate(examples_iter(files)):
        # Got RuntimeError when using tqdm here
        if i >= num_examples:
            break
        pos = positions[i]
        arrs = process_sa(state, action, filename, turn_number)
        result["map_input"][pos] = arrs["current_map_state"]
        result["vec_input"][pos] = arrs["current_vec_state"]
        result["map_output"][pos] = arrs["map_action"]
        result["vec_output"][pos] = arrs["vec_action"]
        result["filename"][pos] = filename
        result["turn_number"][pos] = turn_number
    if i < num_examples - 1:
        raise Exception("too few examples")

    return result


def get_keras_examples_args(args):
    map_width = args.map_width
    num_players = args.num_players
    return get_keras_examples(map_width, num_players, no_return=True)


def _process_examples(map_width, num_players):
    files = sorted(glob.glob(f"./processed_replays/{map_width}/{num_players}/*.hlt"))
    # load State and Action objects from the files and convert to arrays
    return make_example_arr(files, map_width, num_players)


def get_keras_examples(map_width, num_players, use_cache=True, no_return=False):
    """
    Returns
    -------
    {'map_input': ... , 'vec_input': ..., 'map_output': ..., 'vec_output': ...}
    """
    # see whether there are cached batches
    examples_filename = f"examples/{map_width}/{num_players}/data.npz"
    if use_cache:
        if os.path.exists(examples_filename):
            if no_return:
                return None
            return dict(np.load(examples_filename))
    example_arrs = _process_examples(map_width, num_players)

    print(f"saving to {examples_filename}")
    dirname = os.path.dirname(examples_filename)
    os.makedirs(dirname, exist_ok=True)
    np.savez_compressed(examples_filename, **example_arrs)
    return example_arrs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sps = parser.add_subparsers()
    p_sa = sps.add_parser("state_actions")
    p_sa.add_argument("map_width", type=int)
    p_sa.add_argument("num_players", type=int)
    p_sa.add_argument("--file", type=str)
    p_sa.set_defaults(func=process_state_actions)
    p_a = sps.add_parser("arrays")
    p_a.add_argument("map_width", type=int)
    p_a.add_argument("num_players", type=int)
    p_a.set_defaults(func=get_keras_examples_args)
    clargs = parser.parse_args()
    clargs.func(clargs)

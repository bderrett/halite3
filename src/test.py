#!/usr/bin/env python3
"""Test orientation. """
from common import *
import learners


def run_example(model, inputs, player_id):
    """Flip inputs so that P0 data is now for player_id and then get model predictions.
    """
    d = {k: v for k, v in inputs.items()}
    num_players = 4 if d["map_input"].shape[-1] == 17 else 2

    def idxs(order):
        gather_idxs = [0]
        for p in order:
            gather_idxs.extend([1 + 2 * p, 1 + 2 * p])
        for p in order:
            gather_idxs.extend([1 + 2 * num_players + p])
        for p in order:
            gather_idxs.extend([1 + 3 * num_players + p])
        return gather_idxs

    if player_id in (1, 3):
        d["map_input"] = d["map_input"][
            :, ::-1, :, idxs([1, 0, 3, 2] if num_players == 4 else [1, 0])
        ]
        d["vec_input"] = d["vec_input"][
            :, [0, 2, 1] + ([4, 3] if d["vec_input"].shape[-1] == 5 else [])
        ]
    if player_id in (2, 3):
        d["map_input"] = d["map_input"][:, :, ::-1, idxs([2, 3, 0, 1])]
        d["vec_input"] = d["vec_input"][:, [0, 3, 4, 1, 2]]
    d["player_id_input"] = np.array([player_id])
    m = int(32 / 2) - 1
    return model.predict(d)


def make_predictions(model, inputs, player_id):
    """Undo flipped model inputs. """
    ship_logits, spawn_logit = run_example(model, inputs, player_id)
    if player_id in (1, 3):
        ship_logits = ship_logits[:, ::-1, :, [0, 1, 2, 4, 3, 5]]  # flip e w
    if player_id in (2, 3):
        ship_logits = ship_logits[:, :, ::-1, [0, 2, 1, 3, 4, 5]]  # flip n s
    return ship_logits, spawn_logit


def flip_checker(map_width, num_players):
    model = learners.load_model(map_width, num_players, require_weights=False)
    inputs = {
        "map_input": np.random.RandomState(0).randn(
            1, map_width, map_width, 9 if num_players == 2 else 17
        ),
        "vec_input": np.random.RandomState(0).randn(1, num_players + 1),
    }
    m = int(map_width / 2) - 1

    unflipped_vals = run_example(model, inputs, 0)
    for player_id in range(num_players):
        predictions = make_predictions(model, inputs, player_id)
        for unflipped_output, flipped_output in zip(unflipped_vals, predictions):
            np.testing.assert_allclose(unflipped_output, flipped_output, atol=5e-3)


if __name__ == "__main__":
    flip_checker(int(sys.argv[1]), int(sys.argv[2]))

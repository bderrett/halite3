#!/usr/bin/env python3
"""Train the bot. """
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
import datetime
import glob
import itertools
import logging
import networks
from common import *
import tensorflow.keras
import tensorflow.keras.models
import time
import tqdm

import networks
import learners
import process

BATCH_SIZE = 512


def train_keras(map_width, num_players, lr, nb_epoch):
    model = learners.load_model(map_width, num_players, require_weights=False)
    model.compile(
        optimizer=Adam(lr=lr),
        loss={"map_output": learners.map_loss, "vec_output": learners.vec_loss},
    )

    examples = process.get_keras_examples(map_width, num_players)
    examples["vec_output"] = examples["vec_output"][:, 0]  # legacy dataset
    num_examples = len(examples["map_input"])
    examples["player_id_input"] = np.zeros(num_examples)
    ddir = f"models/{map_width}/{num_players}/"
    os.makedirs(ddir, exist_ok=True)
    model.fit(
        x={k: v for k, v in examples.items() if "input" in k},
        y={k: v for k, v in examples.items() if "output" in k},
        batch_size=512,
        verbose=1,
        validation_split=0.1,
        epochs=nb_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                ddir + "munetv4_weights.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                period=1,
            ),
            tf.keras.callbacks.CSVLogger(ddir + "training.log", append=True),
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("map_width", type=int)
    parser.add_argument("num_players", type=int)

    parser.add_argument("lr", type=float)
    parser.add_argument("nb_epoch", type=int)
    clargs = parser.parse_args()
    train_keras(
        map_width=clargs.map_width,
        num_players=clargs.num_players,
        lr=clargs.lr,
        nb_epoch=clargs.nb_epoch,
    )

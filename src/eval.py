#!/usr/bin/env python3
"""Evaluate a state and state-action value functions. """
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from common import *
import tensorflow.keras.models

import learners
import networks
import shipstate
import train


plt.style.use("seaborn")
plt.rc("axes", titlesize=4)


class color:
    BOLD = "\033[1m"
    END = "\033[0m"


def setup_ax(ax):
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_state(map_arr, label, metadata):
    assert map_arr.ndim == 3
    assert map_arr.shape[-1] in (9, 17)
    map_width = map_arr.shape[1]
    num_players = 2 if map_arr.shape[-1] == 9 else 4

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3, width_ratios=[20, 20, 1])
    gs.update(hspace=0.2, wspace=0.0)
    ax = fig.add_subplot(gs[:, :2])
    c0 = fig.add_subplot(gs[0, 2])
    c1 = fig.add_subplot(gs[1, 2])
    c2 = fig.add_subplot(gs[2, 2])
    c2.axis("off")

    colors = {0: "b", 1: "r", 2: "g", 3: "c"}
    colors = {
        player_id: c for player_id, c in colors.items() if player_id < num_players
    }
    patches = [
        mpatches.Patch(color=c, label=f"P{player_id}")
        for player_id, c in colors.items()
    ]
    c2.legend(handles=patches)

    ax.grid(b=False, which="both")
    ax.set_title(metadata)

    # plot halite
    halite_map = map_arr[:, :, 0].T
    hcb = ax.imshow(halite_map)
    fig.colorbar(hcb, cax=c0)

    # plot ships
    cmap = plt.get_cmap("viridis")
    for i in range(num_players):
        ship_ind_map = map_arr[:, :, 1 + 2 * i + 1]
        ship_energy_map = map_arr[:, :, 1 + 2 * i]
        xpos, ypos = np.where(ship_ind_map)
        facecolors = [cmap(ship_energy_map[x, y] / 1000.0) for x, y in zip(xpos, ypos)]
        ax.scatter(
            xpos, ypos, edgecolors=colors[i], linewidths=1, facecolors=facecolors
        )

    # plot factories
    for i in range(num_players):
        factory_map = map_arr[:, :, 1 + 2 * num_players + i]
        xpos, ypos = np.where(factory_map)
        ax.scatter(xpos, ypos, marker="*", color=colors[i])

    # plot dropoffs
    for i in range(num_players):
        dropoff_map = map_arr[:, :, 1 + 3 * num_players + i]
        xpos, ypos = np.where(dropoff_map)
        ax.scatter(xpos, ypos, marker="X", color=colors[i])

    cbar = mpl.colorbar.ColorbarBase(
        c1, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1000)
    )
    filename = f"figs/{map_width}/{num_players}/state/{label}.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def make_prediction_df(map_width, num_players, player_id):
    examples = dict(np.load(f"examples/{map_width}/{num_players}/data_sample.npz"))
    model = learners.load_model(map_width, num_players)
    print("making predictions")
    examples["player_id_input"] = np.full(len(examples["map_input"]), player_id)
    map_action, vec_action = model.predict(x=examples)
    examples["map_action_logits"] = map_action
    examples["vec_action_logits"] = vec_action
    cols = {}
    for k, v in examples.items():
        if v.ndim == 1:
            cols[(k,)] = v
        elif v.ndim == 2:
            for dim in range(v.shape[1]):
                cols[(k, dim)] = v[:, dim]
        elif v.ndim == 3:
            for dim in range(v.shape[1]):
                for dim2 in range(v.shape[2]):
                    cols[(k, dim, dim2)] = v[:, dim, dim2]
        elif v.ndim == 4:
            for dim in range(v.shape[1]):
                for dim2 in range(v.shape[2]):
                    for dim3 in range(v.shape[3]):
                        cols[(k, dim, dim2, dim3)] = v[:, dim, dim2, dim3]
    df = pd.DataFrame(cols)
    return df


def plot_action_labels(map_action_labels, ax):
    # plot action_labels
    for i in range(6):
        xpos, ypos = np.where(map_action_labels == i)
        ax.scatter(
            xpos,
            ypos,
            marker=networks.ACTION_IDX_MRK[i],
            color="b",
            zorder=11,
            alpha=0.4,
        )


def plot_predictions(model_outputs, losses, ax, axc, metadata, vec_action_labels):
    map_loss, vec_loss = losses
    print("plotting predictions")
    map_action, vec_action = model_outputs
    vec_action = np.exp(vec_action) / (np.exp(vec_action) + np.exp(-vec_action))
    map_action -= map_action.max(-1)[:, :, None]
    map_action2 = np.exp(map_action)
    map_action = map_action2 / map_action2.sum(-1)[:, :, None]
    map_width = map_action.shape[0]

    ax.set_title(
        metadata
        + f"\nspawn prob={vec_action:0.0%} spawn={bool(vec_action_labels)}"
        + f"\nmap_loss={map_loss[0]:0.2f} vec_loss={vec_loss[0]:0.2f}"
    )
    ax.set_xlim([-0.5, map_width - 0.5])
    ax.set_ylim([-0.5, map_width - 0.5])
    ax.invert_yaxis()

    patches = []
    patches_front = []
    colors = []
    colors_front = []
    for idx in range(6):
        map_single_action = map_action[:, :, idx]
        if idx == 1:  # n
            shape = np.array([(0.0, 0.0), (0.5, -0.5), (-0.5, -0.5)])
        elif idx == 2:  # s
            shape = np.array([(0.0, 0.0), (0.5, 0.5), (-0.5, 0.5)])
        elif idx == 3:  # e
            shape = np.array([(0.0, 0.0), (0.5, -0.5), (0.5, 0.5)])
        elif idx == 4:  # w
            shape = np.array([(0.0, 0.0), (-0.5, -0.5), (-0.5, 0.5)])
        offsets = []
        for x in np.arange(map_width):
            for y in np.arange(map_width):
                if idx in (1, 2, 3, 4):
                    poly = shape + np.array([x, y])

                    polygon = mpl.patches.Polygon(poly, True)
                    patches.append(polygon)
                    colors.append(map_single_action[x, y])
                else:
                    if idx == 0:
                        theta1, theta2 = 0, 180
                    if idx == 5:
                        theta1, theta2 = 180, 0
                    polygon = mpl.patches.Wedge(
                        [x, y], 0.25, theta1=theta1, theta2=theta2
                    )

                    patches_front.append(polygon)
                    colors_front.append(map_single_action[x, y])
    norm = mpl.colors.Normalize(vmin=0, vmax=map_action.max())
    p = mpl.collections.PatchCollection(patches, cmap=mpl.cm.viridis, norm=norm)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    cmap = mpl.cm.viridis
    p_front = mpl.collections.PatchCollection(
        patches_front, cmap=cmap, zorder=10, norm=norm
    )
    p_front.set_array(np.array(colors_front))
    ax.add_collection(p_front)
    cbar = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm)


def plot_actions(model_outputs, losses, label, input_data, metadata):
    map_action_labels = input_data["map_output"][0]
    vec_action_labels = input_data["vec_output"][0]
    num_players = 2 if input_data["map_input"].shape[-1] == 9 else 4
    map_width = map_action_labels.shape[0]
    assert map_action_labels.ndim == 2

    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2, width_ratios=[40, 1])
    ax = fig.add_subplot(gs[:, 0])
    # setup axis
    shipstate.setup_ax(ax)
    ax.set_xlim(-1, map_width)
    ax.set_ylim(-1, map_width)
    ax.set_xticks(np.arange(map_width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(map_width + 1) - 0.5, minor=True)
    plt.gca().invert_yaxis()
    ax.grid(b=True, which="minor")
    ax.grid(b=False, which="major")
    ax.set_axisbelow(True)

    axc = fig.add_subplot(gs[:, 1])
    plot_predictions(model_outputs, losses, ax, axc, metadata, vec_action_labels)
    plot_action_labels(map_action_labels, ax)

    filename = f"figs/{map_width}/{num_players}/{label}.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def evaluate(map_width, num_players, num_examples, player_id):
    print("loading examples")
    examples = dict(np.load(f"examples/{map_width}/{num_players}/data_sample.npz"))
    examples = {k: v[:num_examples] for k, v in examples.items()}
    examples["player_id_input"] = np.full(num_examples, player_id)
    examples["vec_output"] = examples["vec_output"][:, 0]  # legacy dataset

    map_labels = tf.placeholder(tf.int32, (1, map_width, map_width), name="map_labels")
    map_logits = tf.placeholder(
        tf.float32, (1, map_width, map_width, 6), name="map_logits"
    )
    map_loss = learners.map_loss(map_labels, map_logits)
    vec_labels = tf.placeholder(tf.float32, (1,), name="vec_labels")
    vec_logits = tf.placeholder(tf.float32, (1,), name="vec_logits")
    vec_loss = learners.vec_loss(vec_labels, vec_logits)
    sess = tf.Session()  # no context manager to avoid CancelledError
    tf.keras.backend.set_session(sess)
    print("loading model")
    model = learners.load_model(map_width, num_players)
    print("making predictions")
    model_outputs = model.predict(x=examples)
    model_outputs[1] = model_outputs[1][:, 0]  # legacy model
    for i in range(num_examples):
        print(i)
        example = {k: v[i : i + 1] for k, v in examples.items()}
        label = str(i)
        metadata = (
            " ".join(sys.argv)
            + "\n"
            + example["filename"][0]
            + f', turn {example["turn_number"][0][0]}\n'
            + str(example["vec_input"][0])
        )
        model_example_outputs = [x[i] for x in model_outputs]
        map_output, vec_output = model_example_outputs
        losses = sess.run(
            [map_loss, vec_loss],
            feed_dict={
                map_logits: map_output[None, :, :, :],
                map_labels: example["map_output"],
                vec_labels: example["vec_output"],
                vec_logits: vec_output[None],
            },
        )
        shipstate.plot_state(example["map_input"][0], label, metadata=metadata)
        plot_actions(model_example_outputs, losses, label, example, metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("map_width", type=int)
    parser.add_argument("num_players", type=int)
    parser.add_argument("num_examples", type=int)
    parser.add_argument("player_id", type=int)
    clargs = parser.parse_args()
    evaluate(
        map_width=clargs.map_width,
        num_players=clargs.num_players,
        num_examples=clargs.num_examples,
        player_id=clargs.player_id,
    )

# pylint: disable=E1129
from common import *
import networks


def map_loss(map_action_labels, map_action_logits):
    map_width = tf.shape(map_action_logits)[1]
    batch_size = tf.shape(map_action_logits)[0]
    flat_labels = tf.reshape(
        map_action_labels, (batch_size, map_width * map_width), name="flat_labels"
    )
    flat_logits = tf.reshape(
        map_action_logits, (batch_size, map_width * map_width, 6), name="flat_logits"
    )
    flat_mask = flat_labels > -1
    map_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.where(
            flat_mask,
            tf.cast(flat_labels, tf.int32),
            tf.zeros([batch_size, map_width * map_width], dtype=tf.int32),
        ),
        logits=flat_logits,
    )
    return tf.reduce_sum(
        tf.where(flat_mask, map_xent, tf.zeros([batch_size, map_width * map_width])), 1
    )


def vec_loss(vec_action_labels, vec_action_logits):
    return (
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=vec_action_labels, logits=vec_action_logits
        )
        * 10
    )  # so that loss is not tiny relative to map loss


def lossfunc(
    map_action_logits, map_action_labels, vec_action_logits, vec_action_labels
):
    return map_loss(map_action_labels, map_action_logits) + vec_loss(
        vec_action_labels, vec_action_logits
    )


def make_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("learn_cls", type=str)
    parser.add_argument("lr", type=float)
    parser.add_argument("activation", type=str)
    parser.add_argument("network", type=str)
    parser.add_argument("dropout_rate", type=float)
    parser.add_argument("optimizer", type=str)
    return parser


def get_learner(clargs):
    learn_cls = eval(clargs.learn_cls)
    return (
        learn_cls,
        {
            "lr": clargs.lr,
            "activation": clargs.activation,
            "network": clargs.network,
            "dropout_rate": clargs.dropout_rate,
            "optimizer": clargs.optimizer,
        },
    )


def load_model(map_width, num_players, require_weights=True):
    model = networks.keras_munetv4_model(map_width, num_players)
    filename = f"models/{map_width}/{num_players}/munetv4_weights.h5"
    if os.path.exists(filename):
        model.load_weights(filename)
    elif require_weights:
        raise Exception("no weights!")
    return model


def parse_args():
    clargs = make_parser().parse_args()
    return get_learner(clargs)

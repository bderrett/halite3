#!/usr/bin/env python3
"""The Q network mapping ShipState to possible ship actions.

energy
local patch
direction to factory
remaining num turns
"""
# pylint bug with tensorflow.keras import
# pylint: disable=import-error
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from common import *
import sys
from actions import *
import pickle


def pad_wrapped(tensor, padding):
    assert isinstance(padding, int)
    assert padding > 0

    vpadded = tf.concat([tensor[:, -padding:], tensor, tensor[:, :padding]], 1)
    result = tf.concat([vpadded[:, :, -padding:], vpadded, vpadded[:, :, :padding]], 2)
    return result


def pad_wrapped_single(tensor):
    vpadded = tf.concat([tensor, tensor[:, :1]], 1)
    result = tf.concat([vpadded, vpadded[:, :, :1]], 2)
    return result


class WrappedConv2D(Conv2D):
    """Leaves first 3 dims of shape unchanged.
    """

    def __init__(self, **kwargs):
        assert isinstance(kwargs["kernel_size"], int)
        assert kwargs["kernel_size"] % 2 == 1
        assert kwargs["kernel_size"] >= 3

        self.wrap_padding = kwargs["kernel_size"] // 2
        kwargs["padding"] == "valid"

        self.output_dim = kwargs["filters"]
        super(WrappedConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        return super(WrappedConv2D, self).call(pad_wrapped(inputs, self.wrap_padding))

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(WrappedConv2D, self).get_config()
        base_config["output_dim"] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = config["kernel_size"][0]  # hack
        config.pop("output_dim")
        return cls(**config)


def wrapped3x3(filters, activation):
    """Leaves first 3 dims of shape unchanged. """
    wrapped = WrappedConv2D(
        filters=filters,
        kernel_size=3,
        activation=activation,
        padding="valid",
        kernel_initializer="he_normal",
    )
    return wrapped


def double_wrapped3x3(filters, inputs, activation, batch_norm):
    """Leaves first 3 dims of shape unchanged. """
    if batch_norm:
        return BatchNormalization()(
            wrapped3x3(filters, activation)(
                BatchNormalization()(wrapped3x3(filters, activation)(inputs))
            )
        )
    else:
        return wrapped3x3(filters, activation, batch_norm)(
            wrapped3x3(filters, activation, batch_norm)(inputs)
        )


def conv_and_pool(filters, inputs, activation, batch_norm, dropout=False):
    """
    inputs : shape (b, w, w, f)

    pooled : shape (b, w // 2, w // 2, filters)
    """
    conv = double_wrapped3x3(filters, inputs, activation, batch_norm)
    if dropout:
        conv = Dropout(0.5)(conv)
    pooled = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pooled


def upsample(filters, activation, input):
    return wrapped3x3(filters, activation)(UpSampling2D(size=(2, 2))(input))


def upsample_merge_conv(
    filters, activation, input, merge_input, batch_norm, need_padding
):
    up = upsample(filters, activation, input)
    if need_padding:
        up = Lambda(
            lambda x: pad_wrapped_single(x),
            output_shape=(up.shape[1] + 1, up.shape[2] + 1, up.shape[2]),
        )(up)
    merge = concatenate([merge_input, up], axis=3)
    return double_wrapped3x3(filters, merge, activation, batch_norm)


class Tile(Layer):
    def __init__(self, multiples, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.multiples = multiples
        self.input_spec = InputSpec()

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        new_shape = []
        for dim, multiple in zip(input_shape, self.multiples):
            if dim is None:
                new_shape.append(None)
            new_shape.append(dim * multiple)
        return tf.TensorShape(new_shape)

    def call(self, inputs):
        return tf.tile(input=inputs, multiples=self.multiples)


def keras_munetv2_model(
    dropout_rate=0.1,
    pretrained_weights=None,
    map_state_size=(32, 32, 9),
    vec_state_size=(3,),
    activation="relu",
    num_filters=9,
    lr=1e-4,
    norm_input=True,
):
    """2 players. """
    assert activation in ("relu", "elu")
    num_actions = len(ACTION_IDX_CHR)
    map_means = tf.constant(
        [
            4.185_180_3e01,
            7.061_820_0e00,
            1.861_389_2e-02,
            6.188_153_3e00,
            1.667_520_0e-02,
            9.765_625_0e-04,
            9.765_625_0e-04,
            7.037_418_0e-04,
            5.857_277_4e-04,
        ],
        dtype=tf.float32,
    )
    map_stds = tf.constant(
        [
            8.260_264_6e01,
            7.096_868_9e01,
            1.364_327_4e-01,
            6.589_895_6e01,
            1.291_324_9e-01,
            3.125_000_0e-02,
            3.125_000_0e-02,
            2.652_813_3e-02,
            2.420_181_2e-02,
        ],
        dtype=tf.float32,
    )
    vec_means = tf.constant([201.18118, 45225.23, 41886.664], dtype=tf.float32)
    vec_stds = tf.constant([116.0121, 26712.072, 26868.19], dtype=tf.float32)

    map_input = Input(shape=map_state_size, name="map_input")
    vec_input = Input(shape=vec_state_size, name="vec_input")
    if norm_input:
        map_state = Lambda(
            lambda x: (x - map_means) / map_stds, output_shape=(32, 32, 9)
        )(map_input)
        vec_state = Lambda(lambda x: (x - vec_means) / vec_stds, output_shape=(3,))(
            vec_input
        )
    else:
        map_state = map_input
        vec_state = vec_input

    layers = []
    pool = map_state
    for i in range(5):
        conv, pool = conv_and_pool(
            num_filters, pool, dropout=False, batch_norm=True, activation=activation
        )
        layers.append(conv)
    flat_pool = Reshape((num_filters,))(pool)
    combined_state = concatenate([flat_pool, vec_state])
    combined_state = Dense(num_filters, activation="relu")(combined_state)
    combined_state = Dropout(dropout_rate)(combined_state)
    combined_state = BatchNormalization()(combined_state)
    combined_state = Dense(num_filters, activation="relu")(combined_state)
    combined_state = Dropout(dropout_rate)(combined_state)
    combined_state = BatchNormalization()(combined_state)
    pool = Reshape((1, 1, num_filters))(combined_state)
    for i in reversed(range(5)):
        pool = upsample_merge_conv(
            num_filters, activation, pool, layers[i], batch_norm=True
        )

    map_action = Conv2D(
        kernel_size=1, filters=6, activation="linear", name="map_output"
    )(pool)

    vec_action = Dense(2, activation=None, name="vec_output")(combined_state)
    return Model(
        inputs=[map_input, vec_input],
        outputs={"map_output": map_action, "vec_output": vec_action},
    )


def keras_munetv3_model(
    pretrained_weights=None,
    map_state_size=(32, 32, 9),
    vec_state_size=(3,),
    activation="relu",
    num_filters=24,
    lr=1e-4,
):
    """2 players. """
    assert activation in ("relu", "elu")
    num_actions = len(ACTION_IDX_CHR)

    map_means = np.array(
        [
            1.009_965_74e02,
            7.061_820_03e00,
            1.861_389_17e-02,
            6.188_153_27e00,
            1.667_520_03e-02,
            9.765_625_00e-04,
            9.765_625_00e-04,
            4.059_703_50e-04,
            3.348_567_99e-04,
        ],
        dtype=np.float32,
    )
    map_stds = np.array(
        [
            1.552_299_7e02,
            7.096_868_9e01,
            1.364_327_4e-01,
            6.589_894_1e01,
            1.291_324_9e-01,
            3.125_000_0e-02,
            3.125_000_0e-02,
            2.014_870_6e-02,
            1.829_909_3e-02,
        ],
        dtype=np.float32,
    )

    vec_means = np.array([201.18118, 14610.374, 14435.242], dtype=np.float32)
    vec_stds = np.array([232.24623, 25212.736, 24789.48], dtype=np.float32)

    map_input = Input(shape=map_state_size, name="map_input")
    vec_input = Input(shape=vec_state_size, name="vec_input")

    map_state = Lambda(lambda x: (x - map_means) / map_stds, output_shape=(32, 32, 9))(
        map_input
    )
    vec_state = Lambda(lambda x: (x - vec_means) / vec_stds, output_shape=(3,))(
        vec_input
    )
    vec_state = RepeatVector(32 * 32)(vec_state)
    vec_state = Reshape((32, 32, 3), name="constant_layers")(vec_state)
    pool = concatenate([map_state, vec_state])  # 12 planes

    layers = []
    for i in range(5):
        conv, pool = conv_and_pool(
            num_filters, pool, dropout=False, batch_norm=True, activation=activation
        )
        layers.append(conv)
    for _ in range(2):
        pool = Conv2D(num_filters, kernel_size=1, activation=activation)(pool)
        pool = BatchNormalization()(pool)
    for i in reversed(range(5)):
        pool = upsample_merge_conv(
            num_filters, activation, pool, layers[i], batch_norm=True
        )

    conv_out = Conv2D(kernel_size=1, filters=8, activation="linear")(pool)
    map_action = Lambda(
        lambda x: x[:, :, :, :6], output_shape=(32, 32, 6), name="map_output"
    )(conv_out)
    vec_action = Lambda(
        lambda x: tf.reduce_mean(x[:, :, :, 6:], [1, 2]),
        output_shape=(2,),
        name="vec_output_",
    )(conv_out)

    MIN_LOGIT = -20
    vec_action = Lambda(
        lambda x: tf.where(x[1][:, 1:] >= 1000, x[0], tf.ones_like(x[0]) * MIN_LOGIT),
        output_shape=(2,),
        name="vec_output",
    )([vec_action, vec_input])

    return Model(
        inputs=[map_input, vec_input],
        outputs={"map_output": map_action, "vec_output": vec_action},
    )


def orient_input(inputs, num_players):
    """
    Orient the inputs. The network only learns how to predict P0 actions. This
    step orients the inputs, so that the network can use its knowledge about
    how to predict P0 actions to predict the actions of other players. This is
    necessary for playing matches online.
    
    0:
    1: flipx
    2: flipy
    3: flipx flipy

    flipx
    0 -> 1
    1 -> 0
    2 -> 3
    3 -> 2

    flipy
    0 -> 2
    2 -> 0
    1 -> 3
    3 -> 1
    """
    map_state, vec_state, player_id_input = inputs
    player_id_input = player_id_input[:, 0]  # working around keras bug
    flipx = tf.math.logical_or(
        tf.equal(player_id_input, 1), tf.equal(player_id_input, 3)
    )
    vec_state_x = tf.where(
        flipx,
        tf.gather(vec_state, [0, 2, 1] + ([4, 3] if num_players == 4 else []), axis=-1),
        vec_state,
    )
    flipped_map_state = map_state[:, ::-1]
    orderx = [1, 0, 3, 2] if num_players == 4 else [1, 0]

    def make_gather_ids(order):
        gather_idxs = [0]
        for player_id in order:  # ships
            i = 1 + 2 * player_id
            gather_idxs.extend([i, i + 1])
        for player_id in order:  # factories
            gather_idxs.append(1 + 2 * num_players + player_id)
        for player_id in order:  # dropoffs
            gather_idxs.append(1 + 3 * num_players + player_id)
        return gather_idxs

    flipped_map_state = tf.gather(flipped_map_state, make_gather_ids(orderx), axis=-1)
    map_state_x = tf.where(flipx, flipped_map_state, map_state)
    flipy = tf.greater_equal(player_id_input, 2)
    if num_players == 2:
        return map_state_x, vec_state_x, flipx, flipy

    vec_state_y = tf.where(
        flipy, tf.gather(vec_state_x, [0, 3, 4, 1, 2], axis=-1), vec_state
    )
    ordery = [2, 3, 0, 1]
    flipped_map_state = map_state_x[:, :, ::-1]
    flipped_map_state = tf.gather(flipped_map_state, make_gather_ids(ordery), axis=-1)
    map_state_y = tf.where(flipy, flipped_map_state, map_state_x)
    return map_state_y, vec_state_y, flipx, flipy


def orient_output(x):
    """
    flipx
    1 -> 1
    2 -> 2
    3 -> 4
    4 -> 3

    flipy
    1 -> 2
    2 -> 1
    3 -> 3
    4 -> 4
    """
    map_output, flipx, flipy = x
    map_output_x = tf.where(
        flipx, tf.gather(map_output[:, ::-1], [0, 1, 2, 4, 3, 5], axis=-1), map_output
    )
    map_output_y = tf.where(
        flipy,
        tf.gather(map_output_x[:, :, ::-1], [0, 2, 1, 3, 4, 5], axis=-1),
        map_output_x,
    )
    return map_output_y


def keras_munetv4_model(map_width, num_players, activation="relu", num_filters=24):
    assert activation in ("relu", "elu")
    input_planes = 9 if num_players == 2 else 17
    vec_state_size = (1 + num_players,)
    map_state_size = (map_width, map_width, input_planes)
    num_actions = len(ACTION_IDX_CHR)

    map_input = Input(shape=map_state_size, name="map_input")
    vec_input = Input(shape=vec_state_size, name="vec_input")
    player_id_input = Input(shape=(1,), name="player_id_input")
    map_state, vec_state, flipx, flipy = Lambda(
        lambda x: orient_input(x, num_players),
        output_shape=[(map_width, map_width, input_planes), vec_state_size, (), ()],
    )([map_input, vec_input, player_id_input])

    ddir = f"examples/{map_width}/{num_players}/"
    map_means = pickle.load(open(ddir + "map_input_mean.pickle", "rb"))
    map_stds = pickle.load(open(ddir + "map_input_std.pickle", "rb"))
    vec_means = pickle.load(open(ddir + "vec_input_mean.pickle", "rb"))
    vec_stds = pickle.load(open(ddir + "vec_input_std.pickle", "rb"))
    map_state = Lambda(
        lambda x: (x - map_means) / map_stds, output_shape=map_state_size
    )(map_state)
    vec_state = Lambda(
        lambda x: (x - vec_means) / vec_stds, output_shape=vec_state_size
    )(vec_state)

    vec_state = RepeatVector(map_width * map_width)(vec_state)
    vec_state = Reshape(
        (map_width, map_width, vec_state_size[0]), name="constant_layers"
    )(vec_state)
    pool = concatenate(
        [map_state, vec_state]
    )  # `input_planes` + vec_state_size[0] planes

    layers = []
    need_padding = []
    num_pools = 5 if map_width < 64 else 6
    for i in range(num_pools):
        need_padding.append(bool(pool.shape[1].value % 2))
        conv, pool = conv_and_pool(
            num_filters, pool, dropout=False, batch_norm=True, activation=activation
        )
        layers.append(conv)
    for _ in range(2):
        pool = Conv2D(num_filters, kernel_size=1, activation=activation)(pool)
        pool = BatchNormalization()(pool)
    for i in reversed(range(num_pools)):
        pool = upsample_merge_conv(
            num_filters,
            activation,
            pool,
            layers[i],
            batch_norm=True,
            need_padding=need_padding.pop(),
        )

    conv_out = Conv2D(kernel_size=1, filters=6 + 1, activation="linear")(pool)
    map_action = Lambda(
        lambda x: x[:, :, :, :6], output_shape=(map_width, map_width, 6)
    )(conv_out)
    vec_action = Lambda(
        lambda x: tf.reduce_mean(x[:, :, :, 6:], [1, 2]),
        output_shape=(num_players,),
        name="vec_output_",
    )(conv_out)

    MIN_LOGIT = -20
    vec_action = Lambda(
        lambda x: tf.where(x[1][:, 1:2] >= 1000, x[0], tf.ones_like(x[0]) * MIN_LOGIT),
        output_shape=(num_players,),
        name="vec_output",
    )([vec_action, vec_input])

    map_action = Lambda(
        lambda x: orient_output(x),
        output_shape=(map_width, map_width, 6),
        name="map_output",
    )([map_action, flipx, flipy])
    return Model(
        inputs=[map_input, vec_input, player_id_input], outputs=[map_action, vec_action]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    clargs = parser.parse_args()
    model = eval(clargs.model)()
    model.summary()

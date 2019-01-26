import learners
from common import *
import train


def test(L=1.0):
    tr_batch = list(train.load_batches("mini_tr_batches.npz"))[0]
    N = 2
    tr_microbatch = {k: v[0:N] for k, v in tr_batch.items()}
    map_action_labels = tr_microbatch["map_action"]
    vec_action_labels = tr_microbatch["vec_action"]
    map_action_logits = np.zeros((N, 32, 32, 6))
    vec_action_logits = np.zeros((N, 2))
    for j in range(N):
        for i in range(6):
            for x in range(32):
                for y in range(32):
                    if map_action_labels[j, x, y] == i:
                        map_action_logits[j, x, y, i] = L
        for n in range(2):
            if vec_action_labels[j, n]:
                vec_action_logits[j, n] = L
            else:
                vec_action_logits[j, n] = -L
    map_action_logits = tf.constant(map_action_logits, dtype=tf.float32)
    map_action_labels = tf.constant(map_action_labels, dtype=tf.int32)
    vec_action_logits = tf.constant(vec_action_logits, dtype=tf.float32)
    vec_action_labels = tf.constant(vec_action_labels, dtype=tf.float32)
    loss = learners.lossfunc(
        map_action_logits, map_action_labels, vec_action_logits, vec_action_labels
    )
    with tf.Session() as sess:
        print(sess.run(loss))


if __name__ == "__main__":
    for L in np.arange(0.0, 100.0, 1.0):
        print(L)
        test(L)

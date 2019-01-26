import glob
import shipstate
import process
import time


def testspeed(map_width=64, num_players=2):
    files = glob.glob(f"./processed_replays/{map_width}/{num_players}/*.hlt")
    examples = process.examples_iter(files)
    start = time.time()
    for i, example in enumerate(examples):
        state = example[0]
        state.to_arrays()
        if i > 25:
            break
    end = time.time()
    elapsed = end - start
    print("{:0.2f}".format(elapsed))


if __name__ == "__main__":
    testspeed()

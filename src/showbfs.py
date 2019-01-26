import pandas as pd
import shipstate
import matplotlib.pyplot as plt


def showbfs():
    contents = pd.read_pickle("processed_replays/64/2/3088193.hlt")
    example_state = contents[-1][0]  # final frame (maybe drops?)
    map_state, vec_state = example_state.to_arrays()
    dists = map_state[:, :, -1]
    plt.imshow(dists.T)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().grid(False)
    plt.colorbar()
    plt.savefig("dists.pdf")


if __name__ == "__main__":
    showbfs()

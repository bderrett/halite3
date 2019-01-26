_Warning: this code was written in a context where the usual considerations of code quality, such as testing and documentation, were less important than is ordinarily the case!_

The bot
-------
This bot was written for the [Halite III competition](https://halite.io). I knew from the forums that people usually hand-code bots for competitions like this, but I was keen to try reinforcement learning. It turned out to be sufficiently difficult to get a bot working using supervised learning, so I stuck with that. The bot is, at the time of writing, at position 41 of 4014.

The main component of the bot is a neural network which models the policy of the ships. For a map of width $W$, the input to the network has shape $(W, W, P)$ and the output has shape $(W, W, 6)$, where $P$ is the number of input planes. These input planes represented: the amount of Halite on each tile, indicators of the ship positions of each player, the amount of Halite carried on ships for each player, and the locations of player dropoffs and shipyards. The 6 outputs represent the possible actions: 4 directions to move, together with collecting or building a dropoff. [I ended up ignoring the suggestions of the neural network with respect to dropoffs.]

The architecture of the network was based on the U-Net, a deep convolutional neural network architecture developed for medical image segmentation. The main changes that I made were to:

* Wrap the input before each convolution, in a way that respects the toroidal wrapping of the game.
* Reduce the number of filters to reduce the number of parameters of the model.
* Orient the input, so the network is always presented the input from the perspective of player 0.

I trained a separate copy of the network for each pairing of map width and number of players.

I sampled from the stochastic policy to get initial moves for the ships. I then modified these moves according to a few rules: ($x$ and $y$ denote parameters that depend on the map width and number of players. They were chosen by looking at the play of the top players.)

* If the ship is in the state 'returning', take the lowest cost path to the shipyard or a dropoff, as established by a breadth-first search. The cost of returning was the number of turns to get back, plus the amount of Halite on the return path divided by 1000 (to account for the cost of moving over Halite-rich tiles).
* Build a dropoff if:
    * The distance from the factory is between $x$ and $y$.
    * The distance from any opponent factory is at least $x$.
    * The number of dropoffs is at most $x$ times the initial Halite amount plus $y$.
* Avoid collisions by not moving to a tile that another ship indended to stay on.

A ship was marked as 'returning' if staying on the current tile would bring the amount of Halite on board to the ship maximum. A ship was removed from the set of returning ships when it reached a shipyard or dropoff.

There was also a map which showed, for each tile, on which turn a ship would have to return at the end of the game in order to get back to the shipyard or a dropoff on time. This was established considering the tiles adjacent to the shipyard and each dropoff and forming a tree for each one. The nodes of the tree were map tiles and there was an edge from adjacent node $a$ to $b$ when the lowest cost return from $b$ involved first moving to tile $a$. Each of these trees was top-sorted and the furthest ship on the tree was set to arrive on the final turn of the game, the second furthest ship was set to return one turn before, etc. When it was time for a ship to return according to this map, it was marked as `returning'.

A new ship was spawned if the turn number was less than $x$.

Problems encountered:

* Initially I had planned to let the neural network decide when to build a dropoff, but the placement of its dropoffs was poor.
* I also tried to get the neural network to model a policy for spawning, but I couldn't figure out how to get the same network to represent both the ship actions and spawning without overfitting the spawning action.
* I only had limited compute resources, so each network was only trained for roughly 20 epochs over the dataset (1000 games for each combination of map width and number of players).
* The training batches for the network were quite large, so instead of storing the batches as arrays, they were stored in a reduced form which was converted to the array representation during the training loop.

Things I would improve if I had more time:

* If I'd had more time, I would have loved to see whether the quality of the networks could have been improved by more training. This may have removed the need for hand-crafted logic about when to build drop-offs and about the desire to avoid collisions.
* I thought it would have been interesting to also get the neural network to learn the value function and then attempt reinforcement learning in the style of AlphaGo.

Things learned:

* Getting a bot working using supervised neural networks revealed problems with how I was processing the data. It was also an opportunity to fix errors in the neural network and to check that the neural network was able to represent strong policies. As such, it was a necessary precursor to doing reinforcement learning.
* I should get a computer with a GPU :)

This repo is the final submission, including the trained neural network parameters, so it should just run as-is in a Python3 environment where the appropriate modules are installed.

Many thanks to the Halite team for running this competition and to mlomb for the Halite statistics tool! 🐢

[Collection of post mortems and bot source code](https://forums.halite.io/t/collection-of-post-mortems-bot-source-code/1335)

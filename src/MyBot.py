#!/usr/bin/env python3
import time
import sys

sys.path.insert(1, "src")

t0 = time.time()
import hlt
from collections import defaultdict, deque
import logging

# logging.info(f"created game by {time.time() -t0:0.2f}s")

import shipstate
from common import *
from copy import copy

# logging.info(f"loaded tf by {time.time() -t0:0.2f}s")

import learners
import wrapped
import networks
import actions

# logging.info(f"game init by {time.time() -t0:0.2f}s")

###############################################################################
## LOAD MODEL
###############################################################################
warmup = len(sys.argv) >= 2 and sys.argv[1] == "warmup"
if not warmup:
    from hlt import constants
    from hlt.positionals import Direction

    game = hlt.Game()
    game_map = game.game_map
    width = game_map.width
    num_players = len(game.players)
else:
    # placeholder
    width = 32
    num_players = 2
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
model = learners.load_model(width, num_players)
logging.info(f"loaded model by {time.time() -t0:0.2f}s")
# Warm-up model
model.predict(
    {
        "map_input": np.zeros((1, width, width, 9 if num_players == 2 else 17)),
        "vec_input": np.zeros((1, 1 + num_players)),
        "player_id_input": np.array([0]),
    }
)


def wrap(position):
    if isinstance(position, tuple):
        return wrapped.WrappedPosition(position[0], position[1], width, width)
    return wrapped.WrappedPosition(position.x, position.y, width, width)


def get_commands(direction_prefs, crash_locs, turns_remaining):
    """
    ship -> int
    """
    while True:
        dropoff_ships = set()
        for ship_prefs in direction_prefs.values():
            for action_idx in ship_prefs:
                assert isinstance(action_idx, int), action_idx
            assert ship_prefs[-1] == 0

        # Assign ships to new positions
        used_positions = defaultdict(list)
        for ship, action_idxs in direction_prefs.items():
            action_idx = action_idxs[0]
            wpos = wrap(ship.position)
            if action_idx == 0:
                used_positions[wpos].append(ship)
            elif action_idx in [1, 2, 3, 4]:
                delta = actions.ACTION_IDX_DIR[action_idx]
                new_pos = wpos + wrap(delta)
                used_positions[new_pos].append(ship)
            else:
                assert action_idx == 5
                dropoff_ships.update([ship])

        done = True
        for position, ships in used_positions.items():
            if len(ships) == 1:
                continue
            assert len(ships) > 1

            if position in crash_locs:
                allow_crash = False
                for ship in ships:
                    if ship in forced_return_ships:
                        allow_crash = True
                if allow_crash:
                    continue

            for ship in ships:
                if direction_prefs[ship][0] > 0:
                    direction_prefs[ship].pop(0)
                    done = False
                    break
            else:
                raise Exception("this shouldn't happen")

        # Only allow one ship to become a dropoff
        if len(dropoff_ships) > 1:
            ship = dropoff_ships.pop()
            direction_prefs[ship].pop(0)
            done = False

        if done:
            break
    commands = []
    new_positions = []
    dropoff_cost = 0
    for ship, ship_prefs in direction_prefs.items():
        action_idx = ship_prefs[0]
        if action_idx < 5:
            direction = actions.ACTION_IDX_DIR[action_idx]
            new_positions.append(wrap(ship.position) + wrap(direction))
            commands.append(ship.move(direction))
        else:
            dropoff_cost += 4000
            commands.append(ship.make_dropoff())
    logging.info("commands={}".format(commands))
    logging.info("new_positions={}".format(new_positions))
    return commands, new_positions, dropoff_cost


def nearest_drop_map(
    map_width,
    factory_loc,
    dropoff_locs,
    halite_map,
    halite_scale=100.0,
    avoidance_map=None,
    build_return_trees=False,
    ship_ind_arr=None,
):
    """For each tile, provide number of turns to nearest drop + halite cost / halite_scale.

    The Halite cost is to be used to distinguish between routes that are equally long.

    Use BFS.
    """
    locs = [factory_loc] + dropoff_locs
    locs_copy = copy(locs)
    out_arr = np.full_like(halite_map, np.inf)
    locq = deque()
    if build_return_trees:
        import networkx

        # edges go from depos (root to leaf)
        digraph = networkx.DiGraph()
    for loc in locs:
        out_arr[loc[0], loc[1]] = 0
        locq.append(((loc[0], loc[1]), None))
        if build_return_trees:
            digraph.add_node((loc[0], loc[1]))
    del locs
    while locq:
        (loc, parent) = locq.popleft()
        if parent is not None:
            val = (
                out_arr[parent[0], parent[1]]
                + 1
                + int(halite_map[loc[0], loc[1]] / 10) / halite_scale
            )
        if parent is None or (val < out_arr[loc[0], loc[1]]):
            if (
                (avoidance_map is not None)
                and avoidance_map[loc[0], loc[1]]
                and ((loc[0], loc[1]) not in locs_copy)
            ):
                pass
            else:
                for d in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    new_loc = ((loc[0] + d[0]) % map_width, (loc[1] + d[1]) % map_width)
                    locq.append((new_loc, loc))
        if parent is not None:
            curr_val = out_arr[loc[0], loc[1]]
            if val < curr_val:
                out_arr[loc[0], loc[1]] = val
                if build_return_trees:
                    digraph.add_edge((parent[0], parent[1]), (loc[0], loc[1]))
            else:
                out_arr[loc[0], loc[1]] = curr_val
    if not build_return_trees:
        return out_arr
    else:
        return_turn_arr = np.full_like(halite_map, hlt.constants.MAX_TURNS)
        depo_nodes = (n for n in digraph.nodes() if digraph.in_degree(n) == 0)
        depo_adj_nodes = []
        for node in depo_nodes:
            successors = digraph.successors(node)
            for successor in successors:
                depo_adj_nodes.append(successor)
        from networkx.algorithms.traversal.depth_first_search import dfs_tree
        from networkx.algorithms import topological_sort

        for node in depo_adj_nodes:
            subtree = dfs_tree(digraph, node)
            root_out = list(topological_sort(subtree))
            i = 0
            for node in reversed(root_out):
                return_turn = hlt.constants.MAX_TURNS - i
                return_turn_arr[node[0], node[1]] = return_turn
                if ship_ind_arr[node[0], node[1]]:
                    i += 1
        return out_arr, return_turn_arr


# min distance from other drop or opp shipyard
# smoothed version of what I observed from teccles
DROP_DIST = {
    32: {2: (12, 25), 4: (12, 16)},
    40: {2: (13, 30), 4: (13, 20)},
    48: {2: (14, 35), 4: (14, 24)},
    56: {2: (15, 40), 4: (15, 28)},
    64: {2: (17, 45), 4: (17, 32)},
}[width][num_players]
# if turns_remaining is smaller than this, no dropoff
DROP_TURNS = 200

# fitted (see ols and dropoff count script)
DROP_ALPHA_BETA = {
    32: {2: (-1.16, 1.28e-05), 4: (-0.65, 9.25e-06)},
    40: {2: (-1.87, 1.38e-05), 4: (-1.55, 1.23e-05)},
    48: {2: (-1.77, 1.45e-05), 4: (0.95, 2.56e-06)},
    56: {2: (1.33, 6.06e-06), 4: (-3.01, 1.26e-05)},
    64: {2: (-2.26, 1.07e-05), 4: (-3.91, 1.09e-05)},
}[width][num_players]

if not warmup:
    INITIAL_HALITE = 0
    for x in range(width):
        for y in range(width):
            position = hlt.Position(x, y)
            INITIAL_HALITE += game_map[position].halite_amount
    MAX_DROPS = np.floor(DROP_ALPHA_BETA[0] + INITIAL_HALITE * DROP_ALPHA_BETA[1])


# median last spawn turn from teccles games
SPAWN_TURNS = {
    32: {2: 300, 4: 145},
    40: {2: 318, 4: 197},
    48: {2: 336, 4: 204},
    56: {2: 369, 4: 223},
    64: {2: 375, 4: 291},
}[width][num_players]

returning_ships = set()
forced_return_ships = set()
last_actions = {}
if warmup:
    exit()

me = game.me
shipyardwpos = wrap(me.shipyard.position)

logging.info(f"ready by {time.time() -t0:0.2f}s")

game.ready("MyPythonBotRL")

logging.info(f"player_id={game.my_id}")

###############################################################################
## GAME LOOP
###############################################################################
while True:
    game.update_frame()

    ###########################################################################
    ## CONSTRUCT STATE FOR POLICY NET
    ###########################################################################
    halite_amount_arr = np.empty((width, width))
    for x in range(width):
        for y in range(width):
            position = hlt.Position(x, y)
            halite_amount = game_map[position].halite_amount
            halite_amount_arr[x, y] = halite_amount

    ships = {}
    factory_locs = {}
    dropoff_locs = {}
    scores = {}
    for player_id, player in game.players.items():
        scores[player_id] = player.halite_amount
        dropoff_locs[player_id] = []
        dropoffs = player.get_dropoffs()
        for dropoff in dropoffs:
            dpos = dropoff.position
            dropoff_locs[player_id].append((dpos.x, dpos.y))
        factory_loc = player.shipyard.position
        factory_locs[player_id] = (factory_loc.x, factory_loc.y)
        ships[player_id] = {}
        player_ships = player.get_ships()
        for ship in player_ships:
            position = ship.position
            ships[player_id][(position.x, position.y)] = ship.halite_amount
    turns_remaining = hlt.constants.MAX_TURNS - game.turn_number
    state = shipstate.State(
        halite_map=halite_amount_arr,
        turns_remaining=turns_remaining,
        ships=ships,
        factory_locs=factory_locs,
        dropoff_locs=dropoff_locs,
        scores=scores,
        num_players=len(game.players),
    )
    map_state, vec_state = state.to_arrays()
    input_data = {
        "map_input": map_state[None, :, :, :],
        "vec_input": vec_state[None, :],
        "player_id_input": np.array([game.my_id]),
    }
    start = time.time()

    ###########################################################################
    ## GET ACTION PROBABILITIES
    ###########################################################################
    map_action, vec_action = [x[0] for x in model.predict(input_data)]
    sypos = me.shipyard.position
    map_action[sypos.x, sypos.y, 0] = -20  # don't stay on the shipyard
    map_action1 = map_action - map_action.max(-1)[:, :, None]
    map_action2 = np.exp(map_action1)
    probabilities = map_action2 / map_action2.sum(-1)[:, :, None]

    logging.info("computed actions in {:0.2f}s".format(time.time() - start))

    ###########################################################################
    ## GET OCCUPIED LOCATIONS
    ###########################################################################
    occupied_locs = set()
    for p in game.players.values():
        occupied_locs.update([wrap(p.shipyard.position)])
        for dropoff in p.get_dropoffs():
            occupied_locs.update([wrap(dropoff.position)])

    ###########################################################################
    ## GET DISTANCE MAP
    ###########################################################################
    def get_locs(player):
        dropoff_locs = [d.position for d in player.get_dropoffs()]
        dropoff_locs = [(p.x, p.y) for p in dropoff_locs]
        factory_loc = player.shipyard.position
        factory_loc = (factory_loc.x, factory_loc.y)
        return factory_loc, dropoff_locs

    if num_players == 4:
        opp_ind_map_idxs = []
        opp_ids = [i for i in range(4) if i != game.my_id]
        for opp_id in opp_ids:
            opp_ind_map_idxs.append(2 + 2 * opp_id)
        opp_ind = map_state[:, :, opp_ind_map_idxs].sum(2)
        opp_locs = list(
            map(wrap, [(int(x), int(y)) for x, y in zip(*np.where(opp_ind))])
        )
        adj_locs = set()
        for opp_loc in opp_locs:
            for d in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                adj_locs.update([opp_loc + wrap(d)])
            adj_locs.update([opp_loc])

    factory_loc, dropoff_locs = get_locs(me)
    distance_map, return_turn_arr = nearest_drop_map(
        width,
        factory_loc,
        dropoff_locs,
        halite_amount_arr,
        avoidance_map=opp_ind if num_players == 4 else None,
        build_return_trees=True,
        ship_ind_arr=map_state[:, :, 2 + 2 * game.my_id],
    )
    # if game.turn_number % 50 == 0:
    #     import pandas as pd

    #     pd.to_pickle(distance_map, f"/tmp/distance_map{game.turn_number}.pickle")
    #     pd.to_pickle(return_turn_arr, f"/tmp/return_arr{game.turn_number}.pickle")

    if num_players == 2:
        # We continue to use Halite scaling for the opp map, so that it's
        # comparable with our map.
        opp = game.players[1 - game.my_id]
        opp_factory_loc, opp_dropoff_locs = get_locs(opp)
        opp_distance_map = nearest_drop_map(
            width, opp_factory_loc, opp_dropoff_locs, halite_amount_arr
        )

    ###########################################################################
    ## DROP RANK
    ###########################################################################
    drop_rank = []
    dropoffs = me.get_dropoffs()
    if (turns_remaining > DROP_TURNS) and (len(dropoffs) < MAX_DROPS):
        wmspos = wrap(game.me.shipyard.position)
        for ship in me.get_ships():
            pos = ship.position
            wpos = wrap(pos)
            # -1 to take into account that the distance map also includes halite
            if distance_map[pos.x, pos.y] - 1 < DROP_DIST[0]:
                continue
            if wmspos.x.dist(wpos.x) + wmspos.y.dist(wpos.y) > DROP_DIST[1]:
                continue
            dist = np.inf
            for player in game.players.values():
                spos = wrap(player.shipyard.position)
                d1 = wpos.x.dist(spos.x)
                d2 = wpos.y.dist(spos.y)
                dist = min(d1 + d2, dist)
                logging.info(f"yy {spos} {wpos} {dist}")
            if dist < DROP_DIST[0]:
                continue
            cost = 4000 - ship.halite_amount - halite_amount_arr[pos.x, pos.y]
            drop_rank.append((ship, cost))
    drop_rank.sort(key=lambda p: p[1])
    if len(drop_rank):
        drop_ship = drop_rank.pop(0)[0]
        logging.info(f"drop ship: {drop_ship}")
    else:
        drop_ship = None

    ###########################################################################
    ## SAMPLE ACTIONS
    ###########################################################################
    def pref_iter(ship):
        pos = ship.position
        moves = list(enumerate(probabilities[pos.x, pos.y]))
        if ship.halite_amount < int(halite_amount_arr[pos.x, pos.y] / 10):
            moves = [(i, p) for i, p in moves if i not in [1, 2, 3, 4]]
        can_drop = True
        if wrap(ship.position) in occupied_locs:
            # Disallow drop-off on shipyard or dropoff
            can_drop = False
        if not can_drop:
            moves = [(i, p) for i, p in moves if i != 5]

        probs = np.array(moves)
        while len(probs):
            probs[:, 1] /= probs[:, 1].sum()
            action_idx = np.random.choice(probs[:, 0], p=probs[:, 1])
            yield action_idx
            probs = np.delete(probs, np.where(probs[:, 0] == action_idx), axis=0)

    crash_locs = [wrap(me.shipyard.position)] + list(map(wrap, dropoff_locs))
    want_drop = False

    def update_prefs(ship, ship_prefs):
        global want_drop
        ship_prefs = list(map(int, ship_prefs))

        pos = ship.position
        wpos = wrap(ship.position)
        fpos = me.shipyard.position
        wfpos = wrap(fpos)

        # Ignore drop-off advice from net
        ship_prefs = [p for p in ship_prefs if p != 5]

        # If ship is on a tile which has no Halite, prefer moving.
        if (halite_amount_arr[pos.x, pos.y] == 0) and (ship_prefs[0] != 5):
            ship_actions = [p for p in ship_prefs if p in [1, 2, 3, 4]]
            other = [p for p in ship_prefs if p not in [1, 2, 3, 4]]
            ship_prefs = ship_actions + other

        if (ship.halite_amount == 0) and (ship in returning_ships):
            returning_ships.remove(ship)
        returning = False

        # If ship will hit max, return.
        collection_amt = int(halite_amount_arr[pos.x, pos.y] / 4.0)
        if ship.halite_amount + collection_amt >= 1000:
            returning = True

        # If ship is already on the way back, return.
        if ship in returning_ships:
            returning = True

        # If ship needs to get back for the end of the game, return.
        force_return = False
        if (
            game.turn_number
            >= return_turn_arr[pos.x, pos.y] - distance_map[pos.x, pos.y] - 4
        ):
            force_return = True
            forced_return_ships.update([ship])
            returning = True

        if returning:
            returning_ships.update([ship])
            # Return ship to factory or dropoff
            action_values = []
            for action_idx in [1, 2, 3, 4]:
                new_pos = wpos + wrap(actions.ACTION_IDX_DIR[action_idx])
                return_cost = distance_map[new_pos.x.get(), new_pos.y.get()]
                action_values.append((action_idx, return_cost))
            action_values.sort(key=lambda t: t[1])

            # Originally, I use the code which is now code for 4P (v26).
            # I realized it was buggy, so I replaced it with the 2P version.
            # That version players 2P much better (v27), but appears to be
            # uniformly worse at 4P. Hence my attempt to combine them...
            # This is purely empirical speculation...
            if num_players == 2:
                ship_prefs = [t[0] for t in action_values] + [0]
            else:
                ship_prefs.extend([t[0] for t in action_values])
                ship_prefs.append(0)

        if num_players == 2:
            # 2P:
            # * if there's a ship adjacent with more halite and we're closer to
            #   our bases, ram it.
            # * if we're next to a ship with less Halite and we're closer to the
            #   enemy base, avoid it.
            # This takes precedence over returning, since otherwise we can
            # accidentally return via the opp's shipyard... 4835051 turn 312
            # Slight approximation:
            opp_id = 1 - game.my_id
            my_dist = wpos.x.dist(wfpos.x) + wpos.y.dist(wfpos.y)
            wofpos = wrap(game.players[opp_id].shipyard.position)
            opp_dist = wpos.x.dist(wofpos.x) + wpos.y.dist(wofpos.y)
            diff = opp_dist - my_dist
            opp_ind_map = map_state[:, :, 2 + 2 * opp_id]
            if diff > 1:  # further for opp
                for action_idx in (1, 2, 3, 4):
                    new_pos = wpos + wrap(actions.ACTION_IDX_DIR[action_idx])
                    new_pos_tuple = (new_pos.x.get(), new_pos.y.get())
                    if opp_ind_map[new_pos_tuple[0], new_pos_tuple[1]]:
                        opp_energy_map = map_state[:, :, 1 + 2 * opp_id]
                        if (
                            opp_energy_map[new_pos_tuple[0], new_pos_tuple[1]]
                            > ship.halite_amount
                        ):
                            ship_prefs = [action_idx] + [
                                p for p in ship_prefs if p != action_idx
                            ]
                            break
            elif diff < -1:  # further for me
                for action_idx in (1, 2, 3, 4):
                    new_pos = wpos + wrap(actions.ACTION_IDX_DIR[action_idx])
                    new_pos_tuple = (new_pos.x.get(), new_pos.y.get())
                    opp_energy_map = map_state[:, :, 1 + 2 * opp_id]
                    opp_energy = opp_energy_map[new_pos_tuple[0], new_pos_tuple[1]]
                    if opp_ind_map[new_pos_tuple[0], new_pos_tuple[1]]:
                        if opp_energy < ship.halite_amount:
                            ship_prefs = [p for p in ship_prefs if p != action_idx]
        else:
            # 4P:
            # * don't move onto tiles containing opp or adjacent to opp
            # Skip if ships are returning and close to prevent shipyard blocking
            skip = returning and distance_map[pos.x, pos.y] < 4
            skip = skip or force_return
            if not skip:
                for action_idx in (1, 2, 3, 4):
                    new_pos = wpos + wrap(actions.ACTION_IDX_DIR[action_idx])
                    if (new_pos in adj_locs) and (new_pos not in crash_locs):
                        ship_prefs = [p for p in ship_prefs if p != action_idx]

        if ship == drop_ship:
            if (
                ship.halite_amount + halite_amount_arr[pos.x, pos.y] + me.halite_amount
                < 4000
            ):
                # Does it want drop-off?
                want_drop = True
            else:
                ship_prefs = [5] + ship_prefs
                logging.info(f"drop_ship prefs: {ship_prefs}")

        assert 0 in ship_prefs
        ship_prefs = ship_prefs[: ship_prefs.index(0) + 1]
        return ship_prefs

    prefs = {ship: update_prefs(ship, list(pref_iter(ship))) for ship in me.get_ships()}
    logging.info(f"want_drop={want_drop}")
    logging.info(f"returning_ships={returning_ships}")
    logging.info(f"prefs={prefs}")
    commands, new_positions, dropoff_cost = get_commands(
        prefs, crash_locs, turns_remaining
    )

    ###########################################################################
    ## SPAWN?
    ###########################################################################
    if game.turn_number < SPAWN_TURNS:
        if (
            (me.halite_amount - dropoff_cost >= constants.SHIP_COST)
            and (wrap(me.shipyard.position) not in new_positions)
            and (not want_drop)
        ):
            commands.append(me.shipyard.spawn())
        else:
            pass
            # used to raise an exception here. some ships come back to base and
            # block it, but hopefully the nn can learn to avoid that

    logging.info("commands={}".format(commands))
    game.end_turn(commands)

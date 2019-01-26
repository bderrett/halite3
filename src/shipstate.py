from collections import defaultdict, namedtuple
from copy import deepcopy

import numpy as np
import os

import actions


class Actions:
    def __init__(self, spawns, moves, constructions, map_width, num_players):
        spawns = deepcopy(spawns)
        moves = deepcopy(moves)
        constructions = deepcopy(constructions)
        assert isinstance(spawns, list)
        assert isinstance(moves, dict)
        assert isinstance(constructions, dict)
        self.spawns = spawns

        # player_id -> (position -> direction)
        self.moves = moves

        # player_id -> position
        self.constructions = constructions
        self.map_width = map_width
        self.num_players = num_players

    def to_arrays(self):
        map_actions = np.full((self.map_width, self.map_width), -1)
        direction_lookup = {
            d: actions.ACTION_CHR_IDX[c] for c, d in actions.ACTION_CHR_DIR.items()
        }
        for player_id, player_moves in self.moves.items():
            for position, direction in player_moves.items():
                map_actions[position[0], position[1]] = direction_lookup[direction]
        for player_id, player_constructions in self.constructions.items():
            for position in player_constructions:
                map_actions[position[0], position[1]] = 5

        spawn_arr = np.zeros(self.num_players)
        for player_id in self.spawns:
            spawn_arr[player_id] = 1
        return map_actions, spawn_arr


class State:
    """
    """

    def __init__(
        self,
        halite_map,
        turns_remaining,
        ships,
        factory_locs,
        dropoff_locs,
        scores,
        num_players,
    ):
        halite_map = halite_map.copy()
        ships = deepcopy(ships)
        factory_locs = deepcopy(factory_locs)
        dropoff_locs = deepcopy(dropoff_locs)
        scores = deepcopy(scores)
        assert halite_map.shape[0] == halite_map.shape[1]
        assert turns_remaining >= 0
        assert len(ships) == num_players
        assert len(factory_locs) == num_players
        assert len(dropoff_locs) == num_players
        assert len(scores) == num_players
        # Halite map
        self.halite_map = halite_map
        # Turns remaining
        self.turns_remaining = turns_remaining
        # Position -> Halite amount
        self.ships = ships

        # Factory location
        self.factory_locs = factory_locs
        # Drop-off locations
        self.dropoff_locs = dropoff_locs
        assert len(dropoff_locs) == num_players
        # Score
        self.scores = scores
        self.num_players = num_players

        self.map_width = self.halite_map.shape[0]

    def to_arrays(self):
        """
        halite map
        (ship energy, ship ind) * num_players # 1 + 2 * p, 2 + 2 * p
        factory loc * num_players # 1 + 2 * np + p
        dropoff loc * num_players # 1 + 3 * np + p
        """
        players = range(self.num_players)
        maps = [self.halite_map]
        for player_id in players:
            ship_dict = self.ships[player_id]
            ship_indicators = np.zeros_like(self.halite_map)
            ship_energy = np.zeros_like(self.halite_map)
            for (x, y), energy in ship_dict.items():
                ship_energy[x, y] = energy
                ship_indicators[x, y] = 1
            maps.extend([ship_energy, ship_indicators])
        for player_id in players:
            factory_loc = self.factory_locs[player_id]
            factory_loc_map = np.zeros_like(self.halite_map)
            factory_loc_map[factory_loc[0], factory_loc[1]] = 1
            maps.append(factory_loc_map)
        for player_id in players:
            dropoff_locs = self.dropoff_locs[player_id]
            dropoff_loc_map = np.zeros_like(self.halite_map)
            for dropoff_loc in dropoff_locs:
                dropoff_loc_map[dropoff_loc[0], dropoff_loc[1]] = 1
            maps.append(dropoff_loc_map)
        map_arr = np.array(maps)

        other_state = [self.turns_remaining]
        for player_id in players:
            other_state.append(self.scores[player_id])
        other_arr = np.array(other_state, dtype=np.float64)
        map_arr = np.moveaxis(map_arr, 0, -1)
        assert map_arr.shape == (
            self.map_width,
            self.map_width,
            9 if self.num_players == 2 else 17,
        )
        return map_arr, other_arr

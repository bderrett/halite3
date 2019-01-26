import numpy as np


class ModularInt:
    """Integer modulo some integer"""

    def __init__(self, val, mod):
        assert isinstance(val, int)
        assert isinstance(mod, int)
        self.mod = mod
        self.val = val

    def get(self, range_start=0):
        val = self.val
        while val < range_start:
            val += self.mod
        while val >= self.mod + range_start:
            val -= self.mod
        return val

    def __sub__(self, other):
        assert self.mod == other.mod
        return ModularInt(self.val - other.val, self.mod)

    def __add__(self, other):
        assert self.mod == other.mod
        return ModularInt(self.val + other.val, self.mod)

    def __eq__(self, other):
        assert self.mod == other.mod
        return self.val % self.mod == other.val % self.mod

    def __neq__(self, other):
        return not (self == other)

    def __repr__(self):
        return f"ModularInt({self.val}, {self.mod})"

    def __hash__(self):
        return hash((self.val % self.mod, self.mod))

    def dist(self, other):
        assert self.mod == other.mod
        diff = abs(self.val - other.val)
        while diff > self.mod // 2:
            diff = (self.mod - diff) % self.mod
        return diff


class WrappedPosition:
    def __init__(self, x, y, width, height):
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        self.x = ModularInt(x, width)
        self.y = ModularInt(y, height)
        self.width = width
        self.height = height

    def __sub__(self, other):
        assert self.width == other.width
        assert self.height == other.height
        new_x = self.x - other.x
        new_y = self.y - other.y
        return WrappedPosition(new_x.val, new_y.val, self.width, self.height)

    def __add__(self, other):
        assert self.width == other.width
        assert self.height == other.height
        new_x = self.x + other.x
        new_y = self.y + other.y
        return WrappedPosition(new_x.val, new_y.val, self.width, self.height)

    def __eq__(self, other):
        assert self.width == other.width
        assert self.height == other.height
        return (self.x == other.x) and (self.y == other.y)

    def __neq__(self, other):
        return not (self == other)

    def __repr__(self):
        assert self.width == self.height
        return f"WP(x={self.x.val}, y={self.y.val}, w={self.width})"

    def __hash__(self):
        return hash((self.x, self.y))


def parse_initial_production(production_map):
    production_arr = np.full(
        (production_map["width"], production_map["height"]), np.NaN
    )
    for y, map_row in enumerate(production_map["grid"]):
        for x, production in enumerate(map_row):
            production_arr[x, y] = production["energy"]
    return production_arr

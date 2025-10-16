import os
import string

import numpy as np

from typing import List

class MapBuilder:
    '''Class to create any custom map.'''

    def __init__(self, filepath: str = None) -> None:
        self.map: List[str] = []
        self.desc: np.ndarray = self._get_desc()
        self.locs = []

        self.wall = '|'
        self.free = ':'
        self.corner = '+'
        self.lim = '-'
        self.locations = list(string.ascii_uppercase)

        self.nrows = 0
        self.ncols = 0
        self.nlocs = len(self.locs)
        self.size = 0

        if filepath is not None:
            self.load(filepath)

    def _get_desc(self) -> np.ndarray:
        return np.asarray(self.map, dtype='c')

    def _get_input_space(self):
        '''Returns the size of the input space.'''
        return self.nrows * self.ncols * self.nlocs * (self.nlocs - 1)

    def _update(self):
        self.desc = self._get_desc()
        self.locs = []
        # locations
        for i in range(1, self.desc.shape[0] - 1):
            for j in range(1, self.desc.shape[1], 2):
                if self.desc[i, j] != b' ':
                    self.locs.append((i - 1, int((j - 1) / 2)))

        self.nlocs = len(self.locs)
        self.nrows = self.desc.shape[0] - 2
        self.ncols = int((self.desc.shape[1] - 1) / 2)
        self.size = self._get_input_space()

    def create_empty(self, nrows: int, ncols: int, update: bool = True):
        '''Creates an empty map of size (@nrows, @ncols).'''
        # x = nrows + 2
        # y = 2 * ncols + 1
        border = '+' + '-'.join(['' for _ in range(2 * ncols)]) + '+'
        empty_line = '|' + ':'.join([' ' for _ in range(ncols)]) + '|'
        # assert (len(border) == y) and (len(empty_line) == y)
        self.map = [border]
        for _ in range(nrows):
            self.map.append(empty_line)
        self.map.append(border)
        if update:
            self._update()
        return

    def add_wall(self, xstart: int, ystart: int, length: int):
        assert (xstart >= 0) and (xstart <= self.nrows), 'invalid xstart'
        assert (ystart >= 0) and (ystart <= self.ncols), 'invalid ystart'
        assert (length > 0) and (length < (xstart + self.nrows)), 'wall too long'

        update = False
        for x in range(1 + xstart, 1 + xstart + length):
            if self.map[x][1 + 2 * ystart] in self.locations:
                print('add_wall: a location has been removed')
                update = True
            line = list(self.map[x])
            line[2 + 2 * ystart] = '|'
            self.map[x] = ''.join(line)

        if update:
            self._update()
        return

    def add_location(self, x, y):
        '''Adds a location.'''
        assert (x >= 0) and (x <= self.nrows), 'invalid x'
        assert (y >= 0) and (y <= self.ncols), 'invalid y'

        line = list(self.map[1 + x])
        line[1 + 2 * y] = self.locations[len(self.locs)]
        self.map[1 + x] = ''.join(line)
        self.locs.append((x, y))
        self.nlocs = len(self.locs)
        return

    def save(self, fp: str, erase: bool = False):
        if os.path.exists(fp) and (not erase):
            fp = 'copy_' + fp
        with open(fp, 'w') as f:
            [f.write(line + '\n') for line in self.map]
        return

    def load(self, fp: str):
        if not os.path.exists(fp):
            print(f'file {fp} does not exist: can\'t load.')
        with open(fp, 'r') as f:
            self.map = [line[:-1] for line in f.readlines()]
        self._update()

    def generate_input_space(self) -> np.ndarray:
        inputs = [
            np.array([i, j, k, l])
            for i in range(0, self.nrows)
            for j in range(0, self.ncols)
            for k in range(0, self.nlocs)
            for l in range(0, self.nlocs)
            if k != l]
        return np.array(inputs, dtype=int)

    def __str__(self) -> str:
        return '\n'.join(self.map)


if __name__ == '__main__':
    map = MapBuilder()
    map.load('map_large.txt')
    print(map)
    print(map.nrows, map.ncols)
    for l in map.locs:
        print(l, map.map[1 + l[0]][1 + 2 * l[1]])
    print('input space size of', map._get_input_space())
    print(len(map.generate_input_space()))
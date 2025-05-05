import numpy as np
from typing import Union

from nav_error_profile import NavErrorProfile


class Coverage:
    """
    A coverage profile
    """
    def __init__(self,
                 nep: NavErrorProfile,
                 width_m: int,
                 height_m: int,
                 step_size: float = 0.01,
                 overlap_pct: float = 0.1,
                 brush_width_m: float = 0.56
                 ):

        self._nep = nep
        self._width_cm = self.m_to_cm(width_m)
        self._height_cm = self.m_to_cm(height_m)
        brush_width_cm = self.m_to_cm(brush_width_m)

        # make a map of the area at cm precision
        self._map = np.zeros((self._height_cm, self._width_cm), dtype=int)
        self._step_size = step_size

        # calculate the spacing required to meet criteria, round to nearest cm to ensure overlap
        self._spacing = int(brush_width_cm - (brush_width_cm * overlap_pct))

        # round up on the number of legs to ensure coverage
        self._num_legs = int(round(self._height_cm / self._spacing, 0))
        self._mission_length = self._num_legs * self._width_cm

    def gen_x_walk(self, num_steps: int):
        x_history = []
        y_history = []
        for i in range(num_steps):
            # all calculations in cm
            ex, ey = self._nep.sample()
            x_history.append(ex + self._step_size)
            y_history.append(ey)
        return x_history, y_history

    def simulate(self):
        # first generate poses with error in a single direction
        # this is basically a Gillespie Event Queueing approach
        x_poses, y_poses = self.gen_x_walk(self._mission_length)
        # integer value of vehicle position (in cm!), to allow indexing into map
        v_y = int(self._spacing / 2)
        v_x = 0
        # true positions, stored as float to allow error to accrue with more precision
        t_y = v_y
        t_x = v_x

        # add the starting cleaning position
        self.add_coverage(v_x, v_y)
        for leg in range(self._num_legs):

            if leg != 0:
                # move to new leg pass if not the first pass
                e_turn_x, e_turn_y = self._nep.sample(self._spacing)
                t_y += self._spacing + e_turn_y
                t_x += e_turn_x

            # grab the posses from this leg. Note: here we make an assumption that the vehicle is always
            # commanded to fly the same number of steps regardless of error - which should be true
            x_leg = x_poses[self._width_cm * leg: (self._width_cm * (leg + 1))]
            y_leg = y_poses[self._width_cm * leg: (self._width_cm * (leg + 1))]

            for x, y in zip(x_leg, y_leg):
                # now convert to map resolution
                if leg % 2 == 0:
                    # if it's an even leg, apply x and y normally
                    # yes I know this conversion to meters seems weird but we have to convert the results
                    # to the units of the map
                    t_x += self.m_to_cm(x)
                    t_y += self.m_to_cm(y)
                else:
                    # apply in reverse (both x and y)
                    t_x -= self.m_to_cm(x)
                    t_y -= self.m_to_cm(y)

                # now convert float cm positions to int cm positions via rounding
                v_x = int(round(t_x, 0))
                v_y = int(round(t_y, 0))
                # add coverage for entire brush to new location
                self.add_coverage(v_x, v_y)
        return self._map

    def add_coverage(self, x_center: int, y_center: int):
        # first add the center point
        if self.is_in_range(y_center, x_center):
            self._map[y_center, x_center] += 1
            # then add all the width of the brush deck
            lower_y_bound, upper_y_bound = self.get_y_deck_bounds(y_center)
            # direct array slicing (rather than looping) yields a slight speedup. Maybe do this for x dim too?
            self._map[lower_y_bound: upper_y_bound, x_center] += 1

    def is_in_range(self, i, j):
        return 0 <= i < self._height_cm and 0 <= j < self._width_cm

    def get_y_deck_bounds(self, y):
        lower = max(y - self._spacing // 2, 0)
        upper = min(y + self._spacing // 2, self._height_cm - 1)
        return lower, upper

    def calc_coverage(self):
        # first convert to bool to prevent double counting of places covered twice (or more)
        bool_map = self._map.astype(bool)
        covered = np.sum(bool_map)
        return covered / (self._width_cm * self._height_cm)

    @staticmethod
    def m_to_cm(m_value: Union[float, int]) -> Union[float, int]:
        # okay I know this is kind of an obvious calculation by why not make things nice!
        return m_value * 100  # handy way to preserve type!


# TODO: remove, just for testing
if __name__ == "__main__":
    import utils
    import pathlib
    step_size = 0.01
    width_m = 10
    height_m = 10

    p = pathlib.Path("..", "data", "measured_eep_bad.json")
    cov = np.array(utils.load_profile(p)["covariance"])
    cm_cov_mat = utils.scale_cov_mat(cov, step_size)
    nep = NavErrorProfile(cm_cov_mat)
    c = Coverage(nep, width_m, height_m)
    c_map = c.simulate()

    save_name = pathlib.Path("test_single_coverage.png")
    utils.array_to_image(c_map, save_name)

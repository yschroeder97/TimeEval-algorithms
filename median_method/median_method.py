import numpy as np
import bottleneck as bn


class MedianMethod:
    def __init__(self, neighbourhood_size=100):
        self._neighbourhood_size = neighbourhood_size
        self._buffer = np.array([])
        self._window_size = neighbourhood_size * 2 + 1
    
    def update_buffer(self, new_data):
        if len(self._buffer) == 0:
            self._buffer = new_data
            print(f"Buffer init: {self._buffer}")
            return

        # Append new data and trim excess points
        if len(self._buffer) > self._window_size:
            self._buffer = self._buffer[-(self._window_size - 1):]

        self._buffer = np.append(self._buffer, new_data)
        print(f"Buffer after update: {self._buffer}")

    
    def compute_windows(self, type):
        if type == "std":
            windows = bn.move_std(self._buffer, self._window_size)
        else:
            windows = bn.move_median(self._buffer, self._window_size)
        
        return np.roll(windows, -self._neighbourhood_size)
    
    def fit_predict(self, new_data):
        # Update buffer with the latest batch
        self.update_buffer(new_data)
        median = self.compute_windows("median")
        print(f"Median after roll: {median}")
        std = self.compute_windows("std")

        dist_windows = np.absolute(median - self._buffer)
        scores = dist_windows / std
        return np.nan_to_num(scores)


if __name__ == "__main__":
    mm = MedianMethod(neighbourhood_size=1)
    data = np.array([3, 1, 8, 6, 5, 4, 1, 11, 0, 12])
    scores = mm.fit_predict(data)
    new_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    scores = mm.fit_predict(new_data)


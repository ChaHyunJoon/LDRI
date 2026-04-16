## Stable-Baseline3 concept memory buffer: ring buffer

import random
import numpy as np

class MemoryBuffer:
    def __init__(self, args):
        self.maxSize = int(args.buffer_size)
        self.batchSize = int(args.batch_size)
        # Default is without-replacement sampling (current behavior compatibility).
        # Can be switched globally or per random_sample() call.
        self.sample_with_replacement = bool(
            getattr(args, "buffer_sample_with_replacement", False)
        )

        # SB3-style ring buffer storage (preallocated numpy arrays).
        self._state = None
        self._action = None
        self._reward = None
        self._next_state = None
        self._pos = 0
        self.currentSize = 0
        self.counter = 0

    @staticmethod
    def _to_array(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def _init_storage(self, s, a):
        self._state = np.empty((self.maxSize, *s.shape), dtype=np.float32)
        self._action = np.empty((self.maxSize, *a.shape), dtype=np.float32)
        self._reward = np.empty((self.maxSize,), dtype=np.float32)
        self._next_state = np.empty((self.maxSize, *s.shape), dtype=np.float32)

    def store(self, s, a, r, s_, done=None):
        s_arr = self._to_array(s)
        a_arr = self._to_array(a)
        s_next_arr = self._to_array(s_)
        if self._state is None:
            self._init_storage(s_arr, a_arr)

        self._state[self._pos] = s_arr
        self._action[self._pos] = a_arr
        self._reward[self._pos] = np.float32(r)
        self._next_state[self._pos] = s_next_arr

        self._pos = (self._pos + 1) % self.maxSize
        self.counter += 1
        self.currentSize = min(self.counter, self.maxSize)

    def _sample_indices(self, replace):
        if self.currentSize <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if replace:
            return np.random.randint(0, self.currentSize, size=self.batchSize)
        if self.batchSize > self.currentSize:
            # Keep parity with random.sample failure condition.
            raise ValueError("Sample larger than population or is negative")
        return np.fromiter(
            random.sample(range(self.currentSize), self.batchSize),
            dtype=np.int64,
            count=self.batchSize,
        )

    def random_sample(self, replace=None):
        use_replace = self.sample_with_replacement if replace is None else bool(replace)
        idx = self._sample_indices(use_replace)
        s = self._state[idx]
        a = self._action[idx]
        r = self._reward[idx]
        s_ = self._next_state[idx]
        return (s, a, r, s_)
    

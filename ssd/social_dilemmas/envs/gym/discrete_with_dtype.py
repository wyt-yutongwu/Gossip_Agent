from gymnasium.spaces import Discrete
import numpy as np


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype=np.uint8):
        super().__init__(n)             
        self.dtype = np.dtype(np.uint8)     

# class DiscreteWithDType(Discrete):
#     def __init__(self, n, dtype):
#         assert n >= 0
#         self.n = n
#         # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
#         super(Discrete, self).__init__((), dtype)

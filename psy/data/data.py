import os.path
import numpy as np
import base64

__all__ = ['data']


class Data(object):

    BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')

    def __init__(self):
        self._dt = {}

    def __getitem__(self, item):
        if item in self._dt:
            return self._dt[item]
        file_path = os.path.join(self.BASE_DIR, item)
        self._dt[item] = np.loadtxt(file_path)
        return self._dt[item]


data = Data()

import numpy as np

from .distance import euclidean_distance
from .preprocessing import standardize


class KNNBase:

	def __init__(self, k=3, distance='euclidean_distance'):
		self.k = k
		self.distance = distance

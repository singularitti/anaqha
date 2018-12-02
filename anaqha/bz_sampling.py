"""

"""

import numpy as np


def sample(q_weights, quantity):
    if not np.all(np.greater_equal(q_weights, 0)):
        raise ValueError('Weights should all be greater equal than 0!')

    scaled_q_weights = q_weights / np.sum(q_weights)
    return np.dot(quantity.sum(axis=2), scaled_q_weights)

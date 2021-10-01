import matplotlib.pyplot as plt 
import numpy as np

from typing import List, Tuple, Union

def fast_hist(array: List[Union[int, float]], 
              bins: int) -> Tuple[List[int], List[float]]:
    """
    Builds bins' labels and bins' value counts for given array
    :param array: array with numeric values
    :param bins:  number of bins in result distribution
    :return: Two lists: 
             first contains value counts of each bin,
             second contains list of bins' labels
    """
    max_el = max(array)
    min_el = min(array)
    delta = (max_el - min_el) / bins
    bins_names = np.arange(min_el, max_el, delta)
    count_in_bin = np.zeros(bins)
    for val in array:
        count_in_bin[min(int((val - min_el) / delta), bins - 1)] += 1
    return (np.array([int (x) for x in count_in_bin]), bins_names)
    

def to_graph(array: List[Union[int, float]]):
    value_counts, bins_names = fast_hist(array, len(set(array)))
    plt.bar(bins_names, value_counts, width = 0.7)

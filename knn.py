"""
The knn module provides a class and two functions.

The class is a representation of a point in 7-dimensions space.
Concerning the functions, the first one is an implementation of the Minkowski distance calculus between two points
of the class presented above. The second one is a k-Nearest Neighbors algorithm.
"""

from typing import Callable, Tuple


class DimensionError(Exception):
    pass


class Point(object):
    """
    Represents a point in 7D space.
    """
    def __init__(self, point_id: int, coordinates: list[float], label: int) -> None:
        if len(coordinates) != 7: raise DimensionError
        else:
            self.id: int = point_id
            self.coordinates: list[float] = coordinates
            self.label: int = label

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id}, {self.coordinates}, {self.label})'

    def __str__(self, extend: bool = False) -> str:
        return f'Id : {self.id}, Label : {self.label}'


def minkowski(p1: Point, p2: Point, p:float) -> float:
    """
    Calculate the Minkowski distance between two points.
    With ``p`` = 2, it returns the Euclidean distance.
    With ``p`` = 1, it returns the Manhattan distance.

    :param p1: First point
    :param p2: Second point
    :param p: Parameter of the Minkowski distance
    :return: The p-Minkowski distance
    """
    result: float = 0.0
    for i in range(len(p1.coordinates)):
        result += abs(p1.coordinates[i] - p2.coordinates[i]) ** p
    return result ** (1/p)


def knn(k: int, new: Point, dataset: list[Point], fct_distance: Callable[[Point, Point], float]) -> Tuple[int, int]:
    """
    Applies the K-Nearest Neighbor algorithm to ``new`` using the given dataset.

    :param k: Number of neighbors
    :param new: Coordinates of the new point
    :param dataset: The dataset used
    :param fct_distance: The function to calculate the distance
    :return: The id of the new point and the class predicted
    """
    distances: list[Tuple[float, Point]] = []
    for point in dataset:
        distances.append((fct_distance(point, new), point))
    distances.sort(key=lambda x: x[0])
    label_occurrences: dict[int, int] = {}
    for i in range(k):
        label_occurrences[distances[i][1].label] = label_occurrences.get(distances[i][1].label, 0) + 1
    return new.id, max(label_occurrences, key=label_occurrences.get)
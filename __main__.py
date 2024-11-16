from knn import Point
from functools import partial
import knn, random, csv
from typing import Tuple, Callable


def import_csv(name: str) -> list[Point]:
    """
    Convert the database in a list of point.

    :param name: Database name
    :return: The list of points in the database
    """
    result: list[Point] = []
    with open(name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            result.append(Point(int(line[0]), list(map(float, line[1:-1])), int(line[-1])))
    return result

# NOT FINISHED
def create_dataset(dataset: list[Point]) -> list[list[Point]]:
    """
    Separate the dataset into training and test points.
    It chooses randomly 80% of each class for the train dataset and uses the other 20% for test dataset.

    :param dataset: Initial dataset to cut
    :return: A list of training and test points
    """
    result: list[list[Point]] = []
    return result


def fitness(dataset: list[Point], test_set: list[Point], k: int, distance: Callable[[Point, Point], float]) -> float:
    """
    Calculate the percentage of good predictions for ``k`` and ``distance``.

    :param dataset: Train dataset
    :param test_set: Test dataset
    :param k: Number of neighbors to use
    :param distance: Function to calculate distance
    :return: A number between 0 and 1
    """
    good_prediction: int = 0
    for point in test_set:
        if knn.knn(k, point, dataset, distance)[0] == point.label:
            good_prediction += 1
    return good_prediction / len(test_set)


def export_result_csv(result: list[Tuple[int, int]], file_name: str) -> None:
    """
    Export the results to a CSV file.

    :param result: The result to export
    :param file_name: The name of the file where the results will be saved
    """
    with open(file_name, 'w', newline='') as csvfile:
        csvfile = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        csvfile.writerow(['Id', 'Label'])
        csvfile.writerows(result)


if __name__ == '__main__':
    pass
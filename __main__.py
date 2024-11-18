from knn import Point
from functools import partial
import knn, random, csv, math, platform, datetime, time
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


def create_dataset(dataset: list[Point]) -> list[list[Point]]:
    """
    Separate the dataset into training and test points.
    It chooses randomly 80% of each class for the train dataset and uses the other 20% for test dataset.

    :param dataset: Initial dataset to cut
    :return: A list of training and test points
    """
    result: list[list[Point]] = [[], []]
    temporary_list: list[list[Point]] = [[], [], [], []]
    for point in dataset:
        temporary_list[point.label].append(point)
    for l in temporary_list:
        r_list: list[Point] = random.sample(l, round(0.8*len(l)))
        result[0].extend(r_list)
        result[1].extend([e for e in l if e not in r_list])
    return result


def fitness(dataset: list[Point], testing_set: list[Point], k_knn: int, d: Callable[[Point, Point], float]) -> float:
    """
    Calculate the percentage of good predictions for ``k`` and ``distance``.

    :param dataset: Train dataset
    :param testing_set: Test dataset
    :param k_knn: Number of neighbors to use
    :param d: Function to calculate distance
    :return: A number between 0 and 1
    """
    good_prediction: int = 0
    for point in testing_set:
        if knn.knn(k_knn, point, dataset, d)[1] == point.label:
            good_prediction += 1
    return good_prediction / len(testing_set)


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


def test_knn() -> None:
    """
    Create a file with details of each running test and a file with the best combo (k, distance) of a test
    """
    datas: list[Point] = import_csv("data\\train.csv")
    (train_set, test_set) = create_dataset(datas)
    euclidean: partial[float] = partial(knn.minkowski, p=2)
    manhattan: partial[float] = partial(knn.minkowski, p=1)
    supposed_optimal_k: int = round(math.sqrt(len(datas)))
    best_combo: Tuple[int, str, float] = (0, "", 0.0)
    for distance in [euclidean, manhattan]:
        for k in range(1, round(1.1*supposed_optimal_k)):
            start_time: float = time.time()
            score: float = fitness(train_set, test_set, k, distance)
            end_time: float = time.time()
            if score > best_combo[2]:
                best_combo = (k, str(distance), score)
            with open('extended_result.txt', 'a', newline='') as result_file:
                result_file.write(f'Test from {datetime.datetime.now()} :\n'
                                  f' - CPU Infos : {platform.processor()}\n'
                                  f' - Running time : {end_time - start_time} seconds\n'
                                  f' - Train dataset size : {len(train_set)}\n'
                                  f' - Test dataset size : {len(test_set)}\n'
                                  f' - k : {k}\n'
                                  f' - Distance : {str(distance)}\n'
                                  f' - Score : {score}\n\n')
    with open('result.txt', 'a', newline='') as result_file:
        result_file.write(f'k = {best_combo[0]}, p = {best_combo[1][-2]}, score = {best_combo[2]}\n')


if __name__ == '__main__':
    for i in range(10):
        test_knn()
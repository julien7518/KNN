from knn import Point
from functools import partial
import knn, random, csv, math, platform, datetime, time, os, sys
from typing import Tuple, Callable


def import_csv(name: str, full_data: bool = True) -> list[Point]:
    """
    Convert the database in a list of point.

    :param name: Database name
    :param full_data: Specify if we know and need to import the class of the point
    :return: The list of points in the database
    """
    result: list[Point] = []
    with open(name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            if full_data:
                result.append(Point(int(line[0]), list(map(float, line[1:-1])), int(line[-1])))
            else:
                result.append(Point(int(line[0]), list(map(float, line[1:]))))
    return result


def get_max_min(datas: list[Point], column: int) -> Tuple[float, float]:
    """
    Find the maximum value and minimum value in the column ``i``.

    :param datas: The original dataset
    :param column: The number of the column
    :return: The minimum value and the maximum value
    """
    maxi: float = sys.float_info.min
    mini: float = sys.float_info.max
    for point in datas:
        if point.coordinates[column] > maxi:
            maxi = point.coordinates[column]
        if point.coordinates[column] < mini:
            mini = point.coordinates[column]
    return mini, maxi


def min_max_scaler(dataset: list[Point]) -> None:
    """
    Normalize the dataset with the minmax formula.
    For each value `x` in dimension, it transforms it in `x* = (x - min(x)) / (max(x) - min(x))`

    :param dataset: The original dataset
    """
    min_max: list[Tuple[float, float]] = []
    for i in range(len(dataset[0].coordinates)):
        min_max.append(get_max_min(dataset, i))
    for points in dataset:
        for i in range(len(min_max)):
            points.coordinates[i] = (points.coordinates[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])


def get_mean(dataset: list[Point]) -> list[float]:
    """
    Calculate the mean of the dataset.

    :param dataset: The original dataset
    :return: A list which the mean of each column of the dataset
    """
    result: list[float] = [0 for i in range(len(dataset[0].coordinates))]
    for point in dataset:
        for i in range(len(dataset[0].coordinates)):
            result[i] += point.coordinates[i]
    for i in range(len(dataset[0].coordinates)):
        result[i] /= len(dataset)
    return result


def get_standard_deviation(dataset: list[Point]) -> list[float]:
    result: list[float] = [0 for i in range(len(dataset[0].coordinates))]
    means: list[float] = get_mean(dataset)
    for point in dataset:
        for i in range(len(dataset[0].coordinates)):
            result[i] += (point.coordinates[i] - means[i]) **2
    for i in range(len(dataset[0].coordinates)):
        result[i] /= len(dataset)
        result[i] = math.sqrt(result[i])
    return result



def std_mean_normalization(dataset: list[Point]) -> None:
    stds: list[float] = get_standard_deviation(dataset)
    means: list[float] = get_mean(dataset)
    for point in dataset:
        for i in range(len(dataset[0].coordinates)):
            point.coordinates[i] = (point.coordinates[i] - means[i]) / stds[i]


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


def get_id(path: str) -> int:
    """
    Return the next id to use in the file name.

    :param path: The path of the folder
    :return: The next id to use to not replace an existent file
    """
    return len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])


def test_knn(normalize_function: Callable[[list[Point]], None] | None = None,extended: bool = False) -> None:
    """
    Create a file with details of each running test and a file with the best combo (k, distance) of a test.

    :param extended: Specify if it's needed to create an extended result file
    :param normalize_function: Function to normalize the dataset
    """
    datas: list[Point] = import_csv("data\\train.csv")
    if normalize_function is not None:
        normalize_function(datas)
    (train_set, test_set) = create_dataset(datas)
    euclidean: partial[float] = partial(knn.minkowski, p=2)
    manhattan: partial[float] = partial(knn.minkowski, p=1)
    supposed_optimal_k: int = round(math.sqrt(len(datas)))
    best_combo: Tuple[int, str, float] = (0, "", 0.0)
    file_id: int = get_id("C:\\Users\\julie\\OneDrive - De Vinci\\A3 - ESILV\\Datascience-IA\\KNN Competition\\result\\extended")
    for distance in [manhattan, euclidean]:
        for k in range(1, 5):
            start_time: float = time.time()
            score: float = fitness(train_set, test_set, k, distance)
            end_time: float = time.time()
            if score > best_combo[2]:
                best_combo = (k, str(distance), score)
            if extended:
                with open(f'result\\extended\\extended_result{file_id}.txt', 'a', newline='') as result_file:
                    result_file.write(f'Test from {datetime.datetime.now()} :\n'
                                      f' - CPU Infos : {platform.processor()}\n'
                                      f' - Running time : {end_time - start_time} seconds\n'
                                      f' - Train dataset size : {len(train_set)}\n'
                                      f' - Test dataset size : {len(test_set)}\n'
                                      f' - k : {k}\n'
                                      f' - Distance : {str(distance)}\n'
                                      f' - Score : {score}\n\n')
    with open(f'result\\result.txt', 'a', newline='') as result_file:
        result_file.write(f'k = {best_combo[0]}, p = {best_combo[1][-2]}, score = {best_combo[2]}, normalisation = {normalize_function.__name__}\n')


if __name__ == '__main__':
    for boucle in range(10):
        test_knn(min_max_scaler)
    with open(f'result\\result.txt', 'a') as f: f.write("\n")
    # data: list[Point] = import_csv("data\\train.csv")
    # std_mean_normalization(data)
    # test: list[Point] = import_csv("data\\test.csv", False)
    # r: list[Tuple[int, int]] = []
    # for p in test:
    #     r.append(knn.knn(1, p, data, partial(knn.minkowski, p=1)))
    # export_result_csv(r, 'result\\result.csv')
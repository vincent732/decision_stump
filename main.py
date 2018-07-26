
import numpy as np
import math
from tqdm import tqdm


class Hypothesis:

    def __init__(self, s, theta):
        self.s = s
        self.theta = theta

def generate_sample(number_of_sample):
    x = np.random.uniform(-1, 1, number_of_sample)
    y = np.array([np.sign(row) for row in list(x)])
    return zip(x, y)


def add_noise(data, noise_level):
    x, y = zip(*data)
    def fn(i):
        return -1 * i if np.random.random_sample() < noise_level else i
    y = map(fn, y)
    return zip(x, y)


def error_ratio(hypothesis, data):
    err = 0
    for x, y in data:
        yhat = hypothesis.s * np.sign(x - hypothesis.theta)
        if yhat != y:
            err += 1
    return (1.0 * err)/len(data)


def ein(data):
    x, y = zip(*data)
    sample_size = len(data)
    min_err = 1.0
    s_candidates = [1, -1]
    best_hypothesis = Hypothesis(-1, -1)
    for i in range(sample_size + 1):#n points generates n+1 intervals
        for s in s_candidates:
            if i == 0:
                theta = (x[i] - 1) / 2.0
            elif i == sample_size:
                theta = (x[i - 1] + 1) / 2.0
            else:
                theta = (x[i] + x[i - 1]) / 2.0
            err = error_ratio(Hypothesis(s, theta), data)
            if err < min_err:
                min_err = err
                best_hypothesis.s = s
                best_hypothesis.theta = theta
    return min_err, best_hypothesis


def ein_with_multiple_d(data):
    x, y = zip(*data)
    num_of_dimensions = len(x[0])
    min_err = 1.0
    best_hypothesis = Hypothesis(-1, -1)
    best_dimension = -1
    for j in range(num_of_dimensions):
        single_x = [row[j] for row in x]
        err, hypothesis = ein(zip(single_x, y))
        if err < min_err:
            min_err = err
            best_hypothesis.s = hypothesis.s
            best_hypothesis.theta = hypothesis.theta
            best_dimension = j+1
    return min_err, best_hypothesis, best_dimension


def eout(hypothesis):
    return 0.5 + 0.3 * hypothesis.s * (math.fabs(hypothesis.theta) - 1.0)


def decision_stump_with_1d():
    epochs = 5000
    total_ein = 0.0
    total_eout = 0.0
    for i in tqdm(range(epochs)):
        sample_data = add_noise(generate_sample(20), 0.2)
        err, hypothesis = ein(sample_data)
        total_ein += err
        total_eout += eout(hypothesis)

    print("Ein : %.3f" % (total_ein/epochs))
    print("Eout : %.3f" % (total_eout/epochs))


def getData(file_path):
    raw_x, raw_y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().replace('\n', '').split(' ')
            raw_x.append(map(float, row[:-1]))
            raw_y.append(float(row[-1]))
    raw = zip(np.array(raw_x), np.array(raw_y))
    return raw


def decision_stump_with_multiple_d():
    train = getData('./hw2_train.dat')
    err, hypothesis, dimension = ein_with_multiple_d(train)
    test = getData('./hw2_test.dat')
    print(dimension)
    test = [(x[dimension-1], y) for x, y in test]
    print("Ein : %.3f" % (err))
    print("Eout : %.3f" % (error_ratio(hypothesis, test)))

if __name__ == "__main__":
    decision_stump_with_multiple_d()
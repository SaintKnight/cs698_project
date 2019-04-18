#
# helpful utility functions for fly lsh
#

import time

import keras

def benchmark(func):
    ''' Standard benchmarking decorator. '''

    from time import clock

    def wrapper(*args, **kwargs):
        t = clock()
        res = func(*args, **kwargs)
        print("function: ", func.__name__, "benchmark: ", clock()-t, "s")
        return res

    return wrapper

# wants is an array of nums that wanna grab
# e.g. wants = [1,3,4], then returned result will contain 1 3 4 only
def findSpecificTrainInput(data, wants, maxIndex = 50000):
    result = [[],[]]
    total = min(len(data[1]), maxIndex)
    for i in range(total):
        for wanted in wants:
            if data[1][i][wanted] > 0:
                result[0].append(data[0][i])
                result[1].append(data[1][i])
                break
    print("total processed: ", len(result[0]))
    return result

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

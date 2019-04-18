#
# helpful utility functions for fly lsh
#

def benchmark(func):
    ''' Standard benchmarking decorator. '''

    from time import clock

    def wrapper(*args, **kwargs):
        t = clock()
        res = func(*args, **kwargs)
        print("function: ", func.__name__, "benchmark: ", clock()-t, "s")
        return res

    return wrapper

import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        init = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_ = end - init
        print(f"Execution time '{func.__name__}': {time_:.2f} segundos")
        return result

    return wrapper

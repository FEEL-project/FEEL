from functools import wraps
import time

def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f'Running {fn.__name__}')
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("====================================")
        print(f'{fn.__name__} took {end - start} seconds')
        return result
    return wrapper
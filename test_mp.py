import time
import multiprocessing as mp

def worker(num):
    """worker function"""
    # print('Worker: ', num)
    return num*num

if __name__ == '__main__':
    data = range(1000)
    pool_start = time.time()
    pool = mp.Pool()
    results = pool.map(worker, data)
    pool.close()
    pool.join()

    print('Pool: {} sec'.format(time.time() - pool_start))
    print(results)
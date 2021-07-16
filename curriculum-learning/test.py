import boxoban_level_collection as collection
import multiprocessing as mp
import time

def worker(i):
    if i == 1:
        time.sleep(5)
        print(collection.find(0)[3])
    else:
        print(collection.find(0)[3])
        (id, score, trajectory, room, topology) = collection.find(0)
        room[room == 0] = 11
        print(room)
        print(collection.find(0)[3])
    while True:
        pass


if __name__ == '__main__':
    for i in range(2):
        p = mp.Process(target=worker, args=(i,))
        p.daemon = True
        p.start()

    while True:
            pass

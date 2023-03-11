import os


def run():
    actors = ['person_0000', 'person_0004']

    for actor in actors:
        os.system(f'python tracker.py --cfg configs/actors/{actor}.yml')


if __name__ == '__main__':
    run()

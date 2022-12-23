from config import *


class User:

    def __init__(self, uid):
        self._uid = uid
        self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))


if __name__ == "__main__":
    user = User(0)

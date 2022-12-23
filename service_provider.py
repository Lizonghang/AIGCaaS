from config import *


class ServiceProvider:

    def __init__(self, sid):
        self._sid = sid
        self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))

    def _distance_to(self, user):
        return np.sqrt(np.square(self._loc - user._loc).sum())

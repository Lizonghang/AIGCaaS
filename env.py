import time
from user import User
from service_provider import ServiceProvider
from config import *


GLOBAL_CLOCK = time.time()


class AIGCEnv:

    def __init__(self, n_users, n_service_providers):
        self._n_users = n_users
        self._n_service_providers = n_service_providers
        self._users = []
        self._service_providers = []

        self.create_env()

    def create_env(self):
        self._users = [
            User(uid_) for uid_ in np.arange(self._n_users)]
        self._service_providers = [
            ServiceProvider(sid_) for sid_ in np.arange(self._n_service_providers)]


def make_env(*args, **kwargs):
    return AIGCEnv(*args, **kwargs)


if __name__ == "__main__":
    n_users, n_service_providers = 10, 3
    env = make_env(n_users, n_service_providers)

from user import User
from service_provider import ServiceProvider
from task import TaskGenerator
from config import *


class SwarmManager:

    def __init__(self):
        self._n_users = NUM_USERS
        self._n_service_providers = NUM_SERVICE_PROVIDERS
        self._users = [
            User(uid, 0) for uid in range(self._n_users)]
        self._service_providers = [
            ServiceProvider(sid, 0) for sid in range(self._n_service_providers)]
        self._task_generator = TaskGenerator()
        self._querying_user = None

    def check_finished(self, curr_time):
        num_finished = 0
        for service_provider in self._service_providers:
            num_finished += service_provider.check_finished(curr_time)
        return num_finished

    def next_user_task(self):
        self._querying_user = np.random.choice(self._users)
        task, terminate = next(self._task_generator)
        self._querying_user.add_task(task)

        curr_time = task.arrival_time
        self.check_finished(curr_time)
        return self._querying_user, curr_time, terminate

    def assign(self, sid, curr_time):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"

        service_provider = self._service_providers[sid]
        assert service_provider.id == sid, \
            f"Service provider (id={service_provider.id}) not match assigned sid={sid}"

        reward = service_provider.assign_task(self._querying_user.task, curr_time)
        return reward

    @property
    def vector(self):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"

        vec = [service_provider.vector for service_provider in self._service_providers]
        vec += [self._querying_user.vector]
        vec = np.hstack(vec)
        return vec

    def reset(self):
        [obj_.reset() for obj_ in self._users]
        [obj_.reset() for obj_ in self._service_providers]
        self._task_generator.reset()
        self._querying_user = None


if __name__ == "__main__":
    manager = SwarmManager()

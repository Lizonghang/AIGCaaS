import os
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
        self.next_user_task()

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
        return curr_time, terminate

    def assign(self, sid, curr_time):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"

        service_provider = self._service_providers[sid]
        assert service_provider.id == sid, \
            f"Service provider (id={service_provider.id}) not match assigned sid={sid}"
        assert self._querying_user.task.arrival_time == curr_time, \
            f"Arrival time mismatch: {self._querying_user.task.arrival_time} != {curr_time}"

        reward = service_provider.assign_task(self._querying_user.task, curr_time)

        # uneven load penalty
        penaly = self.load_imbalance()

        return reward - penaly

    def load_imbalance(self):
        t_util = [service_provider.norm_available_t
                  for service_provider in self._service_providers]
        return np.abs(t_util - np.mean(t_util)).std()

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
        self.next_user_task()
        return self.vector

    @property
    def total_t_available(self):
        return sum([service_provider.total_t
                    for service_provider in self._service_providers])

    @property
    def total_t_serving(self):
        return sum([service_provider.used_t
                    for service_provider in self._service_providers])

    def monitor(self):
        os.system("cls")
        WIDTH = 98
        print()

        # service provider info
        HEAD = " SID | TASK SERVING | " \
               "\033[0;32mTASK FINISHED\033[0m | " \
               "\033[0;31mTASK CRASHED\033[0m | " \
               "TOTAL T | USED T | AVAILABLE T | " \
               "\033[0;31mNUM CRASHED\033[0m "

        print("-" * WIDTH)
        print(f"\033[7mSERVICE PROVIDER\033[0m "
              f"(TOTAL T AVAILABLE {self.total_t_available})".center(WIDTH))
        print("-" * WIDTH)
        print(HEAD)
        print("-" * WIDTH)

        for service_provider in self._service_providers:
            info = service_provider.info
            print(f"{str(info['id']).center(6)}"
                  f"{str(info['task_serving']).center(15)}"
                  f"\033[0;32m{str(info['task_finished']).center(16)}\033[0m"
                  f"\033[0;31m{str(info['task_crashed']).center(15)}\033[0m"
                  f"{str(info['total_t']).center(10)}"
                  f"{str(info['used_t']).center(9)}"
                  f"{str(info['available_t']).center(14)}"
                  f"\033[0;31m{str(info['num_crashed']).center(14)}\033[0m")

        print("-" * WIDTH)

        # task info
        total_tasks = 0
        total_serving = 0
        total_crashed = 0
        total_finished = 0
        for service_provider in self._service_providers:
            info = service_provider.task_summary()
            total_serving += info['serving']
            total_crashed += info['crashed']
            total_finished += info['finished']
            total_tasks += info['total']
        assert total_tasks == total_serving + total_crashed + total_finished

        print(f"\033[7mTASK\033[0m".center(WIDTH // 5), end='')
        print(f"TOTAL: {str(total_tasks)}".center(WIDTH // 5), end='')
        print(f"SERVING: {str(total_serving)}".center(WIDTH // 5), end='')
        print(f"\033[0;31mCRASHED: {str(total_crashed)}\033[0m".center(WIDTH // 5), end='')
        print("    ", end='')
        print(f"\033[0;32mFINISHED: {str(total_finished)}\033[0m".center(WIDTH // 5))
        print("-" * WIDTH)

        # user info
        print(f"\033[7mUSER\033[0m".center(WIDTH // 5), end='')
        print(f"TOTAL USERS: {str(self._n_users)}".center(WIDTH // 3), end='')
        print(f"TOTAL SERVING T: {str(self.total_t_serving)}".center(WIDTH // 3))
        print("-" * WIDTH)


if __name__ == "__main__":
    manager = SwarmManager()

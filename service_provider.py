from task import TaskType
from config import *


class ServiceProvider:

    def __init__(self, sid, task_type_id):
        self._sid = sid
        self._task_type = TaskType(task_type_id)
        self._reward_coefs = (
            np.random.choice(AX_RANGE),
            np.random.choice(AY_RANGE),
            np.random.choice(BX_RANGE),
            np.random.choice(BY_RANGE))
        self._serving_tasks = []
        self._terminated_tasks = {'crashed': [], 'finished': []}
        self._total_t = np.random.choice(TOTAL_T_RANGE)

        # The following info are not currently considered
        self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))
        self._num_cpu = NUM_CPUS
        self._num_gpu = NUM_GPUS
        self._cpu_mem = CPU_MEM
        self._gpu_mem = GPU_MEM

    def _distance_to(self, user):
        return np.sqrt(np.square(self._loc - user._loc).sum())

    @property
    def used_t(self):
        return sum([task.t for task in self._serving_tasks])

    @property
    def available_t(self):
        return self._total_t - self.used_t

    @property
    def norm_total_t(self):
        max_t = TOTAL_T_RANGE[-1]
        return self._total_t / max_t

    @property
    def norm_available_t(self):
        return self.available_t / self._total_t

    def expected_reward(self, t):
        return REWARD(*self._reward_coefs, t)

    def reset(self):
        self._serving_tasks.clear()
        self._terminated_tasks['crashed'].clear()
        self._terminated_tasks['finished'].clear()

    @property
    def vector(self):
        # (total_t, available_t)
        return np.hstack([self.norm_total_t, self.norm_available_t])


if __name__ == "__main__":
    service_provider = ServiceProvider(0, 0)
    print(service_provider.vector)

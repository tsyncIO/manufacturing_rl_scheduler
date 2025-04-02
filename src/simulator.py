# manufacturing_rl/simulator.py
import numpy as np
from typing import List, Tuple

class ManufacturingSimulator:
    def __init__(self, num_machines: int, num_jobs: int, max_steps: int = 100) -> None:
        self.num_machines = num_machines
        self.num_jobs = num_jobs
        self.max_steps = max_steps
        self.setup_times = self.generate_setup_times()
        self.processing_times = self.generate_processing_times()
        self.job_quantities = self.generate_job_quantities()
        self.job_deadlines = self.generate_job_deadlines()
        self.current_step = 0
        self.previous_machines: List[int] = [-1] * num_jobs
        self.machine_available_time = np.zeros(num_machines)
        self.job_completion_times = np.zeros(num_jobs)
        self.job_machine_assignments: List[int] = [-1] * num_jobs
        self.job_priorities = self.generate_job_priorities()

    def generate_setup_times(self) -> np.ndarray:
        return np.random.randint(1, 5, size=(self.num_machines, self.num_machines))

    def generate_processing_times(self) -> np.ndarray:
        return np.random.randint(5, 15, size=(self.num_machines, self.num_jobs))

    def generate_job_quantities(self) -> np.ndarray:
        return np.random.randint(10, 30, size=self.num_jobs)

    def generate_job_deadlines(self) -> np.ndarray:
        return np.random.randint(50, 150, size=self.num_jobs)

    def generate_job_priorities(self) -> np.ndarray:
        return np.random.randint(1, 4, size=self.num_jobs)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.previous_machines = [-1] * self.num_jobs
        self.machine_available_time = np.zeros(self.num_machines)
        self.job_completion_times = np.zeros(self.num_jobs)
        self.job_machine_assignments = [-1] * self.num_jobs
        self.job_quantities = self.generate_job_quantities()
        self.job_deadlines = self.generate_job_deadlines()
        self.job_priorities = self.generate_job_priorities()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.concatenate([
            self.job_quantities / np.max(self.job_quantities) if np.max(self.job_quantities) > 0 else self.job_quantities,
            self.machine_available_time / self.max_steps,
            self.job_deadlines / self.max_steps,
            self.job_priorities / np.max(self.job_priorities) if np.max(self.job_priorities) > 0 else self.job_priorities,
            np.array(self.job_machine_assignments) / self.num_machines if self.num_machines > 0 else np.array(self.job_machine_assignments),
        ])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.current_step += 1
        reward = 0
        done = False

        job_to_schedule = action
        available_machines = np.where(self.machine_available_time <= self.current_step)[0]
        
        if not available_machines.size:
            return self.get_state(), -10, done

        best_machine = -1
        min_finish_time = np.inf
        for machine in available_machines:
            setup_time = self.setup_times[self.previous_machines[job_to_schedule], machine] if self.previous_machines[job_to_schedule] != -1 else 0
            finish_time = self.current_step + setup_time + self.processing_times[machine, job_to_schedule]
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_machine = machine

        if best_machine == -1:
            return self.get_state(), -5, done

        setup_time = self.setup_times[self.previous_machines[job_to_schedule], best_machine] if self.previous_machines[job_to_schedule] != -1 else 0
        start_time = max(self.current_step, self.machine_available_time[best_machine])
        finish_time = start_time + setup_time + self.processing_times[best_machine, job_to_schedule]
        
        self.machine_available_time[best_machine] = finish_time
        self.job_completion_times[job_to_schedule] = finish_time
        self.job_machine_assignments[job_to_schedule] = best_machine
        self.previous_machines[job_to_schedule] = best_machine
        self.job_quantities[job_to_schedule] -= 1

        reward = 1
        tardiness = max(0, finish_time - self.job_deadlines[job_to_schedule])
        reward -= 2 * tardiness
        reward += self.job_priorities[job_to_schedule]

        if np.all(self.job_quantities <= 0) or self.current_step >= self.max_steps:
            done = True
            if np.all(self.job_quantities <= 0):
                reward += 50

        return self.get_state(), reward, done
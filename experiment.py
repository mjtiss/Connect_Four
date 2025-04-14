from mcts import MCTS
from ConnectsState import ConnectState
import time
import math

print("🚀 Starting experiment...")

def run_experiment(max_iterations_list, time_limit_list, exploration_constants, num_trials=1):
    results = []

    for c in exploration_constants:
        for max_iter in max_iterations_list:
            for t_limit in time_limit_list:
                total_time = 0
                total_rollouts = 0

                print(f"Running config: c={c}, max_iter={max_iter}, time_limit={t_limit}s")

                for _ in range(num_trials):
                    state = ConnectState()
                    mcts = MCTS(state, exploration_constant=c)
                    mcts.search(max_iterations=max_iter, time_limit=t_limit)

                    total_time += mcts.run_time
                    total_rollouts += mcts.num_rollouts

                avg_time = total_time / num_trials
                avg_rollouts = total_rollouts / num_trials

                results.append({
                    "c": c,
                    "max_iter": max_iter,
                    "time_limit": t_limit,
                    "avg_time": round(avg_time, 3),
                    "avg_rollouts": int(avg_rollouts)
                })

    print("\n===== Summary =====")
    for res in results:
        print(f"c={res['c']:.2f}, iter={res['max_iter']}, time={res['time_limit']}s → "
              f"avg_time={res['avg_time']}s, avg_rollouts={res['avg_rollouts']}")

if __name__ == "__main__":
    run_experiment(
        max_iterations_list=[10000, 50000, 100000],
        time_limit_list=[0.5, 1.0, 2.0],
        exploration_constants=[0.5, 1.0, math.sqrt(2), 2.0],
        num_trials=3
    )

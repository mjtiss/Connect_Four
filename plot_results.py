import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV into a DataFrame
df = pd.read_csv("experiment_results.csv")

# Group by each policy and average the results across trials
grouped = df.groupby(["policy_id", "c_value", "max_iterations", "time_limit"]).agg({
    "num_rollouts": "mean",
    "time_taken": "mean"
}).reset_index()

# Plot: Number of rollouts vs c value, colored by max_iterations
plt.figure(figsize=(10, 6))
for max_iter in grouped["max_iterations"].unique():
    subset = grouped[grouped["max_iterations"] == max_iter]
    plt.plot(subset["c_value"], subset["num_rollouts"], marker='o', label=f"{max_iter} iterations")

plt.title("MCTS Rollouts vs Exploration Constant (c)")
plt.xlabel("Exploration Constant (c)")
plt.ylabel("Average Number of Rollouts")
plt.legend(title="Max Iterations")
plt.grid(True)
plt.tight_layout()
plt.savefig("mcts_policy_performance.png")
plt.show()


# Filter to only include policies within time budget
df_valid = df[df["time_taken"] <= 2.0]

# Group by policy and average the rollouts
avg_results = df_valid.groupby(["policy_id", "c_value", "max_iterations", "time_limit"]).agg({
    "num_rollouts": "mean",
    "time_taken": "mean"
}).reset_index()

# Sort by most rollouts
best = avg_results.sort_values("num_rollouts", ascending=False).head(1)

print("\n🏆 Best policy within time limit:")
print(best.to_string(index=False))

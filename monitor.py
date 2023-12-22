import matplotlib.pyplot as plt
import pandas as pd

env_id = 'LunarLander-v2'
showCSV = False

if(showCSV):
    csvFile = F"log_dir_{env_id}/0.monitor.csv"
    df = pd.read_csv(csvFile, skiprows=2, header=None)
    df.columns = ["r", "l", "t"]

    rewards = df["r"].to_numpy()

else:
    rewards = []

    txt_file = F"log_dir_{env_id}/output_log_{env_id}.txt"
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "reward" in line:
                rewards.append(round(float(line.split(":")[1].lstrip()), 2))

plt.figure(figsize=(12, 6))
plt.grid(True)
plt.title(F"Training Rewards")
plt.plot(rewards)
plt.show()

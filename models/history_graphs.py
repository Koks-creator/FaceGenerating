import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("training_log_wgan_v4.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(df["critic_loss"])
axes[0].set_title("Critic Loss")
axes[0].set(xlabel="Epoch [-]", ylabel="Loss [-]")

axes[1].plot(df["gen_loss"])
axes[1].set_title("Generator Loss")
axes[1].set(xlabel="Epoch [-]", ylabel="Loss [-]")

plt.tight_layout()
plt.show()
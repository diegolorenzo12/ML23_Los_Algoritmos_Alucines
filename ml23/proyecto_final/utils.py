import matplotlib.pyplot as plt

def plot_rewards(episode_rewards, lr, actions):
    plt.scatter(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
    plt.title('Rendimiento')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')    

    plt.annotate(f'Learning Rate: {lr}', xy=(0.5, -0.2), xycoords="axes fraction",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.annotate(f'Number of Actions: {actions}', xy=(0.5, -0.3), xycoords="axes fraction",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.tight_layout()
    plt.show()

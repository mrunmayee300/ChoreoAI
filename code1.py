import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load motion capture data
file_path = "mariel_chunli.npy"
data = np.load(file_path)  # Shape: (t, num_joints, 3)

# Ensure data is correctly loaded
print(f"Data shape: {data.shape}")

def update(num, data, scatter, lines, connections):
    """Update function for animation."""
    scatter.set_offsets(data[num, :, :2])  # Update X, Y
    scatter.set_3d_properties(data[num, :, 2], 'z')  # Update Z
    for line, (start, end) in zip(lines, connections):
        line.set_data([data[num, start, 0], data[num, end, 0]], 
                      [data[num, start, 1], data[num, end, 1]])
        line.set_3d_properties([data[num, start, 2], data[num, end, 2]])
    return scatter, *lines

# Define sample connections between joints (update based on data structure)
connections = [(0, 1), (1, 2), (2, 3), (3, 4),
               (1, 5), (5, 6), (6, 7), (7, 8),
               (1, 9), (9, 10), (10, 11), (11, 12)]

def plot_3d_motion(data):
    """Plots an animated 3D visualization of motion capture data."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2], c='r', marker='o')
    lines = [ax.plot([], [], [], 'b')[0] for _ in connections]

    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], 
                                  fargs=(data, scatter, lines, connections), interval=50)
    plt.show()

plot_3d_motion(data)

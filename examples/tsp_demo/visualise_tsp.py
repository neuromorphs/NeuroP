import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from functools import partial
import numpy as np
import random

from helpers import *

# fig, ax = plt.subplots(figsize=(8, 10))
# fig2, ax2 = plt.subplots(figsize=(8, 10))


def draw_solution_tsp(
    frame, ax, ax2, cityNames, cityCoordinates, N, sorted_samples_by_energy
):
    ax.clear()
    ax2.clear()
    # N = len(sch);
    m, sch = parse_op_vec_tsp(sorted_samples_by_energy[frame], N)
    # print(m)
    # Draw the graph
    # ax.figure(figsize=(8, 6))

    # Annotate the names of the cities
    for i, (x, y) in enumerate(cityCoordinates):
        ax.annotate(cityNames[i], (y, x), size="10")

    # Draw the nodes and vertices
    for i in range(N):
        for j in range(i + 1, N):
            x = [cityCoordinates[i][1], cityCoordinates[j][1]]
            y = [cityCoordinates[i][0], cityCoordinates[j][0]]
            ax.plot(x, y, "b--", alpha=0.1)
    ax.scatter(
        [cityCoordinates[i][1] for i in range(N)],
        [cityCoordinates[i][0] for i in range(N)],
        marker="o",
        color="green",
        s=200,
    )

    # Draw the solution
    path_str = ""
    for p in range(N - 1):
        i = sch[p]
        j = sch[p + 1]
        x = [cityCoordinates[i][1], cityCoordinates[j][1]]
        y = [cityCoordinates[i][0], cityCoordinates[j][0]]
        ax.plot(x, y, "r", alpha=0.5)
        ax.annotate(str(p + 1), ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), size="10")
        path_str = path_str + cityNames[sch[p]] + " -> "

    # Draw the implied path from the last city in the solution to the first
    i = sch[0]
    j = sch[N - 1]
    x = [cityCoordinates[i][1], cityCoordinates[j][1]]
    y = [cityCoordinates[i][0], cityCoordinates[j][0]]
    ax.plot(x, y, "r", alpha=0.5)
    ax.annotate(str(N), ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), size="10")
    # Redraw first node clearly in a different colour
    ax.scatter(
        cityCoordinates[i][1], cityCoordinates[i][0], marker="o", color="blue", s=250
    )
    path_str = path_str + cityNames[j] + " -> " + cityNames[i]
    # print("The solution path is: ")
    # print(path_str)

    # Plot the boltzman animation

    # Generate angles evenly spaced around a circle
    angles = np.linspace(0, 2 * np.pi, N * N, endpoint=False)
    # Create empty arrays to store the x and y coordinates
    x_coords = []
    y_coords = []
    x = 1 * np.cos(angles)
    y = 1 * np.sin(angles)
    x_coords.extend(x)
    y_coords.extend(y)
    # Create scatter plot
    ax2.scatter(x_coords, y_coords, s=200, c=np.ravel(m))
    # for i in range(N*N):
    #     for j in range(N*N):
    #         opacity = np.abs(Q[i, j])/np.max(np.abs(Q))
    # ax2.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'k-', alpha=opacity)

    # Create the title
    ax.set_title("Traveling Salesman Problem. Solution Rank: " + str(frame), size="20")
    ax.grid(False)
    ax.axis("off")

    # Create the title
    ax2.set_title("Traveling Salesman Problem. Solution Rank: " + str(frame), size="20")
    ax2.grid(False)
    ax2.axis("off")


# Code for animation

# # Create the animation

# animation = FuncAnimation(fig, partial(draw_solution_tsp, cityNames=cityNames, cityCoordinates=cityCoordinates,sch=sch,N=N), frames=100, interval=100, repeat=True)
# animation2 = FuncAnimation(fig2, partial(draw_solution_tsp, cityNames=cityNames, cityCoordinates=cityCoordinates,sch=sch,N=N), frames=100, interval=100, repeat=True)

# #Show the plot
# plt.show()

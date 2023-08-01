# from examples.tsp_demo.helpers import *
from helpers import *
from visualise_tsp import *
import dimod
import neal

def get_solution_tsp(cityNames, distanceMatrix, start_city, enforce_start_city):
    N = int(len(cityNames))  # numNodes

    # Define symbols for the matrix elements
    elements = symbols("x:%d" % (N * N))
    # print(elements)

    # Create the symbolic matrix
    X = Matrix(N, N, elements)

    obj = get_obective_fct(cityNames, distanceMatrix, N, X)
    # print("The objective function is: ")
    # print(obj)

    Q = get_Q(X, N, distanceMatrix, obj, enforce_start_city, start_city, elements)

    # print("The Q matrix for the QUBO problem is: ")
    # print(Q)
    np.save("Q_TSP_10", Q)

    # Optimise the problem using NEAL
    # Turn q matrix into bqm dict format
    bqm = dimod.BinaryQuadraticModel(Q, "BINARY")
    h, J, offset = dimod.qubo_to_ising(matrix_to_dict(Q))
    sio.savemat(
        "Q_ISING_TSP_10.mat",
        {"J": dict_to_mat(J), "h": dict_to_vect(h), "offset": offset},
    )
    np.savez("ISING_TSP_10", J=dict_to_mat(J), h=dict_to_vect(h), offset=offset)

    # Create the solver instance and solve it
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_sweeps=2000, num_reads=100)
    # Sort samples where the lowest energy is the first sample in the matrix
    animation_sample_size = 10
    sorted_samples_by_energy = sorted(sampleset.record, key=lambda x: x[1])
    sorted_samples_by_energy = [row[0] for row in sorted_samples_by_energy]
    sample_size = len(sorted_samples_by_energy)

    m, sch = parse_op_vec_tsp(sampleset.first.sample, N)

    print("The sample with the lowest energy variable configuration is: ")
    print(m)
    np.save("tsp_10_output_X", m)
    print("The calculated schedule is: ")
    print(sch)

    np.save("sorted_samples_by_energy.npy", sorted_samples_by_energy)

    # Use the schedule to compute the total distance in this path

    # distance = 0
    # for p in range(N-1):
    #     length = distanceMatrix[sch[p]][sch[p+1]]
    #     distance = distance + length
    # length = distanceMatrix[sch[N-1]][sch[0]]
    # distance = distance + length;
    # print("Total calculated distance: ")
    # print(distance)

    distance = get_distance(distanceMatrix, sch, N)
    print("Total calculated distance: ")
    print(distance)

    return sch, distance, sorted_samples_by_energy


if __name__ == "__main__":
    # List of city names
    cityNames = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ]

    # Latitude and longitude coordinates for the cities in the USA
    cityCoordinates = [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437),  # Los Angeles
        (41.8781, -87.6298),  # Chicago
        (29.7604, -95.3698),  # Houston
        (33.4484, -112.0740),  # Phoenix
        (39.9526, -75.1652),  # Philadelphia
        (29.4241, -98.4936),  # San Antonio
        (32.7157, -117.1611),  # San Diego
        (32.7767, -96.7970),  # Dallas
        (37.3382, -121.8863),  # San Jose
    ]

    # Index of the city to start with
    start_city = 0

    # Do we want to enforce start city: 0 no, 1 yes
    enforce_start_city = 1

    distanceMatrix = get_distance_matrix(cityCoordinates)
    np.save("distanceMatrix_TSP_10.npy", distanceMatrix)
    sch, distance, sorted_samples_by_energy = get_solution_tsp(
        cityNames, distanceMatrix, start_city, enforce_start_city
    )

    fig, ax = plt.subplots(figsize=(8, 10))
    fig2, ax2 = plt.subplots(figsize=(8, 10))

    # Create the animation

    animation = FuncAnimation(
        fig,
        partial(
            draw_solution_tsp,
            ax=ax,
            ax2=ax2,
            cityNames=cityNames,
            cityCoordinates=cityCoordinates,
            N=N,
            sorted_samples_by_energy=sorted_samples_by_energy,
        ),
        frames=100,
        interval=100,
        repeat=True,
    )
    animation2 = FuncAnimation(
        fig2,
        partial(
            draw_solution_tsp,
            ax=ax,
            ax2=ax2,
            cityNames=cityNames,
            cityCoordinates=cityCoordinates,
            N=N,
            sorted_samples_by_energy=sorted_samples_by_energy,
        ),
        frames=100,
        interval=100,
        repeat=True,
    )

    # Show the plot
    plt.show()

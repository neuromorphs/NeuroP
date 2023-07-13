import itertools
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from functools import partial
from sympy import *
import os
import scipy.io as sio
import dimod
import neal

# Function to calculate distance between two cities using Haversine formula
def calculateDistance(city1, city2):
    # Earth radius in meters
    R = 6371000

    # Convert latitude and longitude to radians
    lat1, lon1 = map(radians, city1)
    lat2, lon2 = map(radians, city2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def get_distance_matrix(cityCoordinates):
    N = int(len(cityCoordinates)) #numNodes

    # Calculate distance matrix
    distanceMatrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            distanceMatrix[i][j] = calculateDistance(cityCoordinates[i], cityCoordinates[j])
    print("Distance matrix calculated")

    return distanceMatrix

def get_obective_fct(cityNames, distanceMatrix, N, X):
    # Define objective function
    obj = 0

    print(N)
    # Iterate over each element of the symbolic matrix
    for i in range(N):
        for j in range(N):
            for p in range(N-1):
                obj = obj + distanceMatrix[i][j]*X[i,p]*X[j,p+1]

            obj = obj + distanceMatrix[i][j]*X[i,0]*X[j,N-1]

    return obj

def get_Q(X, N, distanceMatrix, obj, enforce_start_city, start_city, elements):
    # Add all symbolic elements of each row together
    row_sums = [sum(X.row(i)) for i in range(N)]
    col_sums = [sum(X.col(i)) for i in range(N)]

    # Construct a new symbolic matrix from the row sums
    C1 = (Matrix(row_sums) - Matrix(np.ones(N))).T.multiply((Matrix(row_sums) - Matrix(np.ones(N))))
    C2 = (Matrix(col_sums) - Matrix(np.ones(N))).T.multiply((Matrix(col_sums) - Matrix(np.ones(N))))

    # Scale to maximum coeff + 1
    Penalty = np.max(distanceMatrix) + 1

    # Construct the objective function
    obj_comb = Matrix([obj]) + Penalty*(C1.row(0) + C2.row(0) + enforce_start_city*Matrix([pow(X[start_city,0]-1,2)]))

    # Compute the quadratic terms using the Hessian matrix
    hess = hessian(obj_comb, elements)/2

    # Compute the linear terms based on binary variable constraint
    gradient = lambda f, v: Matrix([f]).jacobian(v)
    grad = gradient(obj_comb, elements)
    b = np.array([row.as_coefficients_dict()[1] for row in grad])
    diag_m_b = np.diag(b)
    Q = hess + diag_m_b
    Q = np.array(Q.tolist(), dtype=float)

    return Q

def matrix_to_dict(matrix):
    dictionary = {}
    rows, columns = matrix.shape

    for i in range(rows):
        for j in range(columns):
            key = (i, j)
            value = matrix[i, j]
            dictionary[key] = value

    return dictionary

def dict_to_mat(dictionary):
    # Extract the keys and values from the dictionary
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # Determine the matrix dimensions based on the keys
    num_rows = max([key[0] for key in keys]) + 1
    num_cols = max([key[1] for key in keys]) + 1

    # Create the NumPy matrix
    matrix = np.zeros((num_rows, num_cols))

    # Fill the matrix with the values from the dictionary
    for key, value in zip(keys, values):
        matrix[key] = value

    return matrix

def dict_to_vect(dictionary):
    # Extract the keys and values from the dictionary
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # Determine the matrix dimensions based on the keys
    num_rows = len(values)

    # Create the NumPy matrix
    matrix = np.zeros(num_rows)

    # Fill the matrix with the values from the dictionary
    for key, value in zip(keys, values):
        matrix[key] = value

    return matrix

# Map the state vector back to the city index in a schedule vector
def parse_op_vec_tsp(sample, N):
    dim = N
    m = np.empty([dim, dim])
    sch = np.empty([dim])
    for i in range(dim):
        for j in range(dim):
            m[i, j] = sample[j + i * dim]
            if m[i, j] == 1:
                sch[j] = i
    return m, sch.astype(int)

def get_distance(distanceMatrix, sch, N):
    distance = 0
    for p in range(N-1):
        length = distanceMatrix[sch[p]][sch[p+1]]
        distance = distance + length
    length = distanceMatrix[sch[N-1]][sch[0]]
    distance = distance + length;     
    return distance

# import itertools
# from math import radians, sin, cos, sqrt, atan2
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# from matplotlib.animation import FuncAnimation
# from functools import partial
# import numpy as np
# import random

# from sympy import *

# import os
# import scipy.io as sio

# import dimod
# import neal

# # Function to calculate distance between two cities using Haversine formula
# def calculateDistance(city1, city2):
#     # Earth radius in meters
#     R = 6371000

#     # Convert latitude and longitude to radians
#     lat1, lon1 = map(radians, city1)
#     lat2, lon2 = map(radians, city2)

#     # Haversine formula
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))
#     distance = R * c

#     return distance

# def get_distance_matrix(cityNames, cityCoordinates, N=N):

#     # Calculate distance matrix
#     distanceMatrix = [[0] * N for _ in range(N)]
#     for i in range(N):
#         for j in range(N):
#             distanceMatrix[i][j] = calculateDistance(cityCoordinates[i], cityCoordinates[j])
#     print("Distance matrix calculated")

#     return distanceMatrix

# def get_obective_fct(cityNames, distanceMatrix, N, X):
#     #Define objective function
#     obj =0;

#     print(N)
#     # Iterate over each element of the symbolic matrix
#     for i in range(N):
#         for j in range(N):
#             for p in range(N-1):
#                 obj = obj + distanceMatrix[i][j]*X[i,p]*X[j,p+1]
            
#             obj = obj + distanceMatrix[i][j]*X[i,0]*X[j,N-1]
            
#     return obj

# def get_Q(X, N, distanceMatrix, obj, enforce_start_city, start_city, elements):
#     # Add all symbolic elements of each row together
#     row_sums = [sum(X.row(i)) for i in range(N)] 
#     col_sums = [sum(X.col(i)) for i in range(N)] 

#     # Construct a new symbolic matrix from the row sums
#     C1 = (Matrix(row_sums) - Matrix(np.ones(N)))
#     C2 = (Matrix(col_sums) - Matrix(np.ones(N)))

#     C1 = C1.T.multiply(C1);
#     C2 = C2.T.multiply(C2);

#     #Scale to maximum coeff + 1
#     Penalty = np.max(distanceMatrix)+1;
#     #print("the Penalty is: " + str(Penalty))

#     #print('The C1 matrix:')
#     #print(C1)
#     #print('The C2 matrix:')
#     #print(C2)

#     #Construct the objective function
#     obj_comb = Matrix([obj]) + Penalty*(C1.row(0) + C2.row(0) + enforce_start_city*Matrix([pow(X[start_city,0]-1,2)]));
#     #print(obj_comb);

#     #Compute the quadratic terms using the Hessian matrix
#     hess = hessian(obj_comb, elements )/2;
#     #print(hess.shape)


#     #Compute the linear terms based on binary variable constraint 
#     gradient = lambda f, v: Matrix([f]).jacobian(v)
#     grad = gradient(obj_comb, elements)
#     b = np.empty([N*N])
#     for i,row in enumerate(grad,start=0):
#         coeff = row.as_coefficients_dict()[1]
#         b[i] = coeff
#     diag_m_b = np.diag(b)
#     Q = hess + diag_m_b
#     Q = np.array(Q.tolist(), dtype=float)

#     return Q

# def matrix_to_dict(matrix):
#     dictionary = {}
#     rows, columns = matrix.shape

#     for i in range(rows):
#         for j in range(columns):
#             key = (i,j)
#             value = matrix[i, j]
#             dictionary[key] = value

#     return dictionary

# def dict_to_mat(dict):
#     # Extract the keys and values from the dictionary
#     keys = list(dict.keys())
#     values = list(dict.values())

#     # Determine the matrix dimensions based on the keys
#     num_rows = max([key[0] for key in keys]) + 1
#     num_cols = max([key[1] for key in keys]) + 1

#     # Create the NumPy matrix
#     matrix = np.zeros((num_rows, num_cols))

#     # Fill the matrix with the values from the dictionary
#     for key, value in zip(keys, values):
#         matrix[key] = value

#     return matrix

# def dict_to_vect(dict):
#     # Extract the keys and values from the dictionary
#     keys = list(dict.keys())
#     values = list(dict.values())

#     # Determine the matrix dimensions based on the keys
#     num_rows = len(values)
   

#     # Create the NumPy matrix
#     matrix = np.zeros(num_rows)

#     # Fill the matrix with the values from the dictionary
#     for key, value in zip(keys, values):
#         matrix[key] = value

#     return matrix

# # lets map the state vector back to our city index in a schedule vector
# def parse_op_vec_tsp(sample,N):
#     dim = N;
#     m = np.empty([dim,dim]);
#     sch = np.empty([dim]);
#     for i in range(0,dim):
#         for j in range(0,dim):
#             m[i,j] = (sample[j + i*dim])
#             if m[i,j]==1:
#                 sch[j] = i;
#     return m,sch.astype(int)
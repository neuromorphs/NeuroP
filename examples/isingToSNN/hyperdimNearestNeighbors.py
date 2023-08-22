import numpy
import itertools

def nearestNeighborsCalculator(systDim, spinVar):

    ##### Euclidian space definitions #####
    axisVal = numpy.arange((-(spinVar-1)/2), ((spinVar-1)/2+1), 1)
    axisLoop = numpy.roll(axisVal, -int(spinVar/2))
    if spinVar % 2 == 0:
        shiftVal = axisLoop[0]
        axisLoop -= shiftVal
        axisVal -= shiftVal
    axisVal = axisVal.astype(int)
    axisLoop = axisLoop.astype(int)

    ##### Vectorial space definitions #####
    indexCoord = list(enumerate(itertools.product(axisVal, repeat=systDim)))
    coordIndex = {str(coord): i for i, coord in indexCoord}

    ##### Nearest neighbors calculator #####
    nearestNeighbors = [0 for _ in range(spinVar**systDim)]
    for i, coord in indexCoord:
        tmp = []

        for i1 in range(systDim):
            for v in [-1, 1]:
                shiftVec = numpy.zeros(systDim, dtype=int)
                shiftVec[i1] = -1

                nearestNeighbor = coord-shiftVec*v
                for i2 in range(systDim):
                    nearestNeighbor[i2] = axisLoop[nearestNeighbor[i2]]
                tmp.append(coordIndex[str(tuple(nearestNeighbor))])

        nearestNeighbors[i] = tmp


    ##### Adiacency Matrix #####
    adiacencyMatrix = numpy.zeros((spinVar**systDim, spinVar**systDim), dtype=int)
    for i1 in range(spinVar**systDim):
        for i2 in range(systDim*2):
            adiacencyMatrix[i1, nearestNeighbors[i1][i2]] = 1

    return adiacencyMatrix

if __name__ == '__main__':
    systDim = 2
    spinVar = 3
    adiacencyMatrix = nearestNeighborsCalculator(systDim, spinVar)
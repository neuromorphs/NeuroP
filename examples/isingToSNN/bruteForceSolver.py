import numpy
import itertools
from hyperdimNearestNeighbors import nearestNeighborsCalculator as nearestNeighborsCalculatorAlias

def bruteForce(systDim, spinVar, J):

    ##### Possible spin state #####
    spinConf = [numpy.reshape(conf, (spinVar**systDim, 1)) for conf in list(itertools.product([-1, 1], repeat=spinVar**systDim))]


    ##### Minimum energy #####
    H = numpy.zeros((2**(spinVar**systDim), 1))
    for i in range(2**(spinVar**systDim)):
        H[i] = -numpy.matmul(spinConf[i].T, numpy.matmul(J, spinConf[i]))*0.5

    minH = numpy.min(H)
    minConf = []
    for i in range(2 ** (spinVar ** systDim)):
        if minH == H[i]:
            minConf.append(spinConf[i])
    return minH, minConf

if __name__ == '__main__':
    systDim = 2
    spinVar = 3
    J = nearestNeighborsCalculatorAlias(systDim, spinVar)
    minH, minConf = bruteForce(systDim, spinVar, J) # test with uniform weights
    print(f'Minimum energy:\t {minH:0.03f}')
    print(f'Configuration:')
    for conf in minConf:
        print(f'\t{conf.T[0]}')


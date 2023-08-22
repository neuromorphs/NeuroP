import numpy as np

from hyperdimNearestNeighbors import *
from bruteForceSolver import *

from pygenn import GeNNModel
from pygenn import genn_model
import matplotlib.pyplot as plt
import os


##################################
# ##### Problem definition ##### #
##################################
##### Lattice dimension #####
systDim = 2
spinVar = 3
print(f'Total state configuration: {2**(spinVar**systDim):.2E}')

adiacencyMatrix = nearestNeighborsCalculator(systDim, spinVar)

##### Nearest Neighbors mask matrix #####
maskJ = np.triu(adiacencyMatrix)

##### Weights matrix #####
# np.random.seed(132)
# J = np.random.randint(-2, 3, (spinVar**systDim, spinVar**systDim))*maskJ
# J = np.random.random((spinVar**systDim, spinVar**systDim))*maskJ
# J = np.random.uniform(-1.0, 1.0, (spinVar**systDim, spinVar**systDim))*maskJ
J = np.random.choice([-1, 1], (spinVar**systDim, spinVar**systDim))*maskJ
# J = adiacencyMatrix*maskJ
J = (J+J.T)

##### Enable or disable the brute force calculation #####
minH, minConf = bruteForce(systDim, spinVar, J) # test with uniform weights
print(f'Minimum energy:\t {minH:0.03f}')
print(f'Configuration:')
for conf in minConf:
    print(f'\t{conf.T[0]}')


#####################################
# ##### GeNN Model definition ##### #
#####################################
##### Model definition #####
model_name = np.random.randint(10000)
model = GeNNModel(precision='float', model_name=f'{model_name}', backend='SingleThreadedCPU')
model.dT = 1.0


##### Neuron parameters #####
paramLif = {
    'C': 0.25,  # nF
    'TauM': 20.0,  # ms
    'Ioffset': 0.0,  # nA
    'Vrest': -65.0,  # mV
    'Vthresh': -50.0,  # mV
    'Vreset': -70.0,  # mV
    'TauRefrac': 2.0,  # ms
}
varLif = {
    'V': paramLif['Vrest'],  # mV
    'RefracTime': 0.0,  # ms
}


#################################
# ##### Neuron population ##### #
#################################
spinState = 2
popNeur = 30
##### Stimulus populations #####
stimPop = [0 for _ in range(spinVar**systDim)]
for spin in range(len(stimPop)):
    stimPop[spin] = model.add_neuron_population(
        pop_name=f'dtimPop{spin}',
        num_neurons=spinState*popNeur,
        neuron='PoissonNew',
        param_space={'rate': 20}, var_space={'timeStepToSpike': 0}
    )
    # stimPop[spin].spike_recording_enabled = True

custom_poisson = genn_model.create_custom_neuron_class(
    'custom_poisson',
    param_names=['start', 'stop', 'rate'],
    threshold_condition_code='$(t) > $(start) && $(t) <= $(stop) && $(gennrand_uniform) >= exp(-$(rate)/1000)',
    is_auto_refractory_required=False
)
dissPop = [0 for _ in range(spinVar**systDim)]
for spin in range(len(stimPop)):
    dissPop[spin] = model.add_neuron_population(
        pop_name=f'dissPop{spin}',
        num_neurons=spinState*popNeur,
        neuron=custom_poisson,
        param_space={'start': 3000, 'stop': 60000, 'rate': 20}, var_space={}
    )
    # dissPop[spin].spike_recording_enabled = True

##### Variables populations #####
varPop = [0 for _ in range(spinVar**systDim)]
for spin in range(len(varPop)):
    varPop[spin] = model.add_neuron_population(
        pop_name=f'varPop{spin}',
        num_neurons=spinState*popNeur,
        neuron='LIF',
        param_space=paramLif, var_space=varLif
    )
    varPop[spin].spike_recording_enabled = True


###################################
# ##### Synaptic connection ##### #
###################################
##### Stimulus excitation #####
for spin in range(len(stimPop)):
    synapsesExcit = np.zeros(shape=(spinState*popNeur, spinState*popNeur))
    for i in range(spinState*popNeur):
        weight = np.random.uniform(low=1.4, high=1.4)
        synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name=f'stimExcit{spin}', matrix_type='DENSE_INDIVIDUALG',
        source=stimPop[spin], target=varPop[spin],
        postsyn_model='ExpCurr',
        ps_param_space={"tau": 5.0}, ps_var_space={},
        delay_steps=2,
        w_update_model='StaticPulse',
        wu_param_space={}, wu_var_space={'g': synapsesExcit.flatten()},
        wu_pre_var_space={}, wu_post_var_space={}
    )
##### Dissipation inhibition #####
for spin in range(len(stimPop)):
    synapsesInhi = np.zeros(shape=(spinState*popNeur, spinState*popNeur))
    for i in range(spinState*popNeur):
        weight = np.random.uniform(low=-14.5, high=-4.5)
        synapsesInhi[i, i] = weight
    model.add_synapse_population(
        pop_name=f'dissExcit{spin}', matrix_type='DENSE_INDIVIDUALG',
        source=dissPop[spin], target=varPop[spin],
        postsyn_model='ExpCurr',
        ps_param_space={"tau": 5.0}, ps_var_space={},
        delay_steps=2,
        w_update_model='StaticPulse',
        wu_param_space={}, wu_var_space={'g': synapsesInhi.flatten()},
        wu_pre_var_space={}, wu_post_var_space={}
    )

##### Internal inhibition #####
for spin in range(len(varPop)):
    synapsesInhib = np.zeros(shape=(spinState*popNeur, spinState*popNeur))
    for so in range(spinState*popNeur):
        for to in range(spinState*popNeur):
            if so//popNeur != to//popNeur:
                weight = np.random.uniform(low=-0.2, high=0.0)
                synapsesInhib[so, to] = weight*0
    model.add_synapse_population(
        pop_name=f'inteInhib{spin}', matrix_type='DENSE_INDIVIDUALG',
        source=varPop[spin], target=varPop[spin],
        postsyn_model='ExpCurr',
        ps_param_space={"tau": 5.0}, ps_var_space={},
        delay_steps=2,
        w_update_model='StaticPulse',
        wu_param_space={}, wu_var_space={'g': synapsesInhib.flatten()},
        wu_pre_var_space={}, wu_post_var_space={}
    )

##### Lateral inhibition #####
for spinSource in range(len(varPop)):
    for spinTarget in range(len(varPop)):
        nn = adiacencyMatrix[spinSource, spinTarget]
        if nn != 0:
            synapses = np.zeros(shape=(spinState*popNeur, spinState*popNeur))
            weight = J[spinSource, spinTarget]
            for so in range(spinState*popNeur):
                for to in range(spinState*popNeur):
                    if so//popNeur == to//popNeur:
                        synapses[so, to] = weight
                    elif so//popNeur != to//popNeur:
                        synapses[so, to] = -1*weight
            model.add_synapse_population(
                pop_name=f'lateInhib{spinSource}_{spinTarget}', matrix_type='DENSE_INDIVIDUALG',
                source=varPop[spinSource], target=varPop[spinTarget],
                postsyn_model='ExpCurr',
                ps_param_space={"tau": 5.0}, ps_var_space={},
                delay_steps=2,
                w_update_model='StaticPulse',
                wu_param_space={}, wu_var_space={'g': synapses.flatten()},
                wu_pre_var_space={}, wu_post_var_space={}
            )


##########################
# ##### Simulation ##### #
##########################
time_simulation = 60  # s
time_steps = int(time_simulation*1e3)
model.build()
model.load(num_recording_timesteps=time_steps)

while model.timestep < time_steps:
    model.step_time()
model.pull_recording_buffers_from_device()

varPopRec = [0 for _ in range(spinVar**systDim)]
for spin in range(len(varPopRec)):
    times, index = varPop[spin].spike_recording_data
    records = [[] for _ in range(spinState*popNeur)]
    for t, i in list(zip(times, index)):
        records[i].append(t)
    varPopRec[spin] = records

finalConf = np.zeros((len(varPopRec), 1), dtype=int)
for spin in range(len(varPopRec)):
    spikeStateUp = np.sum([len(n) for n in varPopRec[spin][0:popNeur]])
    spikeStateDo = np.sum([len(n) for n in varPopRec[spin][popNeur:]])

    if spikeStateUp > spikeStateDo:
        finalConf[spin] = 1
    elif spikeStateUp < spikeStateDo:
        finalConf[spin] = -1

HSNN = -np.matmul(finalConf.T, np.matmul(J, finalConf))[0, 0]*0.5
print(f'Minimum energy:\t {HSNN:0.03f}')
print(f'Configuration SNN:\n\t{finalConf.T[0]}')


########################################
# ##### Graphical representation ##### #
########################################
##### Graphical representation of neurons activity #####
figsize = np.array((6.4*9, 6.4))
fig, axs = plt.subplots(nrows=1, ncols=spinVar**systDim, figsize=figsize)
for spin in range(spinVar**systDim):
    axs[spin].eventplot(varPopRec[spin])
    axs[spin].set_xticks([])
    axs[spin].set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

##### Delete simulation foldar after execution #####
os.system('rm -r *_CODE')

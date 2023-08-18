"""Base-classes for neuromorphic hardware that can run models derived from QUBO.

.. autoclass:: neurop.Backend.SimulatedAnnealingBackend
"""

import numpy as np
import random
from collections.abc import Iterable
from neurop import BaseBackend
from neurop import BaseModel
from neurop.model import QUBOModel
from tqdm import tqdm

class SimulatedAnnealingBackend(BaseBackend):
    """A software simulation of annealing to minimize QUBO problems."""
    def __init__(self, temperatures) -> None:
        self.temperatures = temperatures
        super().__init__()
    
    def supports_model(self, model: BaseModel) -> bool:
        return type(model) in [QUBOModel]
    
    def run(self, model: BaseModel, **kwargs):
        if not self.supports_model(model):
            raise ValueError("The given model is not supported by this backend!")
        
        if type(model) == QUBOModel:
            init = model.initializer()
            E,x = self.run_qubo(model.Q, np.array([init[var] for var in model.variables]).reshape((-1,1)))
            return int(E), {var: x[i].item() for i,var in enumerate(model.variables)}
        
    def modQ(self, Q):
        newQ = np.triu(Q) + np.triu(Q).T
        np.fill_diagonal(newQ, np.diag(Q))
        return newQ

    def run_qubo(self, Q, x_init):
        #Q = self.modQ(Q)
        N = int(np.shape(Q)[0])
        
        
        x = x_init.astype(int)
        x_best = x.copy()
        
        best_E = cur_E = x.T @ Q @ x
        #best_cut_trace = np.zeros((math.ceil(heat_max/100/2e-3)+1))

        for temp in (pbar := tqdm(self.temperatures)):
            pbar.set_description("Temperature: {}  Current energy: {}".format(temp, cur_E.item()), refresh=False)
            n = random.randint(0,N-1)
            Qnj = Q[n,:]

            # flip the bit at position n
            x_new = x.copy()
            x_new[n] = 1-x[n]
            
            # old = np.sum(np.multiply((x_old * x), Qij))
            # new = np.sum(np.multiply((x_tmp * x_new), Qij))
            # chg = float(new - old)

            #chg = float(np.inner((x_tmp * x_new.T - x_old * x.T), Qij))
            chg = (1-2*int(x[n]))*(2*(Qnj @ x) + Qnj[n]*(1-2*x[n]))

            r = random.random()
            #temp *= 1+np.abs(best_E)
            tmp = np.clip(-chg/temp, -100, 100)

            #print("Energy change: {} Acceptance: {}".format(chg, np.exp(tmp)))

            if r <= np.exp(tmp):

                x = x_new.copy()
                cur_E += chg

                if cur_E < best_E:
                    x_best = x.copy()
                    best_E = cur_E

        return best_E, x_best

        

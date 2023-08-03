"""QUBO model.

.. autoclass:: neurop.Model.QUBOModel
"""

from neurop import BaseModel

class QUBOModel(BaseModel):
    def __init__(self, Q, variables, initializer) -> None:
        self.Q  = Q
        super().__init__(variables=variables, initializer=initializer)
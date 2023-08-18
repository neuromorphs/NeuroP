"""SNN model.

.. autoclass:: neurop.Model.SNNModel
"""
from neurop import BaseModel


class SNNModel(BaseModel):
    def __init__(self, W, b, theta, alpha, initializer, backend) -> None:
        super().__init__(initializer=initializer, backend=backend)
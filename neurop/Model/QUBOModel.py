from neurop import BaseModel


class QUBOModel(BaseModel):
    def __init__(self, Q, variables, initializer, from_problem_parameters, to_problem_parameters, backend) -> None:
        self.Q  = Q
        super().__init__(variables=variables, initializer=initializer, from_problem_parameters=from_problem_parameters, to_problem_parameters=to_problem_parameters, backend=backend)
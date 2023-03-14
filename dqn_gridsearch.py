from dpn_hyperparameters import DQNHyperparameters
from typing import List


class DQNGridsearch():

    def __init__(self, lr_parmas = [0.0005, 0.005], gamma_params = [0.99, 0.1]):
        """Initialize a DQNGridsearch object.
        
        Params
        ======
            lr_parmas (array): learning rate parameters
            gamma_params (array): discount factor parameters
        """
        self.lr_params = lr_parmas
        self.gamma_params = gamma_params

    def create_gridsearch_params(self) -> List:
        """Returns an array with DQNHyperparameters objects."""
        dqn_param_list = [DQNHyperparameters(lr=lr, gamma=g) for lr in self.lr_params for g in self.gamma_params]
        return dqn_param_list
    
    def __str__(self) -> str:
        """Returns a DQNGridsearch object as a string."""
        return f"""lr: {self.lr_params}
gamma: {self.gamma_params}"""
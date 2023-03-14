class DQNHyperparameters():

    def __init__(self,  lr, gamma, buffer_size = int(1e5), batch_size = 64, tau = 1e-3, update_every = 4):
        """Initialize a DQNHyperparameters object.
        
        Params
        ======
            lr (float): learning rate
            gamma (float): discount factor
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            tau (float): for soft update of target parameters
            update_every (int): how often to update the network
        """
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every

    def __str__(self) -> str:
        """Returns a DQNHyperparameters object as a string."""
        return f"""lr: {self.lr}
gamma: {self.gamma}
buffer_size: {self.buffer_size}
batch_size: {self.batch_size}
tau: {self.tau}
update_every: {self.update_every}"""
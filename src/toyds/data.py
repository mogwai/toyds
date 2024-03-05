from torch.utils.data import IterableDataset
import random

class ToyDataset(IterableDataset):
    """
    Produces an infinite stream of toy tasks for training debugging
    """

    def __init__(self, challenges:list[callable], probs:list[float]=None, vocab_size=512):
        self.challenges = challenges
        self.probs = probs

        if self.probs is None:
           self.probs = [1/len(challenges)]*len(challenges)

        assert sum(self.probs) == 1

        self.pad_token = 0
        self.eos_token = 1
        # Start sequence or used to denote commands
        self.command_token = 2
        self.tokens = vocab_size - 3

    def gen(self):
        while True:
            func = random.choices(self.challenges, self.probs)[0]
            yield func()

    def __iter__(self):
        return self.gen()



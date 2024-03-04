from torch.utils.data import IterableDataset
import random

class ToyDataset(IterableDataset):

    def __init__(self, challenges:list[callable], probs:list[float]=None, vocab_size=512):
        # A task might need to create vocab tokens
        self.challenges = challenges

        if probs is None:
            probs = [1/len(challenges)]*len(challenges)

        assert sum(probs) == 1

        self.pad_token = 0
        self.eos_token = 1
        # Start sequence or used to denote commands
        self.command_token = 2
        # Determine the vocabulary size based on the problem
        self.tokens = vocab_size - 3


    def __iter__(self):
        while True:
            yield random.choices(self.challenges, self.probs)()


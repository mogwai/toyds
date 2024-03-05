from torch.utils.data import IterableDataset
import random

class ToyDataset(IterableDataset):
    """
    Produces an infinite stream of toy tasks for training debugging
    """

    def __init__(self, challenges:list[callable], probs:list[float]=None):
        self.challenges = challenges
        self.probs = probs

        if self.probs is None:
           self.probs = [1/len(challenges)]*len(challenges)

        assert sum(self.probs) == 1

    def gen(self):
        while True:
            func = random.choices(self.challenges, self.probs)[0]
            yield func.generate(), func

    def __iter__(self):
        return self.gen()


def collate_fn(batch):
    sequences = [b[0] for b in batch]
    lengths = torch.tensor([s.shape[-1] for s in sequences])
    loss_funcs = {}
    for i, s in enumerate(batch):
        task = s[1]
        if task.name not in loss_funcs:
            loss_funcs[task.name] = {
                "loss": task.train,
                "items": []
            }
        loss_funcs[task.name]["items"].append(i)

    tokens = pad_sequence(sequences, batch_first=True)
    return {"tokens": tokens, "loss_funcs": loss_funcs, "lengths": lengths}

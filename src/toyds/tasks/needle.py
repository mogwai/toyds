import torch
import random
import torch.nn.functional as F
from toyds.tasks.task import Task
from toyds.utils import cat

def filter_seq(vocab_size=100, max_seq_len=100, querylen=None):
    """
    Given a sequence, filter out all but a queried set of numbers in the prompt, 
    producing the numbers from the original sequnce in order that they appear.

    Example:

    [D, D, C, A, A, A, B, C, D, |, A, |, A, A, A ]

    [D, D, A, B, D, D, C, A, C, |, C, A, | A, C, A, C ]

    """
    querylen = vocab_size//10
    # Has to be at least one thing to search for
    minL = 5
    length = random.randint(minL, max_seq_len//2)

    vocab_size -= 1
    query = random.sample(range(3, vocab_size),k=querylen)
    haystack = torch.randint(3, vocab_size, (length,))
    indices = [(haystack == elem).nonzero() for elem in query if (haystack== elem).any()]
    
    if len(indices):
        indices = torch.cat(indices)
    
    result = haystack[indices].flatten()
    haystack = cat(haystack, 2, query, 2, result, 1).long()

    if haystack.shape[-1] > max_seq_len:
        # Try again
        return filter_seq(vocab_size, max_seq_len, querylen)
    else:
        return haystack

class Filter(Task):

    def generate(self):
        return filter_seq(*self.args, **self.kwargs)

    def train(self, logits, seq, lengths):
        # Find the index of the last 2 (command token)
        # Command to 1 is the sequence

        targets = torch.stack([seq[i, lengths[i]-1] for i in range(len(lengths))])
        return F.cross_entropy(logits, targets)


def lookup_item(vocab_size=1000, max_seq_len=5000, occurences=1):
    """
    With a large vocab size, determine wether and item is an a sequence

    [A, ......, Z, | Z | T]
    """

    ML = max_seq_len - 4

    # Has to be at least one thing to search for
    minL = 5
    length = random.randint(minL, ML)

    vocab_size -= 1

    randtoken = lambda:  random.randint(4, vocab_size)
    needle = randtoken()
    haystack = torch.randint(4, vocab_size, (length,))

    for i in (haystack == needle).nonzero():
        rand = randtoken()

        while rand == needle:
            rand = randtoken()

        haystack[i] = rand

    contains = random.random() < .5
    if contains:
        idxs = random.sample(list(range(len(haystack))), k=occurences)
        haystack[idxs] = needle

    # 0 pad 1 EOS 2 Command 3 True 4 False
    contains = 3 if contains else 4
    haystack = cat(haystack, 2, needle, 2, contains)
    return haystack


class LookupItem(Task):

    def generate(self):
        return lookup_item(*self.args, **self.kwargs)

    def train(self, logits, seq, lengths):
        """
        We only really need the last logits
        """
        logits = torch.stack([logits[i, lengths[i]-1] for i in range(len(lengths))])
        targets = torch.stack([seq[i, lengths[i]-1] for i in range(len(lengths))])
        return F.cross_entropy(logits, targets)


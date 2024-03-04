import torch
import random

"""
Needle in the haystack problems

Will generate one sequence each call

These problems are good to check how well a model performs on long sequences lengths

"""



def filter_sequence(vocab_size=10, max_len=10):
    """
    Given a sequence, filter out all but a queried set of numbers  in the prompt, producing the numbers from the original sequnce in order that they appear.

    Examples:


    [D, D, C, A, A, A, B, C, D, |, A, |, A, A, A ]

    [D, D, A, B, D, D, C, A, C, |, C, A, | A, C, A, C ]

    """

    pass


def lookup_item(vocab_size=1000, max_len=5000, occurences=1):
    """
    This method require

    With a large vocab size, determine wether and item is an a sequence

    [A, ......, Z, | Z | T]
    """
    ML = max_len - 4
    length = random.randint(0, ML)

    lookfor = random.randint(5, vocab_size)
    tosearch = torch.randint(5, vocab_size, (length,))

    for i in (tosearch == lookfor).nonzero():
        rand = random.randint(5, vocab_size)

        while rand == lookfor:
            rand = random.randint(5, vocab_size)

        tosearch[i] = rand

    contains = random.random() < .5

    if contains:
        idxs = random.sample(list(range(len(tosearch))), k=occurences)
        tosearch[idxs] = lookfor

    # 0 pad 1 EOS 2 Command 3 True 4 False
    contains = 3 if contains else 4
    tosearch = torch.cat((tosearch, torch.tensor([2, lookfor, 2, contains])))

    print(tosearch.shape, tosearch[-5:])
    return tosearch


print(lookup_item())

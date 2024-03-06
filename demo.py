import toyds

"""
Filter Examples Sequences:
	| is command token
    
    [D, D, C, A, A, A, B, C, D, |,    A, |, A, A, A]
    [D, D, A, B, D, D, C, A, C, |, C, A, |, A, C, A, C]
"""
vs = 10
msl = 10


ds = toyds.ToyDataset([
    toyds.Filter(vocab_size=vs, max_seq_len=msl),
])

dsiter = iter(ds)


for i in range(2):
    print(next(dsiter)[0])

"""
Outputs:
0 is padding
1 is EOS
2 is Command token (|)

Find all 4's
tensor([8, 6, 3, 8, 6, 2, 4, 2, 1])

# Find all 8's
tensor([8, 6, 4, 7, 5, 2, 8, 2, 8, 1])
"""
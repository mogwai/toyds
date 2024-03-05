import toyds

vs = 10
msl = 10

ds = toyds.ToyDataset([
    toyds.LookupItem(vocab_size=vs, max_seq_len=msl),
    toyds.Filter(vocab_size=vs, max_seq_len=msl),
])

dsiter = iter(ds)


for i in range(10):
    print(next(dsiter))

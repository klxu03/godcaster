example = [("four", 4), ("five", 5), ("one", 1), ("two", 2), ("three five", 3.5), ("three", 3)]

min_ind = min((index for index, (word, number) in enumerate(example) if number > 1.5), key=lambda idx: example[idx][1], default=None)

print(min_ind)

example.pop(min_ind)

min_ind = min((index for index, (word, number) in enumerate(example) if number > 1.5), key=lambda idx: example[idx][1], default=None)

print(min_ind)
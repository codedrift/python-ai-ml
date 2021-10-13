import numpy as np

with np.load('./mnist.npz') as data:
    print(dir(data))
    # np.save("mnist", mnist);
    print(data.files)
    for file in data.files:
        print(data[file])
        np.save(file, data[file])

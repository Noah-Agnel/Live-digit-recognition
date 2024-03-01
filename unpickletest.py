import pickle
import gzip
import numpy as np

file = open("trainednet.pkl", 'rb')
network = pickle.load(file)

f = gzip.open('/Users/noah/Desktop/COMP Prog/digitrecog/data.pkl.gz', 'rb')
u = pickle._Unpickler( f )
u.encoding = 'latin1'
training_data, validation_data, test_data = u.load()
f.close()


for i in range(10):
    print(test_data[1][i])
    new = np.reshape(test_data[0][i], (784, 1))
    print(network.recognize(new))

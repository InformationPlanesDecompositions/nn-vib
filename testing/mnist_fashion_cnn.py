import struct
import numpy as np

def read_idx(filename):
  with open(filename, 'rb') as f:
    zero1, zero2, data_type, dims = struct.unpack('>BBBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
    data = np.frombuffer(f.read(), dtype=np.uint8 if data_type == 0x08 else None)
    return data.reshape(shape)

if __name__ == "__main__":
  train_images = read_idx('train-images-idx3-ubyte')
  train_labels = read_idx('train-labels-idx1-ubyte')

  #test_images = read_idx('t10k-images-idx3-ubyte')
  #test_labels = read_idx('t10k-labels-idx1-ubyte')

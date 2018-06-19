import numpy as np
import os,  sys                                                                                           
from six.moves import urllib
import tarfile
import gzip


#  process the mnist dataset
URL='https://storage.googleapis.com/cvdf-datasets/mnist/'
directory='data'
MNIST_DATA=['train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz']

## the image is 28X28X1, greyscale pictures
IMAGE_SIZE=28
NUM_CHANNELS=1
num_train=60000
num_valid=10000



def down_and_check_data():
    for filename in MNIST_DATA:
        if not os.path.exists(directory):
            print("path do not exist, create path: ",directory)
            os.makedirs(directory)
        dir_filename=os.path.join(directory,filename)
        if not os.path.exists(dir_filename):
            print(" file ",dir_filename,"not exists, begin downloading it")
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(URL+filename, dir_filename, _progress)
            print("file already downloaded, extracting it!!!")

def get_data_lables():
    """
    process the images, extract into 4D tensor [image index,y,x,channels]    
    data values rescaled from [0,255] to [-0.5,0.5]
    for the labels, extract into a vector of int64 IDs
    train_image     num_train   image
    train_labels    num_train   label
    valid_image     num_valid   image
    valid_labels    num_valid   label

    """   

    for idx in range(4):
        filename = MNIST_DATA[idx]

        number=0
        if(idx//2==0):
            number=num_train
        else:
            number=num_valid 
        print("for file: " ,filename  )
        filename=os.path.join(directory,filename)
        if(idx%2==0):
            with gzip.open(filename) as bytestream:
                bytestream.read(16)  
                buf=bytestream.read(IMAGE_SIZE*IMAGE_SIZE*number*NUM_CHANNELS)  
                data=np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
                print(type(buf),type(data),len(data))
                data=data/255-0.5
                data=data.reshape(number,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
                print(len(data))
                print(data)

        else:
            with gzip.open(filename) as bytestream:
                bytestream.read(8) 
                buf=bytestream.read(1*number)
                labels=np.frombuffer(buf,dtype=np.uint8).astype(np.int64)
                print(len(labels))
                print(labels)


def error_rate(predictions,labels):
    """
    Return the error rate based on dense predictions and sparse labels
    
    """
    result=np.sum(np.argmax(predictions,1)==labels)
    return result/predictions.shape[0]





down_and_check_data()
get_data_lables()




























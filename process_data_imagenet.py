import numpy as np
import os,  sys                                                                                           
from six.moves import urllib
import tarfile

#  process the mnist dataset
URL='https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY='data'
MNIST_DATA=['train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz']


# process the ImageNet files

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
directory="./data"

def down_and_check_data():
    if not os.path.exists(directory):
        print("path do not exist, create path: ",directory)
        os.makedirs(directory)
    filename=os.path.join(directory,DATA_URL.split('/')[-1])
    if not os.path.exists(filename):
        print(" file ",filename,"not exists, begin downloading it")
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filename, _progress)
        print("file already downloaded, extracting it!!!")
        tarfile.open(filename, 'r:gz').extractall(directory)

def get_data_lables(filename,num_image,width,height,channels):
    """
    process the images, extract into 4D tensor [image index,y,x,channels]    
    data values rescaled from [0,255] to [-0.5,0.5]
    for the labels, extract into a vector of int64 IDs
    """    
    with gzip.open(filename) as bytestream:
        bytestream.read(16)  
        buf=bytestream.read()  


def error_rate(predictions,labels):
    """
    Return the error rate based on dense predictions and sparse labels
    
    """
    result=np.sum(np.argmax(predictions,1)==labels)
    return result/predictions.shape[0]





down_and_check_data()



## construct the CNN
def



def 




## construct the ResNet network
def ResNet():






















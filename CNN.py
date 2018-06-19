import tensorflow as tf



#data type:   use float16 or float32 ?
DATA_TYPE=tf.float16


# this is for the mnist data set:
BATCH_SIZE=64
IMAGE_SIZE=28
IMAGE_SIZE=28
NUM_CHANNELS=1





train_data_node=tf.placeholder(
    DATA_TYPE,
    shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE.NUM_CHANNELS) )

train_labels_node=tf.placeholder(
    tf.int64,shape=(BATCH_SIZE,)),
    
eval_data=tf.placeholder(
    DATA_TYPE,
    shape=(EVAL_BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
    )

# construct the convolutional network
#  filter:  5X5  depth 32
#   
conv1_w=tf.Variable(
    tf.truncated_normal(
        [5,5,NUM_CHANNELS,32],
        stddev=0.1,
        seed=SEED,
        dtype=data_type()  )
        )
conv1_b= tf.Variable(tf.zeros([32], dtype=DATA_TYPE)  )


# construct the seconde CNN 
#    5X5  depth  64
#
#
conv2_w=tf.Variable(
    tf.truncated_normal(
        [5,5,32,64],
        stddev=0.1,
        seed=SEED,
        dtype=DATA_TYPE
        )
    )
conv2_b=tf.Variable(
    tf.constant(0.1,shape=[64],dtype=DATA_TYPE)
)

#construct fully connected network
# fully connected network,  depth: 512
fc1_w=tf.Variable(
    tf.truncated_normal(
        [5,5,32,64],stddev=0.1,
        seed=SEED,dtype=DATA_TYPE
    )
)

fc1_b = tf.Variable(
    tf.constant(
        0.1, shape=[512],
        dtype=DATA_TYPE
    )
)

fc2_w=tf.Variable(tf.truncated_normal(
    [512,NUM_LABELS],
    stddev=0.1,
    seed=SEED,
    dtype=DATA_TYPE
    ))

fc2_b = tf.Variable(tf.constant(
    0.1,shape=[NUM_LABELS],
    dtype=DATA_TYPE)
    )

####  replicate model structure for training subgraph
#### 

def model(data,train=False):
    """
    2D convolution, with 'SAME' padding(
    i.e. the output feature map has the same size as the input
    ) {strides} is 4D array: [image index,y,x,depth]
    """
    conv=tf.nn.conv2d(
        data,
        conv1_weight,
        strides=[1,1,1,1],        
        padding="same"
    )

    # add Bias and rectified linear non-linearity 
    ## RLU
    relu=tf.nn.relu(tf.nn.bias_add(conv,conv1_biases) )
    # max pooling. 
    pool=tf.nn.max_pool(
        relu,ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME'
        )
    conv=tf.nn.conv2d(
        pool,
        conv2_w,
        strides=[1,1,1,1],
        padding='SAME'
    )
    relu=tf.nn.relu(tf.nn.bias_add(conv,conv2_b) )
    pool=tf.nn.max_pool(
        relu,
        ksize=[1,2,2,1],    
        strides=[1,2,2,1],
        padding='SAME'
    )
    
    ### Reshape the feature map cuboid into a 2D matrix to feed it to 
    ### the fully connected layer
    pool_shape=pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3] ]
        )
    
    hidden=tf.nn.relu(tf.matul(reshape,fc1_weights)+fc1_b)
    # add a 50% dropout during training, 
    # 
    
    if train:
        hidden = tf.nn.dropout(hidden,0.5,seed=SEED)
    return tf.matmul(hidden,fc_weights) + fc2_b


# Training computation: logits + cross-entropy loss
#  
logits=model(train_data_node,True)
loss = tf.reduce_mean(
    rf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node,logits=logits
    )
)








































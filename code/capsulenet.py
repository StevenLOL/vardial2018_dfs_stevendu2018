"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...

    
"""

from keras import layers, models
from keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
max_features = 5000
maxlen = 60
embed_dim = 50




def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=(maxlen,))
    embed = layers.Embedding(max_features, embed_dim, input_length=maxlen)(x)

    conv1 = layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(
        embed)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(maxlen, activation='sigmoid')(x_recon)
    # x_recon = layers.Reshape(target_shape=[1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint], verbose=1)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-' * 50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-' * 50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()



def loadData(fname):
    return pd.read_csv(fname, sep='\t',header=-1)


#22ste

def toCnnIndexAllinOne(lin,LengthOfInputSequences=60):
    rv=[]
    for s in lin.split()[:LengthOfInputSequences]:
        if s in tfidfDict:
            rv.append(tfidfDict[s])
    while(len(rv)<LengthOfInputSequences):
        rv.append(0)
    return rv

def fitlerLine(lin):
    lin=lin.lower()
    for k in ['.',',','?',"'s","'t",',']:
        lin=lin.replace(k,' '+k+' ')
    #for k in ['.',',','?','!']:
    #    lin=lin.replace(k,' ')
    while '  ' in lin:
        lin=lin.replace('  ',' ')
    return lin.strip()
    
    
    
import pandas as pd
fileTrain='../DFS/train.txt'
fileDev='../DFS/dev.txt'
fileTest='../DFS/dfs-test.txt'
fileTest='../gold/DFS/dfs-gold.txt'
df=loadData(fileTrain)
trainxRaw=list(map(fitlerLine,df[0]))
trainyRaw=df[1].values
trainy=np.array(trainyRaw)
testdata=loadData(fileTest)
testy=testdata[1].values
testxRaw=list(map(fitlerLine,testdata[0]))
tfv = TfidfVectorizer(min_df=2,use_idf=1,
                  smooth_idf=1,ngram_range=(1,1),
                 )
tfv.fit(trainxRaw)
max_features=len(tfv.get_feature_names())
print(len(tfv.get_feature_names()),max_features)

tfidfDict={s:sindex for sindex,  s in enumerate(  tfv.get_feature_names())}
trainCnnIndex=list(map(toCnnIndexAllinOne,trainxRaw))
trainCnnIndex=np.array(trainCnnIndex)
testCnnIndex=list(map(toCnnIndexAllinOne,testxRaw))
testCnnIndex=np.array(testCnnIndex)
trainy=[0 if s=='DUT' else 1 for s in trainy]
testy=[0 if s=='DUT' else 1 for s in testy]    
trainy=np.array(trainy)
testy=np.array(testy)
def load_imdb(maxlen=60):
   

    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    #x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (trainCnnIndex, trainy), (testCnnIndex, testy)


if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lam_recon', default=0.0005, type=float)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_imdb()
    print(x_train.shape)
    
    print(y_train.shape)
    print(x_train[0])
    # define model
    model = CapsNet(input_shape=x_train.shape,
                    n_class=1,
                    num_routing=args.num_routing)
    model.summary()
    plot_model(model, to_file=args.save_dir + '/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=model, data=(x_test, y_test))

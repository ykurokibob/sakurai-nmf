import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from optimizer.utility import get_train_ops

old_session = K.get_session()

from pprint import pprint


class Mnist(object):
    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        with tf.Graph().as_default() as graph:
            model = Sequential()
            model.add(Dense(1000, activation='relu', input_shape=[784, ]))
            model.add(Dense(500, activation='relu'))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(10, activation=None))
            model.compile('rmsprop', 'mse', metrics=['acc'])
        self.graph = graph
        self.model = model
    
    def train(self, x, y, test_x, test_y, epochs=None, batch_size=100):
        self.history = self.model.fit(x, y,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_split=0.2)
        scores = self.model.evaluate(test_x, test_y)
        print(scores)


def main(_):
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.astype('float32').reshape(-1, 784) / 255.
    x_test = x_test.astype('float32').reshape(-1, 784) / 255.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(x_train.shape)
    print(x_test.shape)
    
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            K.set_session(sess)
            K.set_learning_phase(1)
            model = Mnist()
            pprint(get_train_ops(graph=model.graph))
            
            # model.train(x_train, y_train, x_test, y_test, epochs=10)


if __name__ == '__main__':
    tf.app.run()
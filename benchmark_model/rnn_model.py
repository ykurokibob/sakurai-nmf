"""64bit RNN model"""

import tensorflow as tf

# inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.GRU(32,
                        input_shape=(None, 10),
                        recurrent_dropout=0.5,
                        dropout=0.1,
                        return_sequences=True,
                        activation=tf.nn.relu)
# outputs = tf.keras.layers.SimpleRNN(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.add(GRU(32,
#               input_shape=(None, input_shape),
#               recurrent_dropout=0.5,
#               dropout=0.1,
#               return_sequences=True))
# model.add(GRU(64, activation='relu',
#               dropout=0.1,
#               recurrent_dropout=0.5))
# model.add(Dense(1))
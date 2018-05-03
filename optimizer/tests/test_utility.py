from pprint import pprint

import tensorflow as tf

import benchmark_model
import losses
from matrix_factorization import semi_nmf
from optimizer import utility

batch_size = 100
label_size = 1


class UtilityTest(tf.test.TestCase):
    def test_correct_bias(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(model.inputs, ops=ops)
        pprint(layers)
        self.assertEqual(layers[0].use_bias, True)
        self.assertEqual(layers[1].use_bias, False)
    
    def test_get_hidden_output(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(inputs=model.inputs, ops=ops)
        pprint(layers)
        self.assertEqual(layers[0].output.get_shape().as_list(), [100, 784])
        self.assertEqual(layers[1].output.get_shape().as_list(), [100, 100])
        self.assertEqual(layers[2].output.get_shape().as_list(), [100, 50])


class FactorizeTest(tf.test.TestCase):
    def test_simplest_factorize(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(inputs=model.inputs, ops=ops)
        
        hidden = layers[-1].output
        last_weights = layers[-1].kernel
        tf_u, tf_v = semi_nmf(model.labels, hidden, last_weights,
                              use_tf=True, use_bias=False, num_iters=2)
        _old_local_loss = losses.frobenius_norm(model.labels, hidden @ last_weights)
        _new_local_loss = losses.frobenius_norm(model.labels, tf_u @ tf_v)
        
        x, y = benchmark_model.build_data()
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            old_local_loss, new_local_loss = sess.run(
                [_old_local_loss, _new_local_loss],
                feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
            self.assertGreater(old_local_loss, new_local_loss)
            print("old {} new {}".format(old_local_loss, new_local_loss))


if __name__ == '__main__':
    tf.test.main()
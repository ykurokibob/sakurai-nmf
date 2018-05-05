from pprint import pprint

import tensorflow as tf

import benchmark_model
import losses
from matrix_factorization import semi_nmf
from optimizer import utility

mat_file = '../../matrix_factorization/tests/np_tests/small_v_neg.mat'


def print_format(lib, algo, a, u, v, old_loss, new_loss, duration):
    print('\n[{}]Solve {}\n\t'
          'a {} u {} v {}\n\t'
          'old loss {}\n\t'
          'new loss {}\n\t'
          'process duration {}'.format(
        lib, algo,
        a.shape, u.shape, v.shape, old_loss, new_loss, duration))


batch_size = benchmark_model.batch_size
label_size = benchmark_model.label_size


class UtilityTest(tf.test.TestCase):
    def test_correct_bias(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(model.inputs, ops=ops)
        pprint(layers)
        self.assertEqual(layers[0].use_bias, True)
        self.assertEqual(layers[1].use_bias, False)
    
    def test_correct_activation(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(model.inputs, ops=ops)
        self.assertEqual(layers[0].activation.type, 'Relu')
        self.assertEqual(layers[1].activation.type, 'Relu')
        self.assertEqual(layers[2].activation, None)
    
    def test_get_hidden_output(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(inputs=model.inputs, ops=ops, graph=None)
        pprint(layers)
        self.assertEqual(layers[0].output.get_shape().as_list(), [batch_size, 784])
        self.assertEqual(layers[1].output.get_shape().as_list(), [batch_size, 100])
        self.assertEqual(layers[2].output.get_shape().as_list(), [batch_size, 50])
    
    def test_combine_kernel_bias(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(inputs=model.inputs, ops=ops)
        layer = layers[0]
        v = tf.concat((layer.kernel, layer.bias[None, ...]), axis=0)
        v, bias = utility.split_v_bias(v)
        
        x, y = benchmark_model.build_data()
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            equal = tf.reduce_all(tf.equal(v, layer.kernel))
            equal = sess.run(equal, feed_dict={model.inputs: x})
            self.assertTrue(equal, msg='shape of factorized v should be kernel of shape.')
            equal = tf.reduce_all(tf.equal(bias, layer.bias))
            equal = sess.run(equal, feed_dict={model.inputs: x})
            self.assertTrue(equal, msg='shape of factorized bias should be bias of shape.')
    
    def test_get_placeholder_ops(self):
        print()
        model = benchmark_model.build_tf_one_hot_model(False)
        import time
        start_time = time.time()
        inputs, labels = utility.get_placeholder_ops(model.loss)
        duration = time.time() - start_time
        print('duration', duration)
        self.assertEqual(model.inputs, inputs)
        self.assertEqual(model.labels, labels)
    
    def test_other_model_get_placeholder_ops(self):
        model = benchmark_model.build_tf_model()
        import time
        start_time = time.time()
        inputs, labels = utility.get_placeholder_ops(model.loss)
        duration = time.time() - start_time
        print('duration', duration)
        self.assertEqual(model.inputs, inputs)
        self.assertEqual(model.labels, labels)
    
    def test_keras_get_placeholder_ops(self):
        _inputs, _labels, loss = benchmark_model.build_keras_model()
        import time
        start_time = time.time()
        inputs, labels = utility.get_placeholder_ops(loss)
        duration = time.time() - start_time
        print('duration', duration)
        self.assertEqual(_inputs, inputs)
        self.assertEqual(_labels, labels)


class FactorizeTest(tf.test.TestCase):
    def test_simplest_factorize(self):
        print()
        model = benchmark_model.build_tf_model()
        ops = utility.get_train_ops()
        layers = utility.zip_layer(inputs=model.inputs, ops=ops)
        
        hidden = layers[-1].output
        last_weights = layers[-1].kernel
        tf_u, tf_v = semi_nmf(model.labels, hidden, last_weights,
                              use_tf=True, use_bias=False, num_iters=3)
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
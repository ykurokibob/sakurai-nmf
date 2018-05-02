import tensorflow as tf

from benchmark_model.double_model import build_model
from optimizer.utility import get_train_ops, zip_layer


class UtilityTest(tf.test.TestCase):
    def test_correct_bias(self):
        print()
        model = build_model()
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            ops = get_train_ops()
            layers = zip_layer(ops=ops)
            self.assertEqual(layers[0].use_bias, True)
            self.assertEqual(layers[1].use_bias, False)


if __name__ == '__main__':
    tf.test.main()
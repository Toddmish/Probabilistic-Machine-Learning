import numpy as np
import tensorflow as tf
import tensorflow_probability as tfpr

"""

KULLBACK-LEIBLER DIVERGENCE: KL(p(x)||q(x))

DEfinition:

KL(p(x)||q(x)) =Ex~p(x) [log( p(x)/q(x)) ] (I)

I.e. it measures the expected excess surprise when
replacing the actual p(x) description with q(x) description

Approximated:

KL(p(x)||q(x)) = 1/L * Σx[log( p(x)/q(x))]   (II)
It converges for large values of L, which means that estimated KL gets close 
to the real value

"""


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def from_keras_kl(px, qx):
    kl = tf.keras.losses.KLDivergence(reduction="auto", name="kl_divergence")
    kl_keras = kl(px, qx).numpy()
    print("TF KERAS ESTIMATED KL( || ) = ", kl_keras)
    return kl_keras


def from_tf_prob_kl(px, qx):
    kl = tfpr.distributions.kl_divergence(px, qx)
    print("TF.PROB ESTIMATED KL( || ) = ", kl)
    return kl


def approximate_kl(px, qx):
    # Approximating KL: KL(p(x)||q(x)) = 1/L * Σx[log( p(x)/q(x))]   (II)
    kl_dive_tensor2 = tf.math.reduce_mean(tf.math.log(px / qx))
    print("APPROXIMATED KL( || ) = ", kl_dive_tensor2)


def simulate_input(arr_len=100):
    # To avoid manual generation of an example intput array we will fill the X array with
    # randomly generated values; we do not care about their distribution we just need a big input X array
    # we generate using Bernoulli distribution or whatever other distributions; we just make up an input
    X = tfpr.distributions.Bernoulli(0.8).sample(arr_len)
    print("INPUT X = ", X)
    return X


def get_input():
    '''
        WE can retrieve data from a file,
        or put them manually,
        or simulate them by generating them randomly using a selected-by-us distribution model
    '''

    # if input contains a small number of points, then approximated KL will return
    # quite different values for different runs. For a large number of points
    # it will converge. Try multiple times to notice it.
    return simulate_input(100000)


if __name__ == '__main__':
    print_hi('PyCharm tries')
    print(__name__)
    print("TensorFlow version:", tf.__version__)
    print("Numpy version:", np.__version__)

    # exam can be classified as passed (1) or failed (0)
    # we simulate some date in absence of real data
    ground_true_exam_results = tfpr.distributions.Bernoulli(0.7)
    predicted_exam_results = tfpr.distributions.Bernoulli(0.6)

    """ Estimating KL(p(x)||q(x)) using tensorflow_probability  """

    """ Estimating KL between two points """
    # KL(ground_true_exam_results || ground_true_exam_results) must be zero
    kl_dive_tensor = from_tf_prob_kl(ground_true_exam_results, ground_true_exam_results)
    print(kl_dive_tensor)

    # KL(predicted_exam_results || predicted_exam_results) must be zero
    kl_dive_tensor = from_tf_prob_kl(predicted_exam_results, predicted_exam_results)
    print(kl_dive_tensor)

    kl_dive_tensor = from_tf_prob_kl(ground_true_exam_results, predicted_exam_results)
    print("CALCULATED KL( || ) = ", kl_dive_tensor)

    """ Estimating KL for an input array X=[0..L]  """

    X = get_input()
    print(X)

    px = ground_true_exam_results.prob(X)
    print("PROBABILITY DISTRIBUTIONS FOR GROUND-TRUE DATA = ", px)

    qx = predicted_exam_results.prob(X)
    print("PROBABILITY DISTRIBUTIONS FOR PREDICTED DATA = ", qx)

    approximate_kl(px, qx)
    #from_keras_kl(px, qx)



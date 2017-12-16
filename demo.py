# Simple example of how to estimate MI between X and Y, where Y = f(X) + Noise(0, noise_variance)
from __future__ import print_function
import kde
import keras.backend as K
import numpy as np

Y_samples = K.placeholder(ndim=2)     

noise_variance = 0.05
entropy_func_upper = K.function([Y_samples,], [kde.entropy_estimator_kl(Y_samples, noise_variance),])
entropy_func_lower = K.function([Y_samples,], [kde.entropy_estimator_bd(Y_samples, noise_variance),])

data = np.random.random( size = (1000, 20) )  # N x dims
H_Y_given_X = kde.kde_condentropy(data, noise_variance)
H_Y_upper = entropy_func_upper([data,])[0]
H_Y_lower = entropy_func_lower([data,])[0]

print("Upper bound: %0.3f nats" % (H_Y_upper - H_Y_given_X))
print("Lower bound: %0.3f nats" % (H_Y_lower - H_Y_given_X))

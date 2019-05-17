import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# change me !
use_estimator = True

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
y_true = 0.5 * x_data + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
my_data = pd.concat([x_df, y_df], axis=1)

batch_size = 8

if not use_estimator:

    my_data.sample(n=250).plot(kind='scatter', x="X Data", y='Y')

    m = tf.Variable(0.3)  # random
    b = tf.Variable(0.1)  # random

    x_ph = tf.placeholder(tf.float32, [batch_size])
    y_ph = tf.placeholder(tf.float32, [batch_size])

    y_model = m*x_ph + b
    error = tf.reduce_sum(tf.square(y_ph-y_model))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        batches = 10000
        for i in range(batches):
            rand_ind = np.random.randint(len(x_data), size=batch_size)
            feed = {x_ph: x_data[rand_ind], y_ph: y_true[rand_ind]}
            sess.run(train, feed_dict=feed)

        model_m, model_b = sess.run([m, b])
    print(model_m, model_b)

    y_hat = x_data*model_m + model_b
    plt.plot(x_data, y_hat, 'r')
    plt.show()

else:

    feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
    x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

    print(x_train)

    input_func = tf.estimator.inputs.numpy_input_fn(
        {'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
    train_input_func = tf.estimator.inputs.numpy_input_fn(
        {'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
    eval_input_func = tf.estimator.inputs.numpy_input_fn(
        {'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)
    estimator.train(input_fn=input_func, steps=1000)

    train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
    eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

    brand_new_data = np.linspace(0, 10, 10)
    input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

    predictions = [pred['predictions'] for pred in estimator.predict(input_fn=input_fn_predict)]
    print(predictions)

    my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(brand_new_data, predictions, 'r*')
    plt.show()

import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet


# sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

# Load traffic signs data.
with open('train.p', mode='rb') as _fh:
    data = pickle.load(_fh)

# Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'])

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(dtype=tf.int64, shape=(None))
y_one_hot = tf.one_hot(y, depth=nb_classes)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
fc8 = tf.layers.dense(fc7, nb_classes, activation=None)
logits = tf.nn.softmax(fc8)

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

y_hat = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float32))

# Train and evaluate the feature extraction model.
epochs = 100
batch_size = 128

nb_steps = (X_train.shape[0] + batch_size - 1) // batch_size

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        train_loss, train_acc = 0, 0

        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, X_train.shape[0], nb_steps):
            end = offset+batch_size
            x_batch = X_train[offset: end]
            y_batch = y_train[offset: end]

            _, cost, acc = sess.run([optimizer, loss, accuracy], feed_dict={
                x: x_batch,
                y: y_batch
            })
            train_loss += (cost * x_batch.shape[0])
            train_acc += (acc * x_batch.shape[0])

        train_loss /= X_train.shape[0]
        train_acc /= y_train.shape[0]
        v_loss, v_acc = sess.run([loss, accuracy], feed_dict={
            x: X_valid,
            y: y_valid
        })
        if epoch % 1 == 0:
            print('Train loss: {}, Train acc: {}, Validn loss: {}, Validn acc: {}'
                  .format(train_loss, train_acc, v_loss, v_acc))

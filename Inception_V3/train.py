from __future__ import absolute_import, division, print_function
from numpy import gradient
import tensorflow as tf
from models import inception_v3
import config
from prepare_data import generate_datasets
import math

def get_model():
    model = inception_v3.InceptionV3(num_class=config.NUM_CLASSES)
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, include_aux_logits=True, training=True)
            loss_aux = loss_object(y_true=labels, y_pred=predictions.aux_logits)
            loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions.logits)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradient, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions.logits)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, include_aux_logits=False, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_state()
        train_accuracy.reset_state()
        valid_loss.reset_state()
        valid_accuracy.reset_state()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print('Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}'.format(epoch+1, config.EPOCHS, step, math.ceil(train_count/config.BATCH_SIZE), train_loss.result(), train_accuracy.result()))
        
        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print('EPOCH: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, valid loss: {:.5f}, valid accuracy: {:.5f}'.format(epoch+1, config.EPOCHS, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()))

    model.save_weights(filepath=config.save_model_dir, save_format='tf')


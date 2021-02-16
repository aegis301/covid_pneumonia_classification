import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

 #boilerplate for gpu usage from tensorflow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
     # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

BATCH_SIZE = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 1
EPOCHS = 1000
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# create data generator for training data with augmentation
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.5,
    rotation_range=90,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    vertical_flip=True
)

# creating test data generator that only performs rescaling and no augmenation
test_data_generator = ImageDataGenerator(
    rescale=1./255
)

train_iterator = train_data_generator.flow_from_directory(
    'data/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    seed=42,
)

val_iterator = test_data_generator.flow_from_directory(
    'data/val',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    )

test_iterator = test_data_generator.flow_from_directory(
    'data/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
)


model = tf.keras.Sequential([
    # Input layers
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    # Convolutional layers
    tf.keras.layers.Conv2D(64, (5, 5), strides=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
    # flatten results to feed to DNN
    tf.keras.layers.Flatten(),
    # # hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=20)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

history = model.fit(
    train_iterator,
    epochs=EPOCHS,
    validation_data=val_iterator,
    callbacks=[es]
)


test_steps_per_epoch = numpy.math.ceil(test_iterator.samples / test_iterator.batch_size)
predictions = model.predict(test_iterator)
test_steps_per_epoch = numpy.math.ceil(test_iterator.samples / test_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = test_iterator.classes
class_labels = list(test_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)


# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()


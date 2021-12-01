import tensorflow as tf



def identity_block(inputs, kernal_size, filters):
    filters1, filters2, filters3 = filters
    x = tf.keras.layers.Conv2D(filters1, (1, 1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, kernal_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.add([x, inputs])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def conv_block(inputs, kernal_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernal_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, padding='same')(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    tf.keras.layers.Activation('relu')(x)

    return x


def resnet50(inputs, classes):
    "开始有7*7的"
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
    x = conv_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    return x


inputs = tf.keras.Input(shape=[32, 32, 3])
model = tf.keras.Model(inputs=inputs, outputs=resnet50(inputs, classes=10))
model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'Recall', 'AUC'])
model.save('restnet50_cifar10')

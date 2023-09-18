import tensorflow as tf
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
import os

batch_size = 8
train_dir = './data/train_data'
IMG_HEIGHT = 48
IMG_WIDTH = 48

# 生成器读取图像
train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           subset='training')
validation_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=train_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical',
                                                              subset='validation')

# 卷积神经网络模型的构建
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), filters=32, kernel_size=3, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_data_gen.class_indices), activation='softmax'))

# 网络编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 模型拟合
epochs = 15
history = model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen)

# 模型保存
if not os.path.exists('./model'):
    os.makedirs('./model')
model.save('./model/model_CNN.h5')

# 模型评估
print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

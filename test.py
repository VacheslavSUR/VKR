# импорт библиотек import tensorflow as tf import keras
import os
from matplotlib import pyplot as plt from IPython import display
from tqdm import tqdm_notebook as tqdm

# инициализация констант IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
EPOCHS = 150

# блок кодирующей части
def downsample(filters, size, apply_batchnorm=True): initializer = tf.random_normal_initializer(0., 0.02) result = tf.keras.Sequential()
result.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same', kernel_initializer=initializer, use_bias=False))
if apply_batchnorm: result.add(tf.keras.layers.BatchNormalization())
result.add(tf.keras.layers.LeakyReLU()) return result

# блок декодирующей части
def upsample(filters, size, apply_dropout=False): initializer = tf.random_normal_initializer(0., 0.02) result = tf.keras.Sequential()
result.add(
tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same', kernel_initializer=initializer,use_bias=False))
result.add(tf.keras.layers.BatchNormalization()) if apply_dropout:
result.add(tf.keras.layers.Dropout(0.5)) result.add(tf.keras.layers.ReLU())
return result

# генератор
def Generator():
inputs = tf.keras.layers.Input(shape=[256,256,3]) down_stack = [
downsample(64, 4, apply_batchnorm=False),#( 128, 128, 64)

downsample(128,	4),	#	(64,	64,	128)
downsample(256,	4),	#	(32,	32,	256)
downsample(512,	4),	#	(16,	16,	512)
 
	downsample(512,	4),	#	(8,	8,	512)
	downsample(512,	4),	#	(4,	4,	512)
	downsample(512,	4),	#	(2,	2,	512)

]	downsample(512,

up_stack = [	4),	#	(1,	1,	512)

	upsample(512,	4,	apply_dropout=True),	#	(2,	2,	1024)
	upsample(512,	4,	apply_dropout=True),	#	(4,	4,	1024)
	upsample(512,	4,	apply_dropout=True),	#	(8,	8,	1024)
	upsample(512,	4),	# (16, 16, 1024)	
	upsample(256,	4),	# (32, 32, 512)	
	upsample(128,	4),	# (64, 64, 256)	
upsample(64, 4), # (128, 128, 128)
]
initializer = tf.random_normal_initializer(0., 0.02)
last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
padding='same', kernel_initializer=initializer, activation='tanh') # (256, 256, 3)
x = inputs skips = []
for down in down_stack: x = down(x) skips.append(x)
skips = reversed(skips[:-1])

for up, skip in zip(up_stack, skips): x = up(x)
x = tf.keras.layers.Concatenate()([x, skip])

x = last(x)
return tf.keras.Model(inputs=inputs, outputs=x)

# загрузка весов обученной модели generator = Generator()
generator.load_weights('/content/drive/My Drive/149/generator.h5')

# загрузка изображения def load_image(path):
test_input_gray = cv2.imread(path)
test_input_gray = cv2.cvtColor(test_input_gray, cv2.COLOR_BGR2GRAY) test_input_gray = cv2.cvtColor(test_input_gray, cv2.COLOR_GRAY2RGB) test_input_gray = tf.cast(test_input_gray, tf.float32) test_input_gray = resize(test_input_gray, 256, 256) test_input_gray = normalize(test_input_gray)
return test_input_gray

# генерация цветного изображения
def generate_images(model, test_input_gray): prediction = model(test_input_gray, training=True) plt.figure(figsize=(15,15))
display_list = [test_input_gray[0], prediction[0]]
 
title = ['Input Image', 'Predicted Image'] for i in range(1):
plt.subplot(1, 3, i+1) plt.title(title[i]) plt.imshow(display_list[i] * 0.5 + 0.5) plt.axis('off')
plt.show()

# список путей тестовых изображений listOfImages = os.listdir('/content/test/')

# тестирование
for i in range(len(listOfImages)):
test_input = load_image(os.path.join('/content/test', listOfImages[i])) generate_images(generator, test_input[tf.newaxis,...])

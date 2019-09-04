import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("discriminator.hdf5")
tflite_model = converter.convert()
open("discriminator.tflite", "wb").write(tflite_model)

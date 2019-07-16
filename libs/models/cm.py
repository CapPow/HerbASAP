import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("mlp_proposal.hdf5")
tflite_model = converter.convert()
open("mlp_proposal.tflite", "wb").write(tflite_model)

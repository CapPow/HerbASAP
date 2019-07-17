import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("mlp_proposal_bn.hdf5")
tflite_model = converter.convert()
open("mlp_proposal_bn.tflite", "wb").write(tflite_model)

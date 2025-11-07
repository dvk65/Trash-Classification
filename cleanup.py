import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)

# Load the old HDF5 model
model = tf.keras.models.load_model("trashclassify.h5")

# Save in the new zipped Keras format
keras.saving.save_model(model, "trashclassify.keras")
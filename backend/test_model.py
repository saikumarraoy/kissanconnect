import keras
from keras import ops
from tensorflow.keras.models import load_model


@keras.saving.register_keras_serializable()
class TrueDivide(keras.Operation):
    def call(self, x, y):
        return ops.divide(x, y)


@keras.saving.register_keras_serializable()
class Subtract(keras.Operation):
    def call(self, x, y):
        return ops.subtract(x, y)


print("Loading model...")

model = load_model(
    "models\\kissanconnect_model.h5",
    compile=False,
    custom_objects={
        "TrueDivide": TrueDivide,
        "Subtract": Subtract,
    },
)

print("MODEL LOADED SUCCESSFULLY")
print("Layers:", len(model.layers))
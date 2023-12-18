import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

class InceptionModel():
  def __init__(self, input_size=150):
     self.base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(input_size,input_size,3)
        )

     self.base_model.trainable = False
     self.inputs = keras.Input(shape=(input_size,input_size,3))
     self.base = self.base_model(self.inputs, training=False)
     self.vectors = keras.layers.GlobalAveragePooling2D()(self.base)
     self.outputs = keras.layers.Dense(3)(self.vectors)
     self.model = keras.Model(self.inputs, self.outputs)

  def build_model(self, args, optimizer, loss):
     self.add_layer(args.model_args[args.model].size) if args.model_args[args.model].inner else None
     self.add_dropout(args.model_args[args.model].droprate, args.model_args[args.model].inner) if args.model_args[args.model].drop else None
     self.model.compile(optimizer=optimizer, loss=loss, metrics=args.metrics)
     return self.model

  def add_layer(self, size=100):
     self.inner = keras.layers.Dense(units=size, activation='relu', name='inner_layer')(self.vectors)
     self.outputs = keras.layers.Dense(3)(self.inner)
     self.model = keras.Model(self.inputs, self.outputs)

  def add_dropout(self, droprate=0.0, inner=True):
     self.drop = keras.layers.Dropout(rate=droprate)(self.inner if inner else self.vectors)
     self.outputs = keras.layers.Dense(3)(self.drop)
     self.model = keras.Model(self.inputs, self.outputs)

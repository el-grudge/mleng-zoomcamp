import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from models.XceptionModel import XceptionModel
from models.InceptionModel import InceptionModel
from models.ResNetModel import ResNetModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input


def load_dataset(args, location, split='train'):
   preprocess_function = {
      'Xception': xception_preprocess_input,
      'Inception': inception_preprocess_input,
      'ResNet': resnet_preprocess_input
      }

   if split != 'train':
    generator = ImageDataGenerator(preprocessing_function = preprocess_function[args.model])
   else:
    generator = ImageDataGenerator(
        preprocessing_function = preprocess_function[args.model],
        rotation_range = args.augmentation_args['rotation_range'] if args.augmentation else 0,
        height_shift_range = args.augmentation_args['height_shift_range'] if args.augmentation else 0.0,
        width_shift_range = args.augmentation_args['width_shift_range'] if args.augmentation else 0.0,
        horizontal_flip = args.augmentation_args['horizontal_flip'] if args.augmentation else False,
        vertical_flip = args.augmentation_args['vertical_flip'] if args.augmentation else False,
        zoom_range = args.augmentation_args['zoom_range'] if args.augmentation else 1.0
        )

   dataset = generator.flow_from_directory(
       location,
       target_size=(args.input_size, args.input_size),
       batch_size=args.batch_size,
       shuffle=False if split != 'train' else args.shuffle
       )

   return dataset


def create_model(args):
  models = {
     'Xception': XceptionModel(args.input_size),
     'Inception': InceptionModel(args.input_size),
     'ResNet': ResNetModel(args.input_size)
  }

  optimizer = keras.optimizers.Adam(learning_rate=args.model_args[args.model].learning_rate)

  loss = keras.losses.CategoricalCrossentropy(from_logits=True)

  model = models[args.model]

  model.build_model(args, optimizer, loss)

  return model


def append_scores(scores_dict, results):
  if not bool(scores_dict):
    scores_dict = results
  else:
    for key in scores_dict.keys():
      scores_dict[key].extend(results[key])

  return scores_dict


def plot_validation_accuracy(df, column):
    # Create a facet wrap plot
    models = df['Model'].unique()

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5), sharey=True)

    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        unique_lr = model_data[column].unique()

        for lr in unique_lr:
            lr_data = model_data[model_data[column] == lr]
            validation_accuracy = lr_data['ValidationAccuracy']
            axes[i].plot(np.arange(len(validation_accuracy)), validation_accuracy, label=f'LR={lr}')

        axes[i].set_title(f'Model {model}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Validation Accuracy')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_best_score(df, column):
    df = df.groupby(['Model', column]).agg({'ValidationAccuracy': 'max'}).unstack()
    df.columns = df.columns.droplevel(0)
    df = df.reset_index()
    df.set_index('Model', inplace=True)
    # Plotting
    ax = df.plot(kind='bar', stacked=False)
    plt.xlabel('Model')  # This sets the x-axis label
    plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
    plt.show()


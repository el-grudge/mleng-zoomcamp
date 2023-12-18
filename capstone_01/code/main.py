from utils import *
from CustomCallback import CustomCallback
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import Namespace

xception_args = Namespace(
    model = 'Xception',
    learning_rate=None,
    inner=False,
    size=None,
    drop=False,
    droprate=None
    )

inception_args = Namespace(
    model = 'Inception',
    learning_rate=None,
    inner=False,
    size=None,
    drop=False,
    droprate=None
    )

resnet_args = Namespace(
    model = 'ResNet',
    learning_rate=None,
    inner=False,
    size=None,
    drop=False,
    droprate=None
    )

args = Namespace(
    model = "",
    model_args = {
        'Xception': xception_args,
        'Inception': inception_args,
        'ResNet': resnet_args,
    },
    epochs=10,
    batch_size=32,
    input_size=150,
    shuffle=False,
    metrics = ['accuracy'],
    augmentation=False,
    augmentation_args = {
        'rotation_range':50,
        'height_shift_range':0.1,
        'width_shift_range':0.1,
        'horizontal_flip':True,
        'vertical_flip':True,
        'zoom_range':0.1
    }
)

path = './where-am-i/Data'
fullpath = f'{path}'

train_location = f'{fullpath}/train'
val_location = f'{fullpath}/Val'


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

architectures = ['Xception', 'Inception', 'ResNet']
scores_dict = {}

# adjusting learning rate
for lr in [0.001, 0.01, 0.1]:
  for architecture in architectures:
    args.model = architecture
    args.model_args[architecture].learning_rate = lr

    train_dataset = load_dataset(args, train_location, 'train')
    val_dataset = load_dataset(args, val_location, 'val')
    model = create_model(args)

    checkpoint_format = f"models/{args.model}_lr_{args.model_args[architecture].learning_rate}_{{epoch:02d}}_{{val_accuracy:.3f}}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_format,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
        )

    custom_callback = CustomCallback(args)
    history = model.model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback, checkpoint])
    scores_dict = append_scores(scores_dict, custom_callback.data)

    print()
    print()

print("the end")

df = pd.DataFrame(scores_dict)
plot_validation_accuracy(df, 'LearningRate')
plot_best_score(df, 'LearningRate')

scores_df = pd.DataFrame(scores_dict)

for architecture in architectures:
  df = scores_df[scores_df['Model'] == architecture]
  args.model_args[architecture].learning_rate = df.loc[df['ValidationAccuracy'].idxmax()].LearningRate

scores_df.to_csv('scores.csv', index=False)

# adding extra layer
for size in [10, 100, 1000]:
  for architecture in architectures:
    args.model = architecture
    args.model_args[architecture].inner = True
    args.model_args[architecture].size = size

    train_dataset = load_dataset(args, train_location, 'train')
    val_dataset = load_dataset(args, val_location, 'val')
    model = create_model(args)
    inner_layer =  [layer.name for layer in model.model.layers if 'inner' in layer.name]
    print(size, model.model.layers[1].name)

    checkpoint_format = f"models/{args.model}_lr_{args.model_args[architecture].learning_rate}_size_{args.model_args[architecture].size}_{{epoch:02d}}_{{val_accuracy:.3f}}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_format,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
        )

    custom_callback = CustomCallback(args)
    history = model.model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback, checkpoint])
    scores_dict = append_scores(scores_dict, custom_callback.data)

    print()
    print()

print("the end")

df = pd.DataFrame(scores_dict)[pd.DataFrame(scores_dict)['ExtraLayer'] != 'NA']
plot_validation_accuracy(df, 'ExtraLayer')
plot_best_score(df, 'ExtraLayer')

scores_df = pd.DataFrame(scores_dict)

for architecture in architectures:
  df = scores_df[(scores_df['Model'] == architecture)]
  inner = df.loc[df['ValidationAccuracy'].idxmax()].ExtraLayer
  args.model_args[architecture].inner = True if inner != 'NA' else False
  args.model_args[architecture].size = inner if inner != 'NA' else 'NA'

scores_df.to_csv('scores.csv', index=False)

# adjusting droprate
for droprate in [.2, .5, .8]:
  for architecture in architectures:
    args.model = architecture
    args.model_args[architecture].drop = True
    args.model_args[architecture].droprate = droprate

    train_dataset = load_dataset(args, train_location, 'train')
    val_dataset = load_dataset(args, val_location, 'val')
    model = create_model(args)
    # inner_layer = [layer.name for layer in model.model.layers if 'inner' in layer.name]
    drop_layer = [layer.name for layer in model.model.layers if 'dropout' in layer.name]
    print(droprate, model.model.layers[1].name)

    checkpoint_format = f"models/{args.model}_lr_{args.model_args[architecture].learning_rate}_size_{args.model_args[architecture].size}_dropout_{args.model_args[architecture].droprate}_{{epoch:02d}}_{{val_accuracy:.3f}}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_format,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
        )

    custom_callback = CustomCallback(args)
    history = model.model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback, checkpoint])
    scores_dict = append_scores(scores_dict, custom_callback.data)

    print()
    print()

print("the end")

df = pd.DataFrame(scores_dict)[pd.DataFrame(scores_dict)['Droprate'] != 'NA']
plot_validation_accuracy(df, 'Droprate')
plot_best_score(df, 'Droprate')

scores_df = pd.DataFrame(scores_dict)

scores_df = pd.DataFrame(scores_dict)

for architecture in architectures:
  df = scores_df[(scores_df['Model'] == architecture)]
  droprate = df.loc[df['ValidationAccuracy'].idxmax()].Droprate
  args.model_args[architecture].drop = True if droprate != 'NA' else False
  args.model_args[architecture].droprate = droprate if droprate != 'NA' else 'NA'

scores_df.to_csv('scores.csv', index=False)


# adding augmentation
for architecture in architectures:
  args.model = architecture
  args.augmentation=True
  args.shuffle=True
  train_dataset = load_dataset(args, train_location, 'train')
  val_dataset = load_dataset(args, val_location, 'val')
  model = create_model(args)

  checkpoint_format = f"models/{args.model}_lr_{args.model_args[architecture].learning_rate}_size_{args.model_args[architecture].size}_dropout_{args.model_args[architecture].droprate}_{{epoch:02d}}_{{val_accuracy:.3f}}_augmented.h5"
  checkpoint = keras.callbacks.ModelCheckpoint(
      checkpoint_format,
      save_best_only=True,
      monitor='val_accuracy',
      mode='max'
      )

  custom_callback = CustomCallback(args)
  history = model.model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback, checkpoint])
  scores_dict = append_scores(scores_dict, custom_callback.data)

  print()
  print()

print("the end")

## plots 
# Create a facet wrap plot
df = pd.DataFrame(scores_dict)[pd.DataFrame(scores_dict)['Augmentation'] != False]
models = df['Model'].unique()

# Plotting
fig, ax = plt.subplots()

# Use a for loop to plot lines for each model
for model in models:
    model_df = df[df['Model'] == model]
    ax.plot(np.arange(10), model_df['ValidationAccuracy'], label=model)

# Adding labels and title
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy Across Epochs for Three Models')

# Adding legend
ax.legend()
plt.show()


# Grouping by 'Model' and finding the maximum validation accuracy for each model
best_performances = df.groupby('Model')['ValidationAccuracy'].max()

# Plotting a bar chart
fig, ax = plt.subplots()
best_performances.plot(kind='bar', ax=ax, color=['orange', 'green', 'blue'])

# Adding labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Best Validation Accuracy for Each Model')

# Display the plot
plt.show()


architectures = list(pd.DataFrame(scores_dict).groupby('Model').\
                     apply(lambda x: x.loc[x['ValidationAccuracy'].idxmax()]).head(2)['Model'])

scores_df.to_csv('scores.csv', index=False)

# training a larger model
args.epochs = 20

for architecture in architectures:
  args.model = architecture
  args.augmentation=True
  args.shuffle=True
  args.input_size=299
  train_dataset = load_dataset(args, train_location, 'train')
  val_dataset = load_dataset(args, val_location, 'val')
  model = create_model(args)

  checkpoint_format = f"models/{args.model}_lr_{args.model_args[architecture].learning_rate}_size_{args.model_args[architecture].size}_dropout_{args.model_args[architecture].droprate}_{{epoch:02d}}_{{val_accuracy:.3f}}_large_model.h5"
  checkpoint = keras.callbacks.ModelCheckpoint(
      checkpoint_format,
      save_best_only=True,
      monitor='val_accuracy',
      mode='max'
      )

  custom_callback = CustomCallback(args)
  history = model.model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[custom_callback, checkpoint])
  scores_dict = append_scores(scores_dict, custom_callback.data)

  print()
  print()

print("the end")

final_scores_df = pd.DataFrame(scores_dict)[pd.DataFrame(scores_dict)['ModelSize'] == 299]

# Create a facet wrap plot
models = final_scores_df['Model'].unique()

# Plotting
fig, ax = plt.subplots()

# Use a for loop to plot lines for each model
for model in models:
    model_df = df[df['Model'] == model]
    ax.plot(np.arange(10), model_df['ValidationAccuracy'], label=model)

# Adding labels and title
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy Across Epochs for Three Models')

# Adding legend
ax.legend()
plt.show()

# Grouping by 'Model' and finding the maximum validation accuracy for each model
best_performances = final_scores_df.groupby('Model')['ValidationAccuracy'].max()

# Plotting a bar chart
fig, ax = plt.subplots()
best_performances.plot(kind='bar', ax=ax, color=['blue', 'orange'])

# Adding labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Best Validation Accuracy for Each Model')

# Display the plot
plt.show()


scores_df = pd.DataFrame(scores_dict)
print(scores_df[scores_df.ValidationAccuracy == scores_df.ValidationAccuracy.max()])

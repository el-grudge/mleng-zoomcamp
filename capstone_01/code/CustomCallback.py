from tensorflow.keras.callbacks import Callback, ModelCheckpoint

# Custom callback to collect training information
class CustomCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.data = {
            'Model': [],
            'LearningRate': [],
            'ExtraLayer': [],
            'Droprate': [],
            'Augmentation': [],
            'ModelSize': [],
            'ValidationAccuracy': []
            }
        self.model_architecture = args.model # Assuming args.model contains the model architecture
        self.inner = args.model_args[args.model].inner
        self.drop = args.model_args[args.model].drop
        self.augmentation = args.augmentation
        self.input_size = args.input_size

    def on_epoch_end(self, epoch, logs):
        # Extract information from the model or training logs
        model_architecture = self.model_architecture # Assuming args.model contains the model architecture
        learning_rate = self.model.optimizer.lr.numpy() if hasattr(self.model.optimizer, 'lr') else 'NA'
        extra_layer = [layer for layer in self.model.layers if 'inner' in layer.name][0].output_shape[1] if self.inner else 'NA'
        droprate =  [layer for layer in self.model.layers if 'drop' in layer.name][0].rate if self.drop else 'NA'
        augmentation = self.augmentation
        input_size = self.input_size
        val_accuracy = logs.get('val_accuracy', 'NA')  # You need to replace this with the actual value

        # Append the information to the data dictionary
        self.data['Model'].append(model_architecture)
        self.data['LearningRate'].append(learning_rate)
        self.data['ExtraLayer'].append(extra_layer)
        self.data['Droprate'].append(droprate)
        self.data['Augmentation'].append(augmentation)
        self.data['ModelSize'].append(input_size)
        self.data['ValidationAccuracy'].append(val_accuracy)

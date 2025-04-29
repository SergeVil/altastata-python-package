import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from altastata import AltaStataTensorFlowDataset
import altastata_config
import keras

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def get_file_extension(path):
    """Extract the file extension from a path, handling AltaStata paths."""
    # If the path has a version suffix (✹), extract just the file part
    if '✹' in str(path):
        # Extract the file part before the version marker
        file_part = str(path).split('✹')[0]
        # Get the extension
        _, ext = os.path.splitext(file_part)
        return ext.lower()
    else:
        # Standard path handling
        _, ext = os.path.splitext(str(path))
        return ext.lower()

@keras.saving.register_keras_serializable(package="CustomLayers")
class EdgeDetectionLayer(tf.keras.layers.Layer):
    """Custom layer for edge detection using Sobel filters."""
    def __init__(self, **kwargs):
        super(EdgeDetectionLayer, self).__init__(**kwargs)
        # Initialize Sobel filters
        self.sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        self.sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        # Reshape for conv2d
        self.sobel_x = tf.reshape(self.sobel_x, [3, 3, 1, 1])
        self.sobel_y = tf.reshape(self.sobel_y, [3, 3, 1, 1])

    def call(self, inputs):
        # Split input into channels
        channels = tf.split(inputs, 3, axis=-1)
        edges = []
        
        for channel in channels:
            # Apply Sobel filters
            edge_x = tf.nn.conv2d(channel, self.sobel_x, strides=[1,1,1,1], padding='SAME')
            edge_y = tf.nn.conv2d(channel, self.sobel_y, strides=[1,1,1,1], padding='SAME')
            edge_mag = tf.sqrt(tf.square(edge_x) + tf.square(edge_y))
            edges.append(edge_mag)
        
        edge_features = tf.concat(edges, axis=-1)
        return tf.concat([inputs, edge_features], axis=-1)

    def get_config(self):
        config = super(EdgeDetectionLayer, self).get_config()
        return config

def create_model():
    """Create a CNN model optimized for shape detection."""
    inputs = tf.keras.Input(shape=(96, 96, 3))
    
    # Add edge detection features
    x = EdgeDetectionLayer()(inputs)
    
    # First conv block
    x = tf.keras.layers.Conv2D(32, (5, 5), 
                              activation='relu', 
                              padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Second conv block
    x = tf.keras.layers.Conv2D(64, (3, 3), 
                              activation='relu',
                              padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Third conv block
    x = tf.keras.layers.Conv2D(128, (3, 3), 
                              activation='relu',
                              padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(128, 
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Final classification layer with temperature scaling
    logits = tf.keras.layers.Dense(2)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / 0.5)  # Temperature scaling
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def preprocess_image(image, label):
    """Preprocess images for training."""
    def _preprocess(img, lbl):
        # Convert to float32 and normalize
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize to 96x96
        img = tf.image.resize(img, [96, 96])
        # Convert label to int32
        lbl = tf.cast(lbl, tf.int32)
        return img, lbl
    
    # Wrap the preprocessing function
    image, label = tf.py_function(
        _preprocess,
        [image, label],
        [tf.float32, tf.int32]
    )
    
    # Set shapes explicitly after preprocessing
    image.set_shape([96, 96, 3])
    label.set_shape([])
    
    return image, label

def augment_image(image, label):
    """Apply enhanced data augmentation to images."""
    def _augment(img, lbl):
        # Random rotation
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        
        # Random flip
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        
        # Random brightness, contrast and saturation
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
        # Random zoom
        scale = tf.random.uniform([], 0.8, 1.2)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = tf.image.resize_with_crop_or_pad(img, 96, 96)
        
        # Ensure values are in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        # Ensure label stays as int32
        lbl = tf.cast(lbl, tf.int32)
        return img, lbl
    
    # Wrap the augmentation function
    image, label = tf.py_function(
        _augment,
        [image, label],
        [tf.float32, tf.int32]
    )
    
    # Set shapes explicitly after augmentation
    image.set_shape([96, 96, 3])
    label.set_shape([])
    
    return image, label

class AltaStataModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom callback to save model checkpoints to AltaStata."""
    def __init__(self, dataset, filepath, monitor='val_loss', save_best_only=True):
        super().__init__()
        self.dataset = dataset
        self.filepath = filepath
        self.checkpoint_dir = os.path.join(os.path.dirname(filepath), 'checkpoints')
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if current < self.best:
                self.best = current
                # Save checkpoint with epoch and metric info
                checkpoint_name = f'model_epoch_{epoch:03d}_loss_{current:.4f}.keras'
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
                self.dataset.save_model(self.model, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
        else:
            # Save checkpoint with epoch info
            checkpoint_name = f'model_epoch_{epoch:03d}.keras'
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.dataset.save_model(self.model, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")

def main():
    # Create training dataset
    train_dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account
        root_dir="data_tensorflow/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Create validation dataset
    val_dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account
        root_dir="data_tensorflow/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total files: {len(train_dataset)}")
    circle_count = sum(1 for path in train_dataset.file_paths if 'circle' in str(path))
    rectangle_count = sum(1 for path in train_dataset.file_paths if 'rectangle' in str(path))
    print(f"Circle images: {circle_count}")
    print(f"Rectangle images: {rectangle_count}\n")
    
    # Create data indices for training and validation splits
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.3 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create training dataset with augmentation
    train_ds = train_dataset.take(len(train_indices))
    
    # Debug: Print shapes before augmentation
    for image, label in train_ds.take(1):
        print("Shape before augmentation:", image.shape, label.shape)
    
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Debug: Print shapes after augmentation
    for image, label in train_ds.take(1):
        print("Shape after augmentation:", image.shape, label.shape)
    
    train_ds = train_ds.batch(8).prefetch(tf.data.AUTOTUNE)
    
    # Debug: Print shapes after batching
    for image, label in train_ds.take(1):
        print("Shape after batching:", image.shape, label.shape)
    
    # Create validation dataset with explicit shape setting
    val_ds = val_dataset.take(len(val_indices))
    val_ds = val_ds.map(
        lambda x, y: (
            tf.ensure_shape(x, [96, 96, 3]),
            tf.ensure_shape(y, [])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(8).prefetch(tf.data.AUTOTUNE)
    
    # Debug: Print validation shapes
    for image, label in val_ds.take(1):
        print("Validation shape after batching:", image.shape, label.shape)
    
    # Create and compile model with focal loss
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        AltaStataModelCheckpoint(
            dataset=train_dataset,
            filepath='data_tensorflow/models/best_model.keras',
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model with class weights
    class_weights = {
        0: 1.0,  # for rectangles
        1: len(train_dataset.file_paths) / (2 * circle_count)  # adjust weight for circles
    }
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    print("\nTraining completed!")
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    for metric, value in history.history.items():
        print(f"{metric}: {value[-1]:.4f}")
    
    # Save the final best model in the main models directory
    model_save_path = 'data_tensorflow/models/best_model.keras'
    print(f"\nSaving final model to AltaStata: {model_save_path}")
    train_dataset.save_model(model, model_save_path)
    print("Model saved successfully")

if __name__ == '__main__':
    main() 
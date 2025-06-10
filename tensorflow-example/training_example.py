import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from altastata import AltaStataTensorFlowDataset
import altastata_config

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_model():
    """Create a simplified, ultra-fast CNN model for shape detection."""
    inputs = tf.keras.Input(shape=(96, 96, 3))
    
    # Ultra-simplified architecture: just 2 conv layers
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Replace dense layers with GAP
    
    # Single output layer
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)  # Force float32 for output
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def preprocess_image(image, label):
    """Ultra-simple preprocessing for speed."""
    # Convert to float32 and normalize
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize to 96x96
    image = tf.image.resize(image, [96, 96])
    # Convert label to int32
    label = tf.cast(label, tf.int32)
    return image, label

def augment_image(image, label):
    """Minimal augmentation for speed - just horizontal flip."""
    image = tf.image.random_flip_left_right(image)
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
        root_dir="tensorflow_test/data/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Create validation dataset
    val_dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account
        root_dir="tensorflow_test/data/images",
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
    
    # Create ultra-fast training pipeline
    train_ds = train_dataset.take(len(train_indices))
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()  # Cache in memory for speed
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)  # Larger batch size
    
    # Create validation dataset
    val_ds = val_dataset.take(len(val_indices))
    val_ds = val_ds.cache()  # Cache validation data too
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # Create ultra-simple model optimized for speed
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),  # Higher learning rate for faster convergence
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Ultra-aggressive callbacks for fast training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,  # Very aggressive early stopping
            restore_best_weights=True,
            min_delta=0.01,
            mode='max'
        ),
        AltaStataModelCheckpoint(
            dataset=train_dataset,
            filepath='tensorflow_test/model/best_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    print("\n🚀 Starting ultra-fast training...")
    start_time = tf.timestamp()
    
    # Train with minimal epochs for speed
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # Reduced epochs for speed
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = tf.timestamp()
    training_time = end_time - start_time
    print(f"\n⚡ Ultra-fast training completed in {training_time:.1f} seconds!")
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    for metric, value in history.history.items():
        print(f"{metric}: {value[-1]:.4f}")
    
    # Save the final best model
    model_save_path = 'tensorflow_test/model/best_model.keras'
    print(f"\nSaving ultra-fast model to AltaStata: {model_save_path}")
    train_dataset.save_model(model, model_save_path)
    
    # Model size info
    print(f"\nModel info:")
    print(f"Total parameters: {model.count_params():,}")
    print("Model saved successfully - ready for ultra-fast inference! 🚀")
    print(f"Provenance file saved: {model_save_path}.provenance.txt with {len(train_dataset.file_paths)} file paths")

if __name__ == '__main__':
    main() 
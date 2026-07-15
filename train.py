"""
Trains the forest-fire risk CNN on a labeled image dataset.

Expected folder layout (Keras image_dataset_from_directory convention):

    data/dataset/train/no_fire_risk/*.jpg
    data/dataset/train/fire_risk/*.jpg
    data/dataset/val/no_fire_risk/*.jpg
    data/dataset/val/fire_risk/*.jpg

Good public sources to seed this with:
  - Kaggle "Wildfire Prediction Dataset (Satellite Images)" (Canada, Sentinel-2)
  - Kaggle "FIRE Dataset" / "Forest Fire Images"
  - Your own crops pulled via fetch_live_satellite.py, labeled using
    FIRMS hotspot overlap as a weak label.

Run:  python train.py
"""

import os
import tensorflow as tf

import config
from model import build_cnn, build_transfer_model


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary" if not config.USE_MULTICLASS else "int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary" if not config.USE_MULTICLASS else "int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def main(use_transfer_learning=False):
    os.makedirs(config.MODEL_OUT_DIR, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()

    model = (
        build_transfer_model()
        if use_transfer_learning
        else build_cnn()
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_auc" if not config.USE_MULTICLASS else "val_accuracy",
            mode="max", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    model.save(config.MODEL_PATH)
    print(f"\nSaved best model to {config.MODEL_PATH}")

    # Quick eval summary
    val_metrics = model.evaluate(val_ds, return_dict=True)
    print("Final validation metrics:", val_metrics)

    return history


if __name__ == "__main__":
    main(use_transfer_learning=False)

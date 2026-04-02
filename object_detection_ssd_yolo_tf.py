import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# =========================
# SSD
# =========================

# Step 1: Define a simple backbone network for SSD
def build_backbone(input_shape=(300, 300, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    base_model = models.Model(inputs, x, name="SSD_Backbone")
    return base_model

# Step 2: Define SSD prediction head for class and bounding box predictions
def ssd_head(x, num_classes, num_boxes):
    class_head = layers.Conv2D(num_boxes * num_classes, (3, 3), padding="same")(x)
    class_head = layers.Reshape((-1, num_classes))(class_head)

    box_head = layers.Conv2D(num_boxes * 4, (3, 3), padding="same")(x)  # 4 values for (x, y, w, h)
    box_head = layers.Reshape((-1, 4))(box_head)

    return class_head, box_head

# Step 3: Build the SSD model combining backbone, SSD heads, and multi-scale feature maps
def build_ssd_model(input_shape=(300, 300, 3), num_classes=21, num_boxes=6):
    # Backbone network
    base = build_backbone(input_shape)
    x = base.output  # Starting feature map from the backbone

    # Multi-scale feature maps for detection heads
    class_outputs, box_outputs = [], []
    for i in range(3):  # 3 scales for simplicity
        class_head, box_head = ssd_head(x, num_classes, num_boxes)
        class_outputs.append(class_head)
        box_outputs.append(box_head)

        # Downsample the feature map for next scale
        x = layers.Conv2D(128, (3, 3), strides=2, padding="same")(x)

    # Concatenate outputs across all scales
    classes_concat = layers.Concatenate(axis=1)(class_outputs)
    boxes_concat = layers.Concatenate(axis=1)(box_outputs)

    # Define the final SSD model
    ssd_model = models.Model(inputs=base.input, outputs=[classes_concat, boxes_concat])
    return ssd_model

# Instantiate SSD model
ssd_model = build_ssd_model()
ssd_model.summary()

# Step 4: Generate random dummy data and test the model
ssd_input = np.random.random((1, 300, 300, 3))
ssd_class_pred, ssd_box_pred = ssd_model.predict(ssd_input)

# Print shapes of predictions to confirm the output
print("\nSSD Class Prediction Shape:", ssd_class_pred.shape)
print("SSD Box Prediction Shape:", ssd_box_pred.shape)


# =========================
# YOLO
# =========================

# Step 1: Define a simple backbone for YOLO
def build_yolo_backbone(input_shape=(416, 416, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    return models.Model(inputs, x, name="YOLO_Backbone")

# Step 2: Define YOLO head (inferred from tutorial text)
def yolo_head(x, num_classes=20, num_boxes=3):
    # output per box = 4 bbox coords + 1 confidence + num_classes
    output_channels = num_boxes * (5 + num_classes)
    output = layers.Conv2D(output_channels, (1, 1), padding="same")(x)
    return output

# Step 3: Combine backbone and YOLO head to build YOLO model
def build_yolo_model(input_shape=(416, 416, 3), num_classes=20, num_boxes=3):
    backbone = build_yolo_backbone(input_shape)
    x = backbone.output

    # YOLO head for detection
    output = yolo_head(x, num_classes, num_boxes)

    # Define the YOLO model
    yolo_model = models.Model(inputs=backbone.input, outputs=output)
    return yolo_model

# Instantiate YOLO model
yolo_model = build_yolo_model()
yolo_model.summary()

# Step 4: Generate dummy data for testing
# Generate a random dummy input image of shape (1, 416, 416, 3)
dummy_input = np.random.random((1, 416, 416, 3))

# Run a prediction on the dummy input
yolo_pred = yolo_model.predict(dummy_input)

# Print the shape of the output to verify
print("YOLO Prediction Shape:", yolo_pred.shape)

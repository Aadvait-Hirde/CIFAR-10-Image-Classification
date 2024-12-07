import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
Loads and returns the CIFAR-10 dataset, splitting it into training and test sets.
Returns: x_train, y_train, x_test, y_test arrays
"""
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_train, y_train.squeeze(), x_test, y_test.squeeze()

"""
Creates, compiles and trains a CNN model for CIFAR-10 classification.
Implements early stopping and learning rate reduction callbacks.
Returns: trained model and training history
"""
def train_cifar10_model(x_train, y_train, x_test, y_test):
    class EarlyStoppingCallback(tf.keras.callbacks.Callback):
        def __init__(self, patience=3):
            super().__init__()
            self.patience = patience
            self.best_val_loss = float('inf')
            self.wait = 0
            
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nValidation loss hasn't improved for {self.patience} epochs. Stopping training!")
                    self.model.stop_training = True

    callbacks = [
        EarlyStoppingCallback(patience=3),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]
    
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    # Modify the model to include data augmentation as the first layer
    model = tf.keras.models.Sequential([
        data_augmentation,
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    print(model.summary())

    history = model.fit(
        x_train, y_train,
        epochs=30,  # Increased epochs since we're using augmentation
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

"""
Makes a prediction on a single image using the trained model.
Returns: predicted class label and confidence score
"""
def predict_class(model, img):
    img_array = np.array([img])
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

"""
Callback function for mouse events in OpenCV window.
Toggles the inference state when left mouse button is clicked.
"""
def toggle_inference(event, x, y, flags, params):
    global start_inference
    if event == cv2.EVENT_LBUTTONDOWN:
        start_inference = not start_inference

"""
Runs the real-time video classification loop using OpenCV.
Captures video from webcam and performs inference on frames when enabled.
"""
def run_opencv_loop(model):
    global start_inference
    start_inference = False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('CIFAR-10 Classification')
    cv2.setMouseCallback('CIFAR-10 Classification', toggle_inference)
    frame_count = 0
    processed_view_visible = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if start_inference:
            processed_view_visible = True
            frame_count += 1

            height, width = frame.shape[:2]
            box_size = 200
            center_x, center_y = width // 2, height // 2
            x1, y1 = center_x - box_size // 2, center_y - box_size // 2
            x2, y2 = center_x + box_size // 2, center_y + box_size // 2
            
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (32, 32))
            
            predicted_class, confidence = predict_class(model, roi_resized)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('CIFAR-10 Classification', frame)
            cv2.imshow('Processed View', roi)
        else:
            cv2.putText(frame, "Click anywhere to start recognition", 
                       (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow('CIFAR-10 Classification', frame)
            if processed_view_visible:
                if cv2.getWindowProperty('Processed View', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Processed View')
                    processed_view_visible = False

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
Main function that orchestrates the program flow.
Loads or trains the model, generates evaluation plots,
and starts the real-time classification system.
"""
def main():
    model = None
    try:
        model = tf.keras.models.load_model('cifar10_model.sav')
        print("Loaded saved model.")
        
        _, _, x_test, y_test = load_cifar10_data()
        x_test = x_test / 255.0
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(2, 3, 2)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.bar(CLASSES, class_accuracy)
        plt.title('Per-class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        
        plt.subplot(2, 3, 3)
        y_test_bin = label_binarize(y_test, classes=range(10))
        y_pred_proba = model.predict(x_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'{CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(2, 3, 4)
        for i in range(10):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'{CLASSES[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(2, 3, 5)
        confidences = np.max(y_pred_proba, axis=1)
        plt.hist(confidences, bins=50)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Number of Samples')
        plt.title('Model Confidence Distribution')
        
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
        
    except:
        print("Training new model...")
        x_train, y_train, x_test, y_test = load_cifar10_data()
        model, history = train_cifar10_model(x_train, y_train, x_test, y_test)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
        
        plt.tight_layout()
        plt.show()
        
        print("Saving model...")
        model.save('cifar10_model.sav')

    print("Launching OpenCV...")
    run_opencv_loop(model)

if __name__ == '__main__':
    main() 
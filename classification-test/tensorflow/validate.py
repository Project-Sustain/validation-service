import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# Make prediction on test data
predictions = model.predict(test_data)

# Convert the predictions to binary labels
threshold = 0.5
predictions = (predictions > threshold).astype(int)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)
print("Confusion matrix")
print(conf_matrix)

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_labels, predictions)
auc = roc_auc_score(test_labels, predictions)
print(f'AUC: {auc:.4f}')

# Sensitivity and Specificity calculation
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')

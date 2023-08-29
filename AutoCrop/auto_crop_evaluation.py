'''
This test is integrated to the main pipeline of this project
'''''
import os
import numpy as np
from PIL import Image
from Alignment.utility_params import Params, get_params_path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perf_measure(y_actual, y_hat):
    '''
    Calculate performance measures (True Positives, False Positives, True Negatives, False Negatives) 
    for binary classification based on predicted labels and actual labels.
    :param y_actual: np array: The actual binary labels.
    :param y_hat: np array: The predicted binary labels.
    :return: tuple: A tuple containing the number of True Positives, False Positives, True Negatives, and False Negatives.
    '''''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)


def remove_ds_store(folder_path):
    '''
    Remove the .DS_Store file from a specified folder, if it exists.
    :param folder_path: str: The path to the folder from which to remove the .DS_Store file.
    :return: None
    '''''
    ds_store_path = os.path.join(folder_path, '.DS_Store')
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)


def load_and_preprocess_images(predicted_folder, truth_folder):
    '''
    Calculate performance measures (True Positives, False Positives, True Negatives, False Negatives) 
    for binary classification based on predicted labels and actual labels.
    :param y_actual: np array: The actual binary labels.
    :param y_hat: np array: The predicted binary labels.
    :return: tuple: A tuple containing the number of True Positives, False Positives, True Negatives, and False Negatives.
    '''''
    predicted_images = []
    true_images = []

    for filename in os.listdir(predicted_folder):
        if filename.endswith(".png"):  # Change this to match your image file format
            predicted_path = os.path.join(predicted_folder, filename)
            truth_path = os.path.join(truth_folder, filename)

            # Load and preprocess images
            predicted_image = np.array(Image.open(predicted_path).convert('L'))  # Convert to grayscale
            true_image = np.array(Image.open(truth_path).convert('L'))  # Convert to grayscale

            # Convert images to binary labels
            predicted_label = (predicted_image > 0).astype(int)
            true_label = (true_image > 0).astype(int)

            predicted_images.append(predicted_label)
            true_images.append(true_label)

    return predicted_images, true_images


def plot_conf_matrix(conf_matrix, iou):
    '''
    Plot the confusion matrix as a heatmap and display the Intersection over Union (IoU).
    :param conf_matrix: np array: Confusion matrix.
    :param iou: float: Intersection over Union (IoU) value.
    :return: None
    '''''
    # Normalize the confusion matrix to show percentages
    # conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Create a heatmap using seaborn
    plt.figure()
    ax = sns.heatmap(conf_matrix, annot=True, cmap="Blues",
                xticklabels=["Class 1", "Class 0"],
                yticklabels=["Class 1", "Class 0"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Move x-axis ticks and labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title("Confusion Matrix")
    # plt.suptitle("IoU: " + str(iou), fontweight='bold', ha='center')
    plt.show()
    return



if __name__ == '__main__':
    # Initialize params variable to call parameters in params.json
    params = Params(get_params_path())

    # Get the current .py file's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the path of the project
    project_path = os.path.dirname(script_directory)

    # Construct the path
    predicted_folder_path = str(project_path) + params.auto_crop_eva_predicted_path
    truth_folder_path = str(project_path) + params.auto_crop_eva_ground_truth_path
    confusion_m_path = str(project_path) + params.auto_crop_eva_confusion_m_path

    # Remove .DS_Store file if any
    remove_ds_store(predicted_folder_path)
    remove_ds_store(truth_folder_path)
    predicted_images, true_images = load_and_preprocess_images(predicted_folder_path, truth_folder_path)

    # Flatten images and convert to 1D arrays
    predicted_labels = np.concatenate(predicted_images).ravel()
    true_labels = np.concatenate(true_images).ravel()

    # Generate confusion matrix
    # Predicted Class 1, 0
    # [[a,b]  # Actual Class 1
    #  [c,d]]  # Actual Class 0
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[1,0])
    (TP, FP, TN, FN) = perf_measure(true_labels, predicted_labels)
    print((TP, FP, TN, FN))

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    iou = jaccard_score(true_labels, predicted_labels)

    # Print confusion matrix and metrics
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("IoU:", iou)

    # Plot the confusion matrix
    plot_conf_matrix(conf_matrix, iou)


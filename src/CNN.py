import os
import argparse
import itertools
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def import_dataset(data_dir):
    # Importing the augmented training dataset and testing dataset to create tensors of images using the filename paths.
    train_aug_path = os.path.join(data_dir, "images_filenames_labels_train_augmented.csv")
    test_path = os.path.join(data_dir, "images_filenames_labels_test.csv")

    # Verifica se tem os arquivos ou se o usuário fez merda
    if not os.path.exists(train_aug_path):
        print(f"\n[ERRO] Arquivo de treino não encontrado: {train_aug_path}")
        print("Verifique se o caminho passado no parâmetro '--data_dir' está correto e se o arquivo existe.")
        sys.exit(1) # O 1 indica para o sistema operacional que o script terminou com erro
        
    if not os.path.exists(test_path):
        print(f"\n[ERRO] Arquivo de teste não encontrado: {test_path}")
        print("Verifique se o caminho passado no parâmetro '--data_dir' está correto e se o arquivo existe.")
        sys.exit(1)

    train_aug_df = pd.read_csv(train_aug_path)
    test_df = pd.read_csv(test_path)

    local_path = f"{data_dir}/content/"
    train_aug_df['filename'] = train_aug_df['filename'].str.replace('/content/content/', local_path, regex=False)
    test_df['filename'] = test_df['filename'].str.replace('/content/content/', local_path, regex=False)

    return train_aug_df, test_df

# Defining a function to return the class labels corresponding to the re-distributed 7 age-ranges.
def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6

def map_class_labels(df):
    df['target'] = df['age'].map(class_labels_reassign)
    return df

def _parse_function(filename, label, num_classes):
    # Defining a function to read the image, decode the image from given tensor and one-hot encode the image label class.
    # Changing the channels para in tf.io.decode_jpeg from 3 to 1 changes the output images from RGB coloured to grayscale.
    
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
    label = tf.one_hot(label, num_classes)

    return image_decoded, label

# def get_train_datasets(filenames_tensor, labels_tensor, num_classes, batch_size):
#     train_aug_dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
#     train_aug_dataset = train_aug_dataset.map(lambda x, y: _parse_function(x, y, num_classes))
#     train_aug_dataset = train_aug_dataset.batch(batch_size)    # Same as batch_size hyperparameter in model.fit() below.
#     return train_aug_dataset

# def get_test_datasets(filenames_tensor, labels_tensor, num_classes, batch_size):
#     test_dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
#     test_dataset = test_dataset.map(lambda x, y: _parse_function(x, y, num_classes))
#     test_dataset = test_dataset.batch(batch_size)    # Same as batch_size hyperparameter in model.fit() below.
#     return test_dataset

def get_train_datasets(filenames_tensor, labels_tensor, num_classes, batch_size):
    train_aug_dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    
    train_aug_dataset = train_aug_dataset.map(lambda x, y: _parse_function(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    train_aug_dataset = train_aug_dataset.batch(batch_size)
    train_aug_dataset = train_aug_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_aug_dataset

def get_test_datasets(filenames_tensor, labels_tensor, num_classes, batch_size):
    test_dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    
    test_dataset = test_dataset.map(lambda x, y: _parse_function(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return test_dataset

def define_cnn(num_classes, conv_filters, dense_nodes):
    # Defining the architecture of the sequential neural network.
    final_cnn = Sequential()

    # Cria as camadas dinamicamente baseado na lista fornecida no terminal
    for i, filters in enumerate(conv_filters):
        if i == 0:
            # Input layer, followed by an AveragePooling2D layer.
            final_cnn.add(Conv2D(filters=filters, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    # 3rd dim = 1 for grayscale images.
        else:
            # Conv2D layers for every successive iteration.
            final_cnn.add(Conv2D(filters=filters, kernel_size=3, activation='relu'))
        
        final_cnn.add(AveragePooling2D(pool_size=(2,2)))

    # A GlobalAveragePooling2D layer before going into Dense layers below.
    final_cnn.add(GlobalAveragePooling2D())

    # One Dense layer so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below.
    final_cnn.add(Dense(dense_nodes, activation='relu'))

    # Output layer (equal to the no. of classes).
    final_cnn.add(Dense(num_classes, activation='softmax'))

    return final_cnn

# def define_cnn(num_classes):
#     # Defining the architecture of the sequential neural network.
#     final_cnn = Sequential()

#     # Input layer with 32 filters, followed by an AveragePooling2D layer.
#     final_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    # 3rd dim = 1 for grayscale images.
#     final_cnn.add(AveragePooling2D(pool_size=(2,2)))

#     # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
#     final_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
#     final_cnn.add(AveragePooling2D(pool_size=(2,2)))

#     final_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
#     final_cnn.add(AveragePooling2D(pool_size=(2,2)))

#     final_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
#     final_cnn.add(AveragePooling2D(pool_size=(2,2)))

#     # A GlobalAveragePooling2D layer before going into Dense layers below.
#     # GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
#     final_cnn.add(GlobalAveragePooling2D())

#     # One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below (7).
#     final_cnn.add(Dense(132, activation='relu'))

#     # Output layer with 7 nodes (equal to the no. of classes).
#     final_cnn.add(Dense(num_classes, activation='softmax'))

#     return final_cnn

def check_final_cnn_history(final_cnn_history):
    # Checking the train and test loss and accuracy values from the neural network above.
    train_loss = final_cnn_history.history['loss']
    test_loss = final_cnn_history.history['val_loss']
    train_accuracy = final_cnn_history.history['accuracy']
    test_accuracy = final_cnn_history.history['val_accuracy']

    return train_loss, test_loss, train_accuracy, test_accuracy

def plot_training_history(history, save_dir):
    # Plotting a line chart to visualize the loss and accuracy values by epochs.
    fig, ax = plt.subplots(ncols=2, figsize=(15,7))

    ax = ax.ravel()

    ax[0].plot(history.history['loss'], label='Train Loss', color='royalblue', marker='o', markersize=5)
    ax[0].plot(history.history['val_loss'], label='Test Loss', color = 'orangered', marker='o', markersize=5)

    ax[0].set_xlabel('Epochs', fontsize=14)
    ax[0].set_ylabel('Categorical Crossentropy', fontsize=14)

    ax[0].legend(fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)

    ax[1].plot(history.history['accuracy'], label='Train Accuracy', color='royalblue', marker='o', markersize=5)
    ax[1].plot(history.history['val_accuracy'], label='Test Accuracy', color='orangered', marker='o', markersize=5)

    ax[1].set_xlabel('Epochs', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)

    ax[1].legend(fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)

    fig.suptitle(x=0.5, y=0.92, t="Lineplots showing loss and accuracy of CNN model by epochs", fontsize=16)

    # Exporting plot image in PNG format.
    plt.savefig(os.path.join(save_dir, 'final_cnn_loss_accuracy.png'), bbox_inches='tight')

def evaluate_model(model, test_dataset):
    # Evaluating the model on test dataset.
    final_cnn_score = model.evaluate(test_dataset, verbose=1)

    # Printing the relevant score summary.
    final_cnn_labels = model.metrics_names
    print(f'CNN model {final_cnn_labels[0]} \t\t= {round(final_cnn_score[0], 3)}')
    print(f'CNN model {final_cnn_labels[1]} \t= {round(final_cnn_score[1], 3)}')

    return final_cnn_score

def generate_predictions(model, test_dataset):
    # Generating predictions from the model above.
    final_cnn_pred = model.predict(test_dataset)
    final_cnn_pred = final_cnn_pred.argmax(axis=-1)

    return final_cnn_pred

def generate_confusion_matrix(test_labels_list, final_cnn_pred):
    # Generating a confusion matrix based on above predictions.
    conf_mat = confusion_matrix(test_labels_list, final_cnn_pred)
    return conf_mat

def plot_confusion_matrix(cm, classes, save_dir, normalize=False, title='Confusion Matrix', export_as='confusion_matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)

    # Exporting plot image in PNG format.
    plt.savefig(os.path.join(save_dir, f'{export_as}.png'), bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="Script para treinamento e avaliação de modelo CNN.")
    parser.add_argument('--data_dir', type=str, default='.', help="Pasta contendo os arquivos CSV de treino e teste.")
    parser.add_argument('--save_dir', type=str, default='./cnn_logs', help="Pasta onde os logs, imagens e o modelo serão salvos.")
    parser.add_argument('--model_name', type=str, default='age_model', help="Nome principal para salvar os checkpoints e o modelo.")
    parser.add_argument('--batch_size', type=int, default=512, help="Tamanho do batch para treinamento.")
    parser.add_argument('--epochs', type=int, default=60, help="Número de épocas (epochs).")
    parser.add_argument('--conv_filters', nargs='+', type=int, default=[32, 64, 128, 256], help="Lista de filtros para cada camada convolucional separada por espaço.")
    parser.add_argument('--dense_nodes', type=int, default=132, help="Número de neurónios na camada Dense antes da saída.")
    args = parser.parse_args()

    # Criação do diretório de salvamento, caso não exista
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Setting random seeds to reduce the amount of randomness in the neural net weights and results
    # The results may still not be exactly reproducible
    np.random.seed(42)
    tf.random.set_seed(42)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    train_aug_df, test_df = import_dataset(args.data_dir)

    train_aug_df = map_class_labels(train_aug_df)
    test_df = map_class_labels(test_df)

    # Converting the filenames and target class labels into lists for augmented train and test datasets.
    train_aug_filenames_list = list(train_aug_df['filename'])
    train_aug_labels_list = list(train_aug_df['target'])

    test_filenames_list = list(test_df['filename'])
    test_labels_list = list(test_df['target'])

    # Creating tensorflow constants of filenames and labels for augmented train and test datasets from the lists defined above.
    train_aug_filenames_tensor = tf.constant(train_aug_filenames_list)
    train_aug_labels_tensor = tf.constant(train_aug_labels_list)

    test_filenames_tensor = tf.constant(test_filenames_list)
    test_labels_tensor = tf.constant(test_labels_list)

    num_classes = 7

    train_aug_dataset = get_train_datasets(train_aug_filenames_tensor, train_aug_labels_tensor, num_classes, args.batch_size)
    test_dataset = get_test_datasets(test_filenames_tensor, test_labels_tensor, num_classes, args.batch_size)

    final_cnn = define_cnn(num_classes, args.conv_filters, args.dense_nodes)
    final_cnn.summary()
    
    # Compiling the above created CNN architecture.
    final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Creating a TensorBoard callback object and saving it at the desired location.
    tensorboard = TensorBoard(log_dir=args.save_dir)

    # Creating a ModelCheckpoint callback object to save the model according to the value of val_accuracy.
    checkpoint_path = os.path.join(args.save_dir, f"{args.model_name}_checkpoint.h5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1
                                 )

    # Fitting the above created CNN model.
    final_cnn_history = final_cnn.fit(train_aug_dataset,
                                      batch_size=args.batch_size,
                                      validation_data=test_dataset,
                                      epochs=args.epochs,
                                      callbacks=[tensorboard, checkpoint],
                                      shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                                      )

    train_loss, test_loss, train_accuracy, test_accuracy = check_final_cnn_history(final_cnn_history)

    plot_training_history(final_cnn_history, args.save_dir)

    final_cnn_score = evaluate_model(final_cnn, test_dataset)

    # Saving the model as a h5 file for use later.
    final_save_path = os.path.join(args.save_dir, f"{args.model_name}_acc_{round(final_cnn_score[1], 3)}.h5")
    final_cnn.save(final_save_path, save_format='h5')

    final_cnn_pred = generate_predictions(final_cnn, test_dataset)

    conf_mat = generate_confusion_matrix(test_labels_list, final_cnn_pred)
    print(conf_mat)

    # Plotting the confusion matrix using the function defined above.
    cm_plot_labels = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

    plt.figure(figsize=(16,8))
    plot_confusion_matrix(conf_mat, cm_plot_labels, save_dir=args.save_dir, normalize=True,
                          title="Confusion Matrix based on predictions from CNN model",
                          export_as=f"{args.model_name}_conf_mat_norm"
                          )

    plt.show()

if __name__ == "__main__":
    main()
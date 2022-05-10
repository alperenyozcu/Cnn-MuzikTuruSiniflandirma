# import json
# from sklearn.model_selection import train_test_split
# import numpy as np
#
# test_size=0.25
# validation_size=0.2
#
# DATA_PATH = "Dataset.json"
#
# def load_data(data_path):
#     """Loads training dataset from json file.
#
#         :param data_path (str): Path to json file containing data
#         :return X (ndarray): Inputs
#         :return y (ndarray): Targets
#     """
#
#     with open(data_path, "r") as fp:
#         data = json.load(fp)
#
#     X = np.array(data["mfcc"])
#     y = np.array(data["labels"])
#     return X, y
#
#
#
# X, y = load_data(DATA_PATH)
#
# # create train, validation and test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
#
# # add an axis to input sets
# X_train = X_train[..., np.newaxis]
# X_validation = X_validation[..., np.newaxis]
# X_test = X_test[..., np.newaxis]
#
# print(X_train.shape[1])
# print(X_train.shape[2])

subprocess.call(['ffmpeg', '-i', file_path,
                         'test.wav'])
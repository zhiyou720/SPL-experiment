import csv
import numpy as np

def save_statistics(log_dir, statistics_file_name, list_of_statistics, create=False):
    """
    Saves a statistics .csv file with the statistics
    :param log_dir: Directory of log
    :param statistics_file_name: Name of .csv file
    :param list_of_statistics: A list of statistics to add in the file
    :param create: If True creates a new file, if False adds list to existing
    """
    if create:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)
    else:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)


def load_statistics(log_dir, statistics_file_name):
    """
    Loads the statistics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param statistics_file_name: The name of the statistics file
    :return: A dict with the statistics
    """
    data_dict = dict()
    with open("{}/{}.csv".format(log_dir, statistics_file_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").replace("\r", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").replace("\r", "").split(",")
            for key, item in zip(data_labels, data):
                if item not in data_labels:
                    data_dict[key].append(item)
    return data_dict

def get_best_validation_model_statistics(log_dir, statistics_file_name):
    """
    Returns the best val epoch and val accuracy from a log csv file
    :param log_dir: The log directory the file is saved in
    :param statistics_file_name: The log file name
    :return: The best validation accuracy and the epoch at which it is produced
    """
    log_file_dict = load_statistics(statistics_file_name=statistics_file_name, log_dir=log_dir)
    val_acc = np.array(log_file_dict['val_c_accuracy'], dtype=np.float32)
    best_val_acc = np.max(val_acc)
    best_val_epoch = np.argmax(val_acc)

    return best_val_acc, best_val_epoch


def build_experiment_folder(experiment_name, log_path):
    saved_models_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "saved_models")
    logs_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "summary_logs")
    import os

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    return saved_models_filepath, logs_filepath

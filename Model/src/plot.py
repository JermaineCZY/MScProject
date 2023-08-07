import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read the csv file
df = pd.read_csv('results.csv')

# Modify the value of the dataset to match the given dataset name
name_mapping = {
    'no_items': 'processed_data_no_items',
    '5_min': 'processed_data_5_min',
    '10_min': 'processed_data_10_min',
    '15_min': 'processed_data_15_min',
    '20_min': 'processed_data_20_min',
    '25_min': 'processed_data_25_min',
    '30_min': 'processed_data_30_min',
    '35_min': 'processed_data_35_min',
    'full': 'processed_data_full'
}

df['dataset'] = df['dataset'].map(name_mapping)

# group accuracy according to the model
accuracies_DNN = df[df['model'] == 'DNN']['accuracy'].values.tolist()
accuracies_RF = df[df['model'] == 'Random_Forest']['accuracy'].values.tolist()
accuracies_LR = df[df['model'] == 'Logistic_Regression']['accuracy'].values.tolist()
accuracies_XGB = df[df['model'] == 'XGBoost']['accuracy'].values.tolist()

datasets = df['dataset'].unique().tolist()


def line_plot(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB):
    plt.figure(figsize=(10, 6))
    plt.plot(datasets, accuracies_DNN, marker='o', label='DNN')
    plt.plot(datasets, accuracies_RF, marker='o', label='Random Forest')
    plt.plot(datasets, accuracies_LR, marker='o', label='Logistic Regression')
    plt.plot(datasets, accuracies_XGB, marker='o', label='XGBoost')

    plt.ylabel('Accuracy')
    plt.title('Accuracy by Models and Datasets')
    plt.xticks(rotation=45)
    plt.legend()

    plt.savefig("Accuracy_line_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def subplots(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs[0].plot(datasets, accuracies_DNN, marker='o')
    axs[0].set_title('DNN')
    axs[0].set_xticks(np.arange(len(datasets)))
    axs[0].set_xticklabels(datasets, rotation=45)
    axs[0].set_ylabel('Accuracy')

    axs[1].plot(datasets, accuracies_RF, marker='o')
    axs[1].set_title('Random Forest')
    axs[1].set_xticks(np.arange(len(datasets)))
    axs[1].set_xticklabels(datasets, rotation=45)
    axs[1].set_ylabel('Accuracy')

    axs[2].plot(datasets, accuracies_LR, marker='o')
    axs[2].set_title('Logistic Regression')
    axs[2].set_xticks(np.arange(len(datasets)))
    axs[2].set_xticklabels(datasets, rotation=45)
    axs[2].set_ylabel('Accuracy')

    axs[3].plot(datasets, accuracies_XGB, marker='o')
    axs[3].set_title('XGBoost')
    axs[3].set_xticks(np.arange(len(datasets)))
    axs[3].set_xticklabels(datasets, rotation=45)
    axs[3].set_ylabel('Accuracy')

    plt.tight_layout()

    plt.savefig("Accuracy_subplots.png", dpi=300, bbox_inches='tight')
    plt.close()


def bar_and_line_plot(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    width = 0.2
    x = np.arange(len(datasets))
    ax1.bar(x - 1.5 * width, accuracies_DNN, width, label='DNN', color='b', alpha=0.3)
    ax1.bar(x - 0.5 * width, accuracies_RF, width, label='Random Forest', color='r', alpha=0.3)
    ax1.bar(x + 0.5 * width, accuracies_LR, width, label='Logistic Regression', color='g', alpha=0.3)
    ax1.bar(x + 1.5 * width, accuracies_XGB, width, label='XGBoost', color='purple', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(datasets, accuracies_DNN, marker='o', color='b', label='DNN')
    ax2.plot(datasets, accuracies_RF, marker='o', color='r', label='Random Forest')
    ax2.plot(datasets, accuracies_LR, marker='o', color='g', label='Logistic Regression')
    ax2.plot(datasets, accuracies_XGB, marker='o', color='purple', label='XGBoost')

    ax1.set_xlabel('Datasets')
    ax1.set_ylabel('Accuracy (Bar)')
    ax2.set_ylabel('Accuracy (Line)')
    ax1.set_title('Accuracy by Models and Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)

    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

    plt.savefig("Accuracy_bar_and_line_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


# Execute
line_plot(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB)
subplots(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB)
bar_and_line_plot(datasets, accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB)


def get_accuracies(file_path):
    df = pd.read_csv(file_path)

    df['dataset'] = df['dataset'].map(name_mapping)

    accuracies_DNN = df[df['model'] == 'DNN']['accuracy'].values.tolist()
    accuracies_RF = df[df['model'] == 'Random_Forest']['accuracy'].values.tolist()
    accuracies_LR = df[df['model'] == 'Logistic_Regression']['accuracy'].values.tolist()
    accuracies_XGB = df[df['model'] == 'XGBoost']['accuracy'].values.tolist()

    return accuracies_DNN, accuracies_RF, accuracies_LR, accuracies_XGB


def line_plot_comparison(datasets, accuracies1, accuracies2, label1, label2):
    plt.figure(figsize=(10, 6))

    for i, model in enumerate(['DNN', 'Random Forest', 'Logistic Regression', 'XGBoost']):
        plt.plot(datasets, accuracies1[i], marker='o', linestyle='-', label=f'{model} {label1}')
        plt.plot(datasets, accuracies2[i], marker='x', linestyle='--', label=f'{model} {label2}')

    plt.ylabel('Accuracy')
    plt.title('Accuracy by Models and Datasets')
    plt.xticks(rotation=45)
    plt.legend()

    plt.savefig("Accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


# Read the accuracies from the two CSV files
accuracies1 = get_accuracies('results.csv')
accuracies2 = get_accuracies('results_new.csv')

# Compare the accuracies
#line_plot_comparison(datasets, accuracies1, accuracies2, 'old', 'new')

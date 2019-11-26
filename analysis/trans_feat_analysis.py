import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
from model_selection.score_util import Score_Util

"""
This class analyzes and plots the results of classification and regression models using only the 14 transformed features 
"""
def prepare_class_model_names():
    model_name = []
    for cl in Dictionaries.classifiers:
        for d in Dictionaries.datasets:
            model_name.append(d + '_' + cl)
    return model_name


def plot_all_F1_scores(task_name, score_name):

    parent_dir = Path.cwd().parent
    score_file = parent_dir.joinpath('default_results', 'score_files')

    model_name = prepare_class_model_names()

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_fect = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_anal.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        y1_feat = df_all_in_fect[score_name].tolist()
        y1_float_feat = [float(i) for i in y1_feat]
        print(model_name, y1_float_feat, sep='\n')

        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.scatter(model_name, y1_float_feat, color='b', marker='*', s=24)
        plt.title(emotion.capitalize() + ' - Classification - Using transformed features', fontsize=20)

        plt.xlabel('Models', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.ylabel('F1 score', fontsize = 20)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        #plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.33, wspace=0.2, hspace=0)
        plt.show()

def prepare_reg_model_names():
    model_name = []
    for reg in Dictionaries.regressors_dict:
        model_name.append(reg)
    return model_name

def plot_all_rmse_scores(task_name, score_name):
    parent_dir = Path.cwd().parent
    score_file = parent_dir.joinpath('default_results', 'score_files')

    model_name = Dictionaries.regressors

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_feat = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_anal.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection

        y1_fect = df_all_in_feat[score_name].tolist()
        y1_float_fect = [float(i) for i in y1_fect]
        print(model_name, y1_float_fect)
        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))

        ax.scatter(model_name, y1_float_fect, color='m', marker='*', s=24)
        plt.title(emotion.capitalize() + ' - Regression - Using transformed features', fontsize=20)

        plt.xlabel('Models', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16)
        plt.ylabel('RMSE', fontsize =20)
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        #plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.13, wspace=0.2, hspace=0)
        plt.show()

def main():
    plot_all_F1_scores('classification_', 'avg_cv_f1')
    plot_all_rmse_scores('regression_', 'avg_cv_rmse')



# init
if __name__ == '__main__':
    main()
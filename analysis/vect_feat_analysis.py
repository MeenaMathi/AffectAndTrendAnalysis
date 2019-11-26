import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
from model_selection.score_util import Score_Util

"""
This class analyzes and plots the results of classification and regression models using only the vectorized features 
"""
def prepare_class_model_names():
    model_name = []
    for vect in Dictionaries.vectorizers:
        for cl in Dictionaries.classifiers:
            for d in Dictionaries.datasets:
                model_name.append(d + '_' + cl + '_' + vect)
    # print(model_name)
    return model_name


def plot_all_F1_scores(task_name, score_name):

    parent_dir = Path.cwd().parent
    score_file = parent_dir.joinpath('default_results', 'score_files')

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_vect = pd.read_csv(score_file.joinpath(task_name + emotion + '_vect_anal.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        # F1 scores of original models
        df_all_in = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_1500 = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_1500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_1000 = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_1000.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_500 = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_250 = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_250.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_100 = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_100.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        model_name = prepare_class_model_names()
        y1_vect = df_all_in_vect[score_name].tolist()
        y1_float_vect = [float(i) for i in y1_vect]
        print(model_name, y1_float_vect, sep='\n')

        y1 = df_all_in[score_name].tolist()
        y2 = df_1500[score_name].tolist()
        y3 = df_1000[score_name].tolist()
        y4 = df_500[score_name].tolist()
        y5 = df_250[score_name].tolist()
        y6 = df_100[score_name].tolist()

        y1_float = [float(i) for i in y1]
        y2_float = [float(i) for i in y2]
        y3_float = [float(i) for i in y3]
        y4_float = [float(i) for i in y4]
        y5_float = [float(i) for i in y5]
        y6_float = [float(i) for i in y6]



        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.scatter(model_name, y1_float, color='orangered', marker='*', s=20)
        ax.scatter(model_name, y2_float, color='orangered', marker='*', s=20)
        ax.scatter(model_name, y3_float, color='orangered', marker='*', s=20)
        ax.scatter(model_name, y4_float, color='orangered', marker='*', s=20)
        ax.scatter(model_name, y5_float, color='orangered', marker='*', s=20)
        ax.scatter(model_name, y6_float, label='Models using all features', color='orangered', marker='*', s=20)

        ax.scatter(model_name, y1_float_vect, label='Models using only vect features', color='b', marker='*', s=20)

        plt.title(emotion.capitalize() + ' - Classification - All features Vs vectorized features', fontsize=20)
        plt.xlabel('Models', fontsize = 20)
        plt.ylabel('F1 score', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #plt.legend(bbox_to_anchor=(-0.05, 0.5, 0., 0.), loc=1)
        plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.45, wspace=0.2, hspace=0)
        plt.show()


def prepare_reg_model_names():
    model_name = []
    for vect in Dictionaries.vectorizers:
        for reg in Dictionaries.regressors_without_linreg:
                model_name.append(reg + '_' + vect)
    # print(model_name)
    return model_name

def plot_all_rmse_scores(task_name, score_name):
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'score_files')

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_vect = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_vect_anal.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        # RMSE scores of original models
        df_all_in = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_650 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_650.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_400 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_400.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_200 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_200.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_100 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_100.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_50 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_50.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        # Remove Linear Regressor model, as the values are extremely high which couldnt be plotted with the other lower error values
        df_all_in_vect_new = df_all_in_vect[df_all_in_vect.regressor != 'LinReg']
        df_all_in_new = df_all_in[df_all_in.regressor != 'LinReg']
        df_650_new = df_650[df_650.regressor != 'LinReg']
        df_400_new = df_400[df_400.regressor != 'LinReg']
        df_200_new = df_200[df_200.regressor != 'LinReg']
        df_100_new = df_100[df_100.regressor != 'LinReg']
        df_50_new = df_50[df_50.regressor != 'LinReg']

        model_name = prepare_reg_model_names()
        print(model_name, sep='\n')
        y1_vect = df_all_in_vect_new[score_name].tolist()
        y1 = df_all_in_new[score_name].tolist()
        y2 = df_650_new[score_name].tolist()
        y3 = df_400_new[score_name].tolist()
        y4 = df_200_new[score_name].tolist()
        y5 = df_100_new[score_name].tolist()
        y6 = df_50_new[score_name].tolist()


        y1_float_vect = [float(i) for i in y1_vect]
        y1_float = [float(i) for i in y1]
        y2_float = [float(i) for i in y2]
        y3_float = [float(i) for i in y3]
        y4_float = [float(i) for i in y4]
        y5_float = [float(i) for i in y5]
        y6_float = [float(i) for i in y6]

        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))

        print(len(model_name), y1_float_vect)

        ax.scatter(model_name, y1_float,  color='orangered', marker='*', s=20)
        ax.scatter(model_name, y2_float,  color='orangered', marker='*', s=20)
        ax.scatter(model_name, y3_float,  color='orangered', marker='*', s=16)
        ax.scatter(model_name, y4_float, color='orangered', marker='*', s=16)
        ax.scatter(model_name, y5_float, color='orangered', marker='*', s=16)
        ax.scatter(model_name, y6_float, label='Models using all features', color='orangered', marker='*', s=16)

        ax.scatter(model_name, y1_float_vect, label='Models using only vect features', color='g', marker='*', s=16)

        plt.title(emotion.capitalize() + ' - Regression - All features Vs vectorized features', fontsize=20)
        plt.xlabel('Models', fontsize=20)
        plt.ylabel('RMSE', fontsize =20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        #plt.legend(bbox_to_anchor=(-0.05, 0.5, 0., 0.), loc=1)
        plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.25, wspace=0.2, hspace=0)
        plt.show()

def main():
    plot_all_F1_scores('classification_', 'avg_cv_f1')
    plot_all_rmse_scores('regression_', 'avg_cv_rmse')


# init
if __name__ == '__main__':
    main()
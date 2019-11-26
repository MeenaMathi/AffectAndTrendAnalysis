import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
"""
This class analyzes and compares the results of classification task models based on order of resampling 
a) Resampling before the cross validation procedure
b) Resampling during the cross validation procedure
"""
def prepare_model_names():
    model_name = []
    resamplers = ['SMOTE', 'TomekLinks', 'SMOTETomek']
    for vect in Dictionaries.vectorizers:
        for cl in Dictionaries.classifiers:
            for d in resamplers:
                model_name.append(d + '_' + cl + '_' + vect)
    # print(model_name)
    return model_name


def create_dataframe(emotion, task_name, model_name, y1, y2, y3, y4, y5, y6, y7,y8,y9,y10):
    parent_dir = Path.cwd().parent
    graph_dir = parent_dir.joinpath('graph_files', task_name + emotion + 'resampling.csv')
    zippedList = list(zip(model_name, y1, y2, y3, y4, y5, y6, y7,y8,y9,y10))
    dataframe = pd.DataFrame(zippedList,columns=['Model_name', 'F1_all_in_old', 'F1_1500_old', 'F1_1000_old', 'F1_500_old', 'F1_250_old',
                                                 'F1_all_in_new', 'F1_1500_new', 'F1_1000_new', 'F1_500_new', 'F1_250_new'])
    dataframe.to_csv(graph_dir)

def plot_F1_scores(task_name, score_name):
    parent_dir = Path.cwd().parent
    score_file_old = parent_dir.joinpath('default_results', 'score_files_old')
    score_file_new = parent_dir.joinpath('default_results', 'score_files')

    task_name = task_name

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_old = pd.read_csv(score_file_old.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_1500_old = pd.read_csv(score_file_old.joinpath(task_name + emotion + '_feat_1500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_1000_old = pd.read_csv(score_file_old.joinpath(task_name + emotion + '_feat_1000.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_500_old = pd.read_csv(score_file_old.joinpath(task_name + emotion + '_feat_500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_250_old = pd.read_csv(score_file_old.joinpath(task_name + emotion + '_feat_250.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        df_all_in_new = pd.read_csv(score_file_new.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_1500_new = pd.read_csv(score_file_new.joinpath(task_name + emotion + '_feat_1500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_1000_new = pd.read_csv(score_file_new.joinpath(task_name + emotion + '_feat_1000.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_500_new = pd.read_csv(score_file_new.joinpath(task_name + emotion + '_feat_500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_250_new = pd.read_csv(score_file_new.joinpath(task_name + emotion + '_feat_250.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        #Remove the models trained on original dataset
        df_all_in_false_order = df_all_in_old[df_all_in_old.training_dataset != 'original']
        df_1500_false_order = df_1500_old[df_1500_old.training_dataset != 'original']
        df_1000_false_order = df_1000_old[df_1000_old.training_dataset != 'original']
        df_500_false_order = df_500_old[df_500_old.training_dataset != 'original']
        df_250_false_order = df_250_old[df_250_old.training_dataset != 'original']

        df_all_in_right_order = df_all_in_new[df_all_in_new.training_dataset != 'original']
        df_1500_right_order = df_1500_new[df_1500_new.training_dataset != 'original']
        df_1000_right_order = df_1000_new[df_1000_new.training_dataset != 'original']
        df_500_right_order = df_500_new[df_500_new.training_dataset != 'original']
        df_250_right_order = df_250_new[df_250_new.training_dataset != 'original']


        model_name = prepare_model_names()
        print(model_name, sep='\n')
        y1_old = df_all_in_false_order[score_name].tolist()
        y2_old = df_1500_false_order[score_name].tolist()
        y3_old = df_1000_false_order[score_name].tolist()
        y4_old = df_500_false_order[score_name].tolist()
        y5_old = df_250_false_order[score_name].tolist()

        y1_new = df_all_in_right_order[score_name].tolist()
        y2_new = df_1500_right_order[score_name].tolist()
        y3_new = df_1000_right_order[score_name].tolist()
        y4_new = df_500_right_order[score_name].tolist()
        y5_new = df_250_right_order[score_name].tolist()

        y1_float_old = [float(i) for i in y1_old]
        y2_float_old = [float(i) for i in y2_old]
        y3_float_old = [float(i) for i in y3_old]
        y4_float_old = [float(i) for i in y4_old]
        y5_float_old = [float(i) for i in y5_old]

        y1_float_new = [float(i) for i in y1_new]
        y2_float_new = [float(i) for i in y2_new]
        y3_float_new = [float(i) for i in y3_new]
        y4_float_new = [float(i) for i in y4_new]
        y5_float_new = [float(i) for i in y5_new]

        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        print(emotion, model_name, y1_float_old, y2_float_old, y3_float_old, y4_float_old, y5_float_old, sep='\n')
        print(emotion, model_name, y1_float_new, y2_float_new, y3_float_new, y4_float_new, y5_float_new, sep='\n')

        ax.scatter(model_name, y1_float_old, color='m', marker='X', s=20)
        ax.scatter(model_name, y2_float_old, color='m', marker='X', s=20)
        ax.scatter(model_name, y3_float_old, color='m', marker='X', s=20)
        ax.scatter(model_name, y4_float_old, color='m', marker='X', s=20)
        ax.scatter(model_name, y5_float_old, label='Resampling before cv', color='m', marker='X', s=20)

        ax.scatter(model_name, y1_float_new, color='b', marker='x', s=20)
        ax.scatter(model_name, y2_float_new, color='b', marker='x', s=20)
        ax.scatter(model_name, y3_float_new, color='b', marker='x', s=20)
        ax.scatter(model_name, y4_float_new, color='b', marker='x', s=20)
        ax.scatter(model_name, y5_float_new, label='Resampling during cv', color='b', marker='x', s=20)

        plt.title(emotion.capitalize() + ' Classification')
        plt.xlabel('Models', fontsize = 20)
        plt.ylabel('F1 score', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.45, wspace=0.2, hspace=0)
        plt.show()



def main():
    plot_F1_scores('classification_', 'avg_cv_f1')


# init
if __name__ == '__main__':
    main()
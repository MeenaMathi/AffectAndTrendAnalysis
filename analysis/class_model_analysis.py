import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
from model_selection.score_util import Score_Util

"""
This class analyzes and plots the results of classification task models
"""


def prepare_model_names():
    model_name = []
    for vect in Dictionaries.vectorizers:
        for cl in Dictionaries.classifiers:
            for d in Dictionaries.datasets:
                model_name.append(d + '_' + cl + '_' + vect)
    return model_name

def plot_all_F1_scores(task_name, score_name):
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'score_files')

    task_name = task_name

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_1500 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_1500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_1000 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_1000.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_500 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_500.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_250 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_250.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_100 = pd.read_csv(pickle_dir.joinpath(task_name + emotion + '_feat_100.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        model_name = prepare_model_names()

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

        print(emotion, model_name, y1_float, y2_float, y3_float, y4_float, y5_float, y6_float,sep='\n')

        ax.scatter(model_name, y1_float, label='k=all_in', color='k', marker='*', s=20)
        ax.scatter(model_name, y2_float, label='k=1500', color='c', marker='*', s=20)
        ax.scatter(model_name, y3_float, label='k=1000', color='m', marker='*', s=20)
        ax.scatter(model_name, y4_float, label='k=500', color='b', marker='*', s=20)
        ax.scatter(model_name, y5_float, label='k=250', color='g', marker='*', s=20)
        ax.scatter(model_name, y6_float, label='k=100', color='orangered', marker='*', s=20)

        plt.title(emotion.capitalize() + ' - Classification - F1 scores', fontsize=20)
        plt.xlabel('Models', fontsize = 20)
        plt.ylabel('F1 score', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.45, wspace=0.2, hspace=0)
        plt.show()

def plot_avg_F1_scores_for_model_components():
    for i, emotion in Dictionaries.emo_dict.items():
        merged_df = Score_Util.get_merged_df_for_emotion(emotion, 'classification_')
        dataset_score_dict = {}
        classifier_score_dict = {}
        vectorizer_score_dict = {}
        k_value_score_dict = {}

        for name in Dictionaries.datasets:
            df = merged_df[merged_df['training_dataset'] == name]
            dataset_score_dict[name] = pd.to_numeric(df['avg_cv_f1']).mean()
        for classifier_name, classifier in Dictionaries.classifier_dict.items():
            df = merged_df[merged_df['classifier']== classifier_name]
            classifier_score_dict[classifier_name] = pd.to_numeric(df['avg_cv_f1']).mean()
        for vectorizer_name, vectorizer in Dictionaries.vectorizer_dict.items():
            df = merged_df[merged_df['vectorizer'] == vectorizer_name]
            vectorizer_score_dict[vectorizer_name] = pd.to_numeric(df['avg_cv_f1']).mean()
        for k_int, k_str in Dictionaries.class_k_dict.items():
            df = merged_df[merged_df['features_selected'] == str(k_int)]
            k_value_score_dict[k_str] = pd.to_numeric(df['avg_cv_f1']).mean()
        print(emotion, dataset_score_dict, classifier_score_dict, vectorizer_score_dict, k_value_score_dict, sep='\n')

        fig = plt.figure()
        fig.suptitle(emotion.capitalize() + ': Classification - Average F1 scores')

        ax1 = plt.subplot(2, 2, 1)
        ax1.bar(*zip(*dataset_score_dict.items()), width= 0.6, color='olive')
        #plt.title('Datasets')
        for x, y in enumerate(dataset_score_dict.values()):
            plt.text(x=x-0.1, y=y+0.1, s=str(round(y,2)), fontdict=dict(fontsize=14))
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel('Datasets', fontsize=20)
        plt.ylabel('F1 scores', fontsize=20)

        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(*zip(*classifier_score_dict.items()), width=0.6, color='sienna')
        for x, y in enumerate(classifier_score_dict.values()):
            plt.text(x=x-0.1, y=y+0.1, s=str(round(y,2)), fontdict=dict(fontsize=14))
        #plt.title('Classifiers')
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel('Classifiers', fontsize=20)
        plt.ylabel('F1 scores', fontsize=20)

        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(*zip(*vectorizer_score_dict.items()), width=0.6, color= 'tan')
        for x, y in enumerate(vectorizer_score_dict.values()):
            plt.text(x=x-0.2, y=y+0.1, s=str(round(y,2)), fontdict=dict(fontsize=14))
        #plt.title('Vectorizers')
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel('Vectorizers', fontsize=20)
        plt.ylabel('F1 scores', fontsize=20)

        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(*zip(*k_value_score_dict.items()), width = 0.6, color='grey')
        for x, y in enumerate(k_value_score_dict.values()):
            plt.text(x=x-0.2, y=y+0.1, s=str(round(y,2)), fontdict=dict(fontsize=14))
        #plt.title('Number of selected features')
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel('Number of selected features', fontsize=20)
        plt.ylabel('F1 scores', fontsize=20)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.07, wspace=0.13, hspace=0.26)
        plt.show()

def plot_avg_F1_scores_for_individual_components(comp_name):

    color = ['olive', 'sienna', 'tan', 'grey' ]
    fig = plt.figure()

    j = 1
    for i, emotion in Dictionaries.emo_dict.items():
        merged_df = Score_Util.get_merged_df_for_emotion(emotion, 'classification_')
        score_dict = {}

        if comp_name == 'datasets':
            for name in Dictionaries.datasets:
                df = merged_df[merged_df['training_dataset'] == name]
                score_dict[name] = pd.to_numeric(df['avg_cv_f1']).mean()
            xlabel = 'Dataset'
        if comp_name == 'classifier':
            for classifier_name, classifier in Dictionaries.classifier_dict.items():
                df = merged_df[merged_df['classifier'] == classifier_name]
                score_dict[classifier_name] = pd.to_numeric(df['avg_cv_f1']).mean()
            xlabel = comp_name.capitalize()
        if comp_name == 'vectorizer':
            for vectorizer_name, vectorizer in Dictionaries.vectorizer_dict.items():
                df = merged_df[merged_df['vectorizer'] == vectorizer_name]
                score_dict[vectorizer_name] = pd.to_numeric(df['avg_cv_f1']).mean()
            xlabel =comp_name.capitalize()
        if comp_name == 'features_selected':
            for k_int, k_str in Dictionaries.class_k_dict.items():
                df = merged_df[merged_df['features_selected'] == str(k_int)]
                score_dict[k_str] = pd.to_numeric(df['avg_cv_f1']).mean()
            xlabel = 'Number of selected features'

        #fig,ax = plt.subplots(figsize = (10,5))
        ax = plt.subplot(2, 2, j)
        ax.bar(*zip(*score_dict.items()), width= 0.4, color= color[j - 1])
        for x, y in enumerate(score_dict.values()):
            plt.text(x=x-0.1, y=y+0.1, s=str(round(y,2)), fontdict=dict(fontsize=14))
        plt.title(emotion.capitalize(), fontsize = 20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        if comp_name=='vectorizer':
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel('F1 scores', fontsize=20)
        j +=1

    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.07, wspace=0.20, hspace=0.44)
    plt.show()

def main():
    plot_all_F1_scores('classification_', 'avg_cv_f1')
    plot_avg_F1_scores_for_individual_components('datasets')
    plot_avg_F1_scores_for_individual_components('classifier')
    plot_avg_F1_scores_for_individual_components('vectorizer')
    plot_avg_F1_scores_for_individual_components('features_selected')

    # REPEATED FOR INDIVIDUAL EMOTION
    #plot_avg_F1_scores_for_model_components()

# init
if __name__ == '__main__':
    main()
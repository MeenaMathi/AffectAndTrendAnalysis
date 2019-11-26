import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
from model_selection.score_util import Score_Util

"""
This class analyzes and plots the results of regression task models
"""
def prepare_model_names():
    model_name = []
    for vect in Dictionaries.vectorizers:
        for reg in Dictionaries.regressors_without_linreg:
                model_name.append(reg + '_' + vect)
    return model_name

def plot_all_rmse_scores(task_name, score_name):
    parent_dir = Path.cwd().parent
    result_dir = parent_dir.joinpath('default_results', 'score_files')

    task_name = task_name

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_0.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
        df_650 = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_650.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_400 = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_400.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_200 = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_200.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_100 = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_100.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        df_50 = pd.read_csv(result_dir.joinpath(task_name + emotion + '_feat_50.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)

        # Remove Linear Regressor model, as the values are extremely high which couldnt be plotted with the other lower error values
        df_all_in_new = df_all_in[df_all_in.regressor != 'LinReg']
        df_650_new = df_650[df_650.regressor != 'LinReg']
        df_400_new = df_400[df_400.regressor != 'LinReg']
        df_200_new = df_200[df_200.regressor != 'LinReg']
        df_100_new = df_100[df_100.regressor != 'LinReg']
        df_50_new = df_50[df_50.regressor != 'LinReg']

        model_name = prepare_model_names()

        y1 = df_all_in_new[score_name].tolist()
        y2 = df_650_new[score_name].tolist()
        y3 = df_400_new[score_name].tolist()
        y4 = df_200_new[score_name].tolist()
        y5 = df_100_new[score_name].tolist()
        y6 = df_50_new[score_name].tolist()

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
        ax.scatter(model_name, y2_float, label='k=650', color='c', marker='*', s=20)
        ax.scatter(model_name, y3_float, label='k=400', color='m', marker='*', s=20)
        ax.scatter(model_name, y4_float, label='k=200', color='b', marker='*', s=20)
        ax.scatter(model_name, y5_float, label='k=100', color='g', marker='*', s=20)
        ax.scatter(model_name, y6_float, label='k=50', color='orangered', marker='*', s=20)

        plt.title(emotion.capitalize() + ' - Regression - RMSE', fontsize=20)
        plt.xlabel('Models', fontsize = 20)
        plt.ylabel('RMSE', fontsize = 20)
        plt.xticks(range(len(model_name)), model_name, fontsize=16, rotation=90)
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        plt.legend(fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.30, wspace=0.2, hspace=0)
        plt.show()

def plot_avg_rmse_scores_for_model_components(task_name, score_name):

    for i, emotion in Dictionaries.emo_dict.items():
        merged_df = Score_Util.get_merged_df_for_emotion(emotion, task_name)
        #Remove Linear Regressor model, as the values are extremely high which couldnt be plotted with the other lower error values
        merged_df_new =  merged_df[merged_df.regressor != 'LinReg']

        regressor_score_dict = {}
        vectorizer_score_dict = {}
        k_value_score_dict = {}

        for regressor_name in Dictionaries.regressor_dict:
            df = merged_df_new[merged_df_new['regressor']== regressor_name]
            regressor_score_dict[regressor_name] = pd.to_numeric(df[score_name]).mean()
            for x, y in enumerate(regressor_score_dict.values()):
                plt.text(x=x - 0.1, y=y + 0.1, s=str(round(y, 3)), fontdict=dict(fontsize=14))
        for vectorizer_name in Dictionaries.vectorizer_dict:
            df = merged_df_new[merged_df_new['vectorizer'] == vectorizer_name]
            vectorizer_score_dict[vectorizer_name] = pd.to_numeric(df[score_name]).mean()
            for x, y in enumerate(regressor_score_dict.values()):
                plt.text(x=x - 0.1, y=y + 0.1, s=str(round(y, 3)), fontdict=dict(fontsize=14))
        for k_int, k_str in Dictionaries.reg_k_dict.items():
            df = merged_df_new[merged_df_new['features_selected'] == str(k_int)]
            k_value_score_dict[k_str] = pd.to_numeric(df[score_name]).mean()
            for x, y in enumerate(regressor_score_dict.values()):
                plt.text(x=x - 0.1, y=y + 0.1, s=str(round(y, 3)), fontdict=dict(fontsize=14))
        print(emotion, regressor_score_dict, vectorizer_score_dict, k_value_score_dict, sep='\n')


        fig = plt.figure()
        fig.suptitle(emotion.capitalize() + ': Regression - Average Root mean square error')

        ax1 = plt.subplot(2, 2, 1)
        ax1.bar(*zip(*regressor_score_dict.items()), width= 0.6, color='olive')
        #plt.title('Regressors')
        for x, y in enumerate(regressor_score_dict.values()):
            plt.text(x=x-0.1, y=y+0.01, s=str(round(y,3)), fontdict=dict(fontsize=14))
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        plt.xlabel('Regressors', fontsize=20)
        plt.ylabel('RMSE', fontsize = 20)

        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(*zip(*vectorizer_score_dict.items()), width=0.6, color='sienna')
        #plt.title('Vectorizers')
        for x, y in enumerate(vectorizer_score_dict.values()):
            plt.text(x=x-0.1, y=y+0.01, s=str(round(y,3)), fontdict=dict(fontsize=14))
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        plt.xlabel('Vectorizers', fontsize=20)
        plt.ylabel('RMSE', fontsize = 20)

        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(*zip(*k_value_score_dict.items()), width=0.6, color= 'tan')
        #plt.title('Number of selected features')
        for x, y in enumerate(k_value_score_dict.values()):
            plt.text(x=x-0.1, y=float(y)+0.01, s=str(round(y,3)), fontdict=dict(fontsize=14))
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        plt.xlabel('Number of selected features', fontsize=20)
        plt.ylabel('RMSE', fontsize=20)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.07, wspace=0.13, hspace=0.26)
        plt.show()

def plot_avg_rmse_scores_for_individual_components(comp_name):
    color = ['olive', 'sienna', 'tan', 'grey' ]
    fig = plt.figure()

    j = 1
    for i, emotion in Dictionaries.emo_dict.items():
        merged_df_temp = Score_Util.get_merged_df_for_emotion(emotion, 'regression_')
        # Remove Linear Regressor model, as the values are extremely high which couldnt be plotted with the other lower error values
        merged_df = merged_df_temp[merged_df_temp.regressor != 'LinReg']
        score_dict = {}

        if comp_name == 'regressor':
            for regressor_name, regressor in Dictionaries.regressor_dict.items():
                df = merged_df[merged_df[comp_name] == regressor_name]
                score_dict[regressor_name] = pd.to_numeric(df['avg_cv_rmse']).mean()
            c = 'sienna'
        if comp_name == 'vectorizer':
            for vectorizer_name, vectorizer in Dictionaries.vectorizer_dict.items():
                df = merged_df[merged_df[comp_name] == vectorizer_name]
                score_dict[vectorizer_name] = pd.to_numeric(df['avg_cv_rmse']).mean()
            c = 'tan'
        if comp_name == 'features_selected':
            for k_int, k_str in Dictionaries.reg_k_dict.items():
                df = merged_df[merged_df[comp_name] == str(k_int)]
                score_dict[k_str] = pd.to_numeric(df['avg_cv_rmse']).mean()
            c = 'grey'

        ax = plt.subplot(2, 2, j)
        ax.bar(*zip(*score_dict.items()), width= 0.4, color=color[j - 1])

        for x, y in enumerate(score_dict.values()):
            plt.text(x=x-0.1, y=y+0.01, s=str(round(y,3)), fontdict=dict(fontsize=14))
        plt.title(emotion.capitalize(), fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        if comp_name == 'vectorizer':
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
        plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        plt.xlabel(comp_name.capitalize(),fontsize=20)
        plt.ylabel('RMSE scores', fontsize=20)
        j+=1


    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.07, wspace=0.20, hspace=0.44)
    plt.show()

def main():
    plot_all_rmse_scores('regression_', 'avg_cv_rmse')
    plot_avg_rmse_scores_for_individual_components('regressor')
    plot_avg_rmse_scores_for_individual_components('vectorizer')
    plot_avg_rmse_scores_for_individual_components('features_selected')

    # REPEATED FOR INIDIVIDUAL EMOTION
    #plot_avg_rmse_scores_for_model_components('regression_', 'avg_cv_rmse')

# init
if __name__ == '__main__':
    main()
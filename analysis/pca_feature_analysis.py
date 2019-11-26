from model_selection.dictionaries import Dictionaries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import pandas as pd
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

"""
This class analyzes and plots the features used in classification task. - NOT INCLUDED IN WRITTEN WORK
"""
def prepare_data_for_analysis(emotion, task_name):
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')
    results_dir = parent_dir.joinpath('score_files')

    # Fit transform the vectorizer with the corresponding preprocessed training data
    if os.path.exists(pickle_dir.joinpath(emotion + '_' +task_name+'_train_preprocess_df.pkl')) & os.path.exists(pickle_dir.joinpath(emotion + '_' +task_name+'_train_feat_transform_df.pkl')):
        #Preprocess the data
        preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_' +task_name+'_train_preprocess_df.pkl'))
        # extract and transform the features
        trans_feat_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_' +task_name+'_train_feat_transform_df.pkl'))
    else:
        # If the file doesnt exist, exit the program with instructions
        print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py and add the files created in \'default_results\' folder')
        sys.exit(1)

    labels = preprocess_train_df['Affect Dimension'].values

    #Vectorize the data
    train_cv_uni = Dictionaries.vectorizer_dict['Count_Vect_Unigram'].fit_transform(preprocess_train_df['preprocessed_text'].values)
    train_tfidf_uni = Dictionaries.vectorizer_dict['TfIdf_Vect_Unigram'].fit_transform(preprocess_train_df['preprocessed_text'].values)
    train_cv_uni_bi = Dictionaries.vectorizer_dict['Count_Vect_Bigram'].fit_transform(preprocess_train_df['preprocessed_text'].values)
    train_tfidf_uni_bi = Dictionaries.vectorizer_dict['TfIdf_Vect_Bigram'].fit_transform(preprocess_train_df['preprocessed_text'].values)

    data_cv_uni = pd.DataFrame(train_cv_uni.toarray(), columns=Dictionaries.vectorizer_dict['Count_Vect_Unigram'].get_feature_names())
    data_tfidf_uni = pd.DataFrame(train_tfidf_uni.toarray(), columns=Dictionaries.vectorizer_dict['TfIdf_Vect_Unigram'].get_feature_names())
    data_cv_uni_bi = pd.DataFrame(train_cv_uni_bi.toarray(), columns=Dictionaries.vectorizer_dict['Count_Vect_Bigram'].get_feature_names())
    data_tfidf_uni_bi = pd.DataFrame(train_tfidf_uni_bi.toarray(), columns=Dictionaries.vectorizer_dict['TfIdf_Vect_Bigram'].get_feature_names())

    #Merge vectorized and transformed features
    features_cv_uni = pd.DataFrame(pd.concat([data_cv_uni, trans_feat_train_df], axis=1))
    features_tfidf_uni = pd.DataFrame(pd.concat([data_tfidf_uni, trans_feat_train_df], axis=1))
    features_cv_uni_bi = pd.DataFrame(pd.concat([data_cv_uni_bi, trans_feat_train_df], axis=1))
    features_tfidf_uni_bi = pd.DataFrame(pd.concat([data_tfidf_uni_bi, trans_feat_train_df], axis=1))

    #Standardize the data
    standardized_cv_uni = StandardScaler().fit_transform(features_cv_uni)
    standardized_tfidf_uni = StandardScaler().fit_transform(features_tfidf_uni)
    standardized_cv_uni_bi = StandardScaler().fit_transform(features_cv_uni_bi)
    standardized_tfidf_uni_bi = StandardScaler().fit_transform(features_tfidf_uni_bi)

    #Apply PCA
    #pca = PCA(0.95) # 95% variance needs about 1780-2040 components
    pca=PCA(n_components=2)
    pca_cv_uni = pca.fit_transform(standardized_cv_uni)
    #print(pca.explained_variance_ratio_)
    pca_tfidf_uni = pca.fit_transform(standardized_tfidf_uni)
    #print(pca.explained_variance_ratio_)
    pca_cv_uni_bi = pca.fit_transform(standardized_cv_uni_bi)
    #print(pca.explained_variance_ratio_)
    pca_tfidf_uni_bi = pca.fit_transform(standardized_tfidf_uni_bi)
    #print(pca.explained_variance_ratio_)

    return pca_cv_uni, pca_tfidf_uni, pca_cv_uni_bi, pca_tfidf_uni_bi, labels


def plot_the_features(dataframes, emotion, labels):
    fig = plt.figure()
    fig.suptitle(emotion.capitalize() + ' - Feature distribution')
    print(dataframes[0].shape, dataframes[1].shape, dataframes[2].shape, dataframes[3].shape)
    ax1 = plt.subplot(2, 2, 1)
    for lab, col in zip((emotion, 'other'),('maroon', 'midnightblue')):
        ax1.scatter(dataframes[0][labels == lab, 0],dataframes[0][labels == lab, 1],label=lab, c=col, marker = '*', s= 20)
        ax1.title.set_text('Count vectorizer - unigram')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')

    ax2 = plt.subplot(2, 2, 2)
    for lab, col in zip((emotion, 'other'), ('maroon', 'midnightblue')):
        ax2.scatter(dataframes[1][labels == lab, 0], dataframes[1][labels == lab, 1], label=lab, c=col, marker='*', s=20)
        ax2.title.set_text('Tf-idf vectorizer - unigram')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')

    ax3 = plt.subplot(2, 2, 3)
    for lab, col in zip((emotion, 'other'), ('maroon','midnightblue')):
        ax3.scatter(dataframes[2][labels == lab, 0], dataframes[2][labels == lab, 1], label=lab, c=col, marker='*', s=20)
        ax3.title.set_text('Count vectorizer - uni+bigram')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')

    ax4 = plt.subplot(2, 2, 4)
    for lab, col in zip((emotion, 'other'), ('maroon', 'midnightblue')):
        ax4.scatter(dataframes[3][labels == lab, 0], dataframes[3][labels == lab, 1], label=lab, c=col, marker='*', s=20)
        ax4.title.set_text('Tf-idf vectorizer - uni+bigram')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')

    plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.05, wspace=0.10, hspace=0.29)
    plt.show()


def main():

    print('Anger -Feature analysis - classification')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels= prepare_data_for_analysis('anger','c')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'anger', labels)

    print('Fear -Feature analysis - classification')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('fear', 'c')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'fear', labels)

    print('Joy -Feature analysis - classification')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('joy', 'c')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'joy', labels)

    print('Sadness -Feature analysis - classification')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('sadness', 'c')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'sadness', labels)

"""
    print('Anger -Feature analysis - regression')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels= prepare_data_for_analysis('anger','r')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'anger', labels)

    print('Fear -Feature analysis - regression')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('fear', 'r')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'fear', labels)

    print('Joy -Feature analysis - regression')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('joy', 'r')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'joy', labels)

    print('Sadness -Feature analysis - regression')
    cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df, labels = prepare_data_for_analysis('sadness', 'r')
    plot_the_features([cv_uni_df, tfidf_uni_df, cv_uni_bi_df, tfidf_uni_bi_df], 'sadness', labels)
"""

# init
if __name__ == '__main__':
    main()

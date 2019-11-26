import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from data_preparation.user_finder import User_Finder
from statistics import mean
import sys
"""
This class is analyze the change of affect over a period of 7 days from 29.04.2019 to 05.05.2019 and correlate with real world scenario, if there is any.
"""

def get_events_and_dates_for_the_user(user_name):
    # Add real world events (if present) that correlates with the number of tweets and affect tweets for each user
    user_event_dict = {
        'katyperry': {},
        'justinbieber': {'30.April':'\nTweet: Count down for his single',
                         '3.May':'\nInstagram picture + message: \nThe singer posted \nabout his mental state',
                        '4.May': '\nInstagram picture + message: \nThe singer defended \nChris Brown'},
        'taylorswift': {'1.May': '\nPerformance in BBMS awards'},
        'barackobama': {},
        'rihanna': {'4.May': '\nJustin Bieber defended Chris Brown'},
        'ladygaga': {'4.May': '\nMET GALA 2019 Opening ceremony'},
        'ellendegeneres': {'3.May': '\nThe Ellen show with Kenan Thompson as guest'},
        'justintimberlake': {'1.May': '\nTweet: Its gonna be May - Meme',
                        '3.May': '\nInstagram: Birthday message to Jessica Biel. \nTweet: Photos from concert at Verizon Arena'},
        'kimkardashianwest': {},
        'britneyspears': {'4.May': '\nInstagram: Yoga work out video'},
        'cristianoronaldo': {'4.May': '\nNews: The player said, \'Barcelona isn\'t for me\'.\nTweet: The player launched Striker Force 7 cartoon and posted the trailor'},
        'selenagomez': {'1.May': '\nInstagram picture: The singer designed and \nlaunched a special swimsuit'},
        'jimmyfallon': {},
        'arianagrande': {'30.April':'\nLive performance in Sweetener World Tour 2019\n at Portland city',
                         '1.May':'\nRemote performance in BBMS awards',
                         '2.May':'\nLive performance in Sweetener World Tour 2019\n at San Jose cityafter winning BBMS award'},
        'shakira': {},
        'ddlovato': {}
    }

    return user_event_dict[user_name]

def read_csv_into_df(path_name):
    parent_dir = Path.cwd().parent
    result_dir = parent_dir.joinpath('default_results', 'prediction_results')
    df = pd.read_csv(result_dir.joinpath(path_name))
    return df

def get_percentage(emo_tweets, total_tweets):
    #Calculate the percentage of the emotion tweets in total number of tweets
    emo_percent_list = []
    for emo,total in zip(emo_tweets, total_tweets):
        emo_percent = str(round((emo * 100.0 /total),2)) + '%'
        emo_percent_list.append(emo_percent)
    return emo_percent_list

#Plot the trend for the given user
def plot_trend_for_user(original_name):
    user_name = original_name.lower().replace(" ", "")
    df_1 = read_csv_into_df('final_results_' + user_name + '_2019-04-29.csv')
    df_2 = read_csv_into_df('final_results_' + user_name + '_2019-04-30.csv')
    df_3 = read_csv_into_df('final_results_' + user_name + '_2019-05-01.csv')
    df_4 = read_csv_into_df('final_results_' + user_name + '_2019-05-02.csv')
    df_5 = read_csv_into_df('final_results_' + user_name + '_2019-05-03.csv')
    df_6 = read_csv_into_df('final_results_' + user_name + '_2019-05-04.csv')
    df_7 = read_csv_into_df('final_results_' + user_name + '_2019-05-05.csv')

    dates = ['29.April', '30.April', '1.May', '2.May', '3.May', '4.May', '5.May']

    df_list = [df_1, df_2, df_3, df_4, df_5, df_6, df_7]
    total_tweets = []
    anger_tweets = []
    fear_tweets = []
    joy_tweets = []
    sadness_tweets = []
    other_tweets = []

    for df in df_list:
        total_tweets.append(df.shape[0])
        anger_tweets.append(df[df['anger'] == 1].shape[0])
        fear_tweets.append(df[df['fear'] == 1].shape[0])
        joy_tweets.append(df[df['joy'] == 1].shape[0])
        sadness_tweets.append(df[df['sadness'] == 1].shape[0])
        other_tweets.append(
            df[(df['anger'] == 0) & (df['fear'] == 0) & (df['joy'] == 0) & (df['sadness'] == 0)].shape[0])

    print(total_tweets, anger_tweets, fear_tweets, joy_tweets, sadness_tweets, other_tweets, sep='\n')
    # print(df_1.shape[0])
    percentage =[]
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    #fig.suptitle(original_name.capitalize())
    ax.plot(dates, total_tweets, color='brown',label='Total tweets')
    ax.plot(dates, anger_tweets, color='red', label='Anger tweets')
    percentage.append(get_percentage(anger_tweets, total_tweets))
    ax.plot(dates, fear_tweets, color='green', label='Fear tweets')
    percentage.append(get_percentage(fear_tweets, total_tweets))
    ax.plot(dates, joy_tweets, color='orange', label='Joy tweets')
    percentage.append(get_percentage(joy_tweets, total_tweets))
    ax.plot(dates, sadness_tweets, color='magenta', label='Sadness tweets')
    percentage.append(get_percentage(sadness_tweets, total_tweets))
    print(percentage)
    #ax.plot(dates, other_tweets, color='blue', label='other tweets')
    events_dict = get_events_and_dates_for_the_user(user_name)
    if events_dict:
        start = 0.5
        # The starting point of the text is selected differently for longer texts
        if user_name in('justintimberlake', 'selenagomez', 'arianagrande','ellendegeneres'):
            start = 0.75
        if user_name in('cristianoronaldo'):
            start = 0.90
        for date, event in events_dict.items():
            ax.axvline(x=date, color='k')
            plt.text(date, max(total_tweets)*start, event, fontsize=14, rotation=90, verticalalignment='top')

    # plot the percentage table under the graph
    tab = plt.table(cellText= percentage,
                          rowLabels=['Anger', 'Fear', 'Joy', 'Sadness'],
                          rowColours=['red', 'green', 'orange', 'magenta'],
                          colLabels=dates,
                          loc='bottom')
    tab.set_fontsize(20)
    plt.gca().get_xaxis().set_visible(False)
    plt.xlabel('Dates', fontsize=20)
    plt.ylabel('Number of tweets', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.20, wspace=0.2, hspace=0.20)
    plt.title(original_name.upper(), fontsize=20)
    plt.legend(fontsize=14)
    plt.show()

def find_average_total_tweets_and_affect_tweets(promi_user_dict):
    user_avg_affects = {}
    for number, names in promi_user_dict.items():
        user_name = names[0].lower().replace(" ", "")
        df_1 = read_csv_into_df('final_results_' + user_name + '_2019-04-29.csv')
        df_2 = read_csv_into_df('final_results_' + user_name + '_2019-04-30.csv')
        df_3 = read_csv_into_df('final_results_' + user_name + '_2019-05-01.csv')
        df_4 = read_csv_into_df('final_results_' + user_name + '_2019-05-02.csv')
        df_5 = read_csv_into_df('final_results_' + user_name + '_2019-05-03.csv')
        df_6 = read_csv_into_df('final_results_' + user_name + '_2019-05-04.csv')
        df_7 = read_csv_into_df('final_results_' + user_name + '_2019-05-05.csv')

        dates = ['29.April', '30.April', '1.May', '2.May', '3.May', '4.May', '5.May']

        df_list = [df_1, df_2, df_3, df_4, df_5, df_6, df_7]
        total_tweets = []
        anger_tweets = []
        fear_tweets = []
        joy_tweets = []
        sadness_tweets = []
        other_tweets = []

        for df in df_list:
            total_tweets.append(df.shape[0])
            anger_tweets.append(df[df['anger'] == 1].shape[0])
            fear_tweets.append(df[df['fear'] == 1].shape[0])
            joy_tweets.append(df[df['joy'] == 1].shape[0])
            sadness_tweets.append(df[df['sadness'] == 1].shape[0])
            other_tweets.append(
                df[(df['anger'] == 0) & (df['fear'] == 0) & (df['joy'] == 0) & (df['sadness'] == 0)].shape[0])

        anger_ratio = round((sum(anger_tweets) * 100.0 /sum(total_tweets)),2)
        fear_ratio = round((sum(fear_tweets) * 100.0 /sum(total_tweets)),2)
        joy_ratio = round((sum(joy_tweets) * 100.0 / sum(total_tweets)), 2)
        sadness_ratio = round((sum(sadness_tweets) * 100.0 / sum(total_tweets)), 2)
        user_avg_affects[names[0]] = [sum(total_tweets), sum(anger_tweets), anger_ratio, sum(fear_tweets), fear_ratio, sum(joy_tweets), joy_ratio, sum(sadness_tweets), sadness_ratio]

    return user_avg_affects

def main():
    #Get the user name and original name
    parent_dir = Path.cwd().parent
    hardvard_csv_data = pd.read_csv(parent_dir.joinpath('documents', 'hardvardDataverseDataset.csv'))
    top_twenty_twitter_users_list = hardvard_csv_data['author'].unique().tolist()

    promi_user_dict = {}
    j=1
    for user in top_twenty_twitter_users_list:
        if user not in ['YouTube', 'cnnbrk', 'Twitter', 'instagram']:
            original_name = User_Finder.get_user_original_name(user)
            promi_user_dict[j] = [original_name, user]
            j+=1

    #Find average number of tweets collected and individual affects for each prominent person.
    avg_affects_dict = find_average_total_tweets_and_affect_tweets(promi_user_dict)
    print('\nUsername: [Total tweets,anger tweets, anger percentage, fear tweets, fear percentage, joy tweets, joy percentage, sadness tweets, sadness percentage\n')
    for key, value in avg_affects_dict.items():
        print(key, value, sep=': ')
    print('\n')

    #Plot the trend of the affects for the selected user
    print(promi_user_dict)  # DO NOT REMOVE THIS PRINT STATEMENT
    selected_user = int(input("Please select the user, by entering (only) the user number: "))
    if selected_user not in range(1, 17):
        print('Please run the program again and enter a valid user number')
        sys.exit(1)
    else:
        plot_trend_for_user(promi_user_dict[selected_user][0])

# init
if __name__ == '__main__':
    main()

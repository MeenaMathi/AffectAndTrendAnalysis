from data_preparation.writer import Writer
from data_preparation.user_finder import User_Finder
import GetOldTweets3 as got
import re
from collections import Counter
import langid
import pandas as pd
import Levenshtein
from pathlib import Path
import urllib.error as err
import sys
from datetime import datetime as dt


"""
A class to collect tweets of a selected period of time for the users
"""
# Create a tweet: hashtags dictionary
tweets_dict = {}

def find_relevant_hashtag_in_tweets(hashtag_list, original_name, keyword):
    """
    A method to find the most relevant hashtag in tweets.
    Relevant hashtags are hashtags that satisfy either
    a) if the hashtag is a superset or subset of the original name tokens
    or b)if the Levenshtein distance between the hashtag and the original name tokens is less than 3
    """
    relevant_hashtags = []
    most_relevant_hashtag = ''
    print(hashtag_list)
    #Note: 'hashtag_list' is a list of lists.
    #Structure: hashtag_list = [hashtags1 (of tweet1), hashtags2 (of tweet2), ...,hashtagsN (of tweetN)]
    #hashtagsX = [hashtag1, hashtag2, ...hashtagM] of tweet X

    for hashtags in hashtag_list:
        #print(hashtags)
        if len(hashtags) != 0:
            for hashtag in hashtags.split(' '):
                #Lower the case and remove #
                hashtag_lower = hashtag.lower().replace('#', '')
                for token in original_name.split(" "):
                    #Check if either the token is a substring of hashtag_lower or vice versa or if the Levenshtein distance is less than 4
                    if (token in hashtag_lower)|(hashtag_lower in token)|(Levenshtein.distance(token, hashtag_lower) < 3):
                        # Dont add the hashtag if it is the keyword used for previous search
                        if (hashtag_lower != keyword):
                            relevant_hashtags.append(hashtag_lower)
    # find the top most ranked relevant hashtags
    if len(relevant_hashtags)> 0:
        relevant_hashtag = Counter(relevant_hashtags).most_common(1)
        most_relevant_hashtag = relevant_hashtag[0][0]
    return most_relevant_hashtag

def remove_urls_from_tweet(tweet):
    """
    A method to remove the urls (that begin with http /www) from the tweets
    """
    url_removed_tweet = tweet
    urls = re.findall('http\S+|www\S+', tweet)
    #print(urls)
    if urls.__len__() > 0:
        for url in urls:
            tweet = tweet.replace(url, "")
            url_removed_tweet = tweet
    return url_removed_tweet

def extract_tweets(keyword, original_name, user_name, username_flag, hashtag_flag, since_time, until_time, max_tweets):
    """
    A method to extract tweets of a selected period of time for the selected user
    """
    print(username_flag, hashtag_flag)
    if max_tweets <= 0:
        max_tweets  = -1 # Default value for unlimited number of tweets

    #if keyword in ('youtube', 'twitter', 'instagram', 'cnnbrk'):
    #    maxTweets = 100000

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword) \
                                                .setSince(since_time) \
                                                .setUntil(until_time) \
                                                .setMaxTweets(max_tweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    print(len(tweets))
    hashtag_list = []
    for item in tweets:
        tweet = item.text
        urls = item.urls
        #print(item.username)
        # Remove URLs to find the original duplicates
        tweet = remove_urls_from_tweet(tweet)
        #Save only the english tweets (after URL removal)
        if langid.classify(tweet)[0] == 'en':
            if len(tweet) > 1:
                # Otherwise, collect other info and proceed further
                list_of_info = []
                hashtags = item.hashtags
                list_of_info.append(item.date)
                list_of_info.append(hashtags)
                list_of_info.append(tweets.__len__())
                hashtag_list.append(hashtags)
                # Removes the duplicate entries automatically by overwriting
                tweets_dict[tweet]= list_of_info
    # Return if atleast 1000 unique Tweets are collected, else, redo the search
    if(tweets_dict.keys().__len__() >= 1000):
        print("Data count >= 1000")
        tweets_final = tweets_dict.copy()
        # Clear the dictionary items for the next user
        tweets_dict.clear()
        print(len(tweets_final))
        return tweets_final
    else:
        print("Data count < 1000")
        # 2nd try: Repeat the search with max_tweets = 6000 for the same keyword
        # 3rd try: Repeat the search with the most relevant hashtag (or the username in case of unavailability) with max_tweets = 4000
        if (hashtag_flag == True) & (username_flag == True):
            print('TT 1')
            # CASE 1: If both the flags are true, there are no enough tweets for the user.
            print('1000 unique tweets could not be collected using the keywords selected for the user:' + original_name)
            return tweets_dict
        if (hashtag_flag == False):
            print('FF 2')
            # CASE 1: If hashtag flag is false( => both the flags are false), repeat the search with relevant hashtag
            most_relevant_hashtag = find_relevant_hashtag_in_tweets(hashtag_list,original_name.lower(),keyword)
            if most_relevant_hashtag != '':
                print(most_relevant_hashtag)
                return extract_tweets('#' + most_relevant_hashtag, original_name, user_name, False, True, since_time, until_time, max_tweets)
            else:
                hashtag_flag = True
        if(hashtag_flag == True) & (username_flag == False):
            print('TF 3')
            # CASE 2: If hashtag flag is true, but username flag is false, repeat the search using username as keyword, only if it is different from keyword
            if (user_name != keyword):
                return extract_tweets(user_name.lower().replace("@", "#"), original_name, user_name, True, True, since_time, until_time, max_tweets)
            else:
                username_flag = True
                return tweets_dict


def get_info_from_user():
    """
    A method to get the user name, period of time, number of tweets from the user
    """
    # Get top 20 twitter usernames from Hardvard Dataverse Dataset
    parent_dir = Path.cwd().parent
    documents_dir = parent_dir.joinpath('documents')
    hardvard_csv_data = pd.read_csv(documents_dir.joinpath("hardvardDataverseDataset.csv"))
    top_twenty_twitter_users_list = hardvard_csv_data['author'].unique().tolist()
    #print(top_twenty_twitter_users_list)
    user_dict= {}
    for user_nr in range(len(top_twenty_twitter_users_list)):
        original_name = User_Finder.get_user_original_name(top_twenty_twitter_users_list[user_nr])
        user_dict[user_nr] = original_name

    print("\nNote:\n"+
          "## The tweet collection is restricted to the period between 01-01-2019 and 31-05-2019 for convenience.\n" +
          "## The until date is not included in search.\n" +
          "## If there are tweets less than the selected number of tweets, only the available tweets are retrieved\n")
    print(user_dict)
    selected_user = int(input("Please enter (only) the number of the user you wish to collect the tweets: "))
    since_date = input("Please enter a valid since date in the format[yyyy-mm-dd] in the year 2019: ")
    until_date = input("Please enter a valid until date in the format[yyyy-mm-dd] : ")
    max_tweets = input("Please give a number > 0 to denote maximum number of tweets to be collected. "
                       "\nType [0] for retrieving all the tweets in the given period of time: ")

    if selected_user not in range(0, 20):
        print('Invalid entry - user selection number is out of range. Please re-run and enter a valid entry.')
        sys.exit(1)
    elif (re.search("^2019-([0][1-5])-([0|1|2][0-9])|([3][0|1])$", since_date)==None) or (re.search("^2019-([0][1-5])-([0|1|2][0-9])|([3][0|1])$", until_date) == None) :
            print('Invalid date format. Please re-run and enter a valid date in the format [yyyy-mm-dd]')
            sys.exit(1)
    elif (re.search("^2019-02-29$", since_date) != None) or (re.search("^2019-02-29$", until_date) != None)\
                or (re.search("^2019-02-30$", since_date) != None) or (re.search("^2019-02-30$", until_date) != None)\
                or (re.search("^2019-02-31$", since_date) != None) or (re.search("^2019-02-31$", until_date) != None)\
                or (re.search("^2019-04-31$", since_date) != None) or (re.search("^2019-04-31$", until_date) != None):
            print( 'The selected month doesnt contain this date')
            sys.exit(1)
    elif max_tweets.isdigit() == False or int(max_tweets) < 0:
        print('Invalid max_tweets entry. Please re-run and enter a valid integer >=0')
        sys.exit(1)
    else:
        print("since_date: " + str(since_date))
        print("until_date: " + str(until_date))
        print("selected_user: " + user_dict[selected_user])
        print("max_tweets: " + max_tweets)
        print('Collecting tweets...')
        u = until_date.split('-')
        s = since_date.split('-')
        u_date = dt(int(u[0]), int(u[1]), int(u[2]))
        s_date = dt(int(s[0]), int(s[1]), int(s[2]))
        print(u_date, s_date)
        if (s_date >= u_date):
            print("Until date is not valid. Please re-run and enter a valid date")
            sys.exit(1)
        else:
            return (user_dict[selected_user]).lower().replace(" ", ""), user_dict[selected_user], top_twenty_twitter_users_list[selected_user], since_date, until_date, int(max_tweets)

def main():
    keyword, original_name, user, since_date, until_date, max_tweets = get_info_from_user()
    print(keyword, original_name, user, since_date, until_date, max_tweets)

    try:
        tweets_dict = extract_tweets(keyword, original_name, user, False, False, since_date, until_date, max_tweets)
        print(tweets_dict)
        if (tweets_dict != None):
            Writer.write_user_test_data_to_predict(tweets_dict, keyword, "_" + since_date)
    except err.HTTPError as error:
        print(error.headers)
        print(error.read())
        print('Please try after few hours')

# init
if __name__ == '__main__':
    main()
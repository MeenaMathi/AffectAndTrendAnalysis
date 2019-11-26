import tweepy
import json
import yaml
from pathlib import Path

class User_Finder:
    """
    A class to find the original name from the given twitter user name
    """
    # Get Twitter credentials from twitter_keys.yml.yaml file
    parent_dir = Path.cwd().parent
    #print(parent_dir)
    file_path = parent_dir.joinpath('documents', 'twitter_keys.yml')
    #print(file_path)
    with open(file_path, 'r') as input_file:
        credentials = yaml.safe_load(input_file)

    auth = tweepy.OAuthHandler(credentials.get('consumer_key'), credentials.get('consumer_secret'))
    auth.set_access_token(credentials.get('access_token'), credentials.get('access_secret'))
    api = tweepy.API(auth)

    def get_user_original_name(input_name):
        name =''
        try:
            twitter_user_obj = User_Finder.api.get_user('@' + input_name)
            #print(twitter_user_obj)
            user_json_attribute = json.dumps(twitter_user_obj._json)
            json_obj = json.loads(user_json_attribute)
            #print(json_obj['name'])
            name = json_obj['name']
        except Exception:
            name = input_name
        return name
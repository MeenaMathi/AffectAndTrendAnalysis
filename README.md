# AffectAndTrendAnalysis
Affect and Trend analysis of Tweets

Requirements: Python 3.7

For installing packages:
    > Run in your shell: pip install -r requirements.txt
    > Additionally, you may also need to download NLTK's 'Stopwords Corpus' using:
            python -m nltk.downloader stopwords
            (or)
            import nltk
            nltk.download('stopwords')

To run individual tasks, run:
    data_preparation > tweets_collector.py      to collect the tweets for the selected user for the selected period of time
    model_selection  > modeling.py              to train the model for the given emotion, task, analysis, k value with 5 fold cross validation

    (The following classes require the documents/results from the default_results folder(not included here) and can be run only after the training phase. The new results are currently stored in new_results folder, which should be renamed as default_results before running predictor and analysis classes.)
    model selection  > model_assessment.py      to predict the affect and intensity of the test(-gold) data and to find the evaluation scores
    trend_analysis   > predictor.py             to predict the affect and intensity of the (default/collected) tweets for the selected top user
    trend_analysis   > affect_trend_analysis    to analyze the trend of the affect for a period of 7 days for the selected user and correlate with the real world events(if present)

    analysis > data_analysis                    to analyze the Sem Eval Task data, the training and test data used in the classification and regression task
    analysis > class_model_analysis.py          to analyze the models and model components used in classification task
    analysis > reg_model_analysis.py            to analyze the models and model components used in classification task
    analysis > vect_feat_analysis.py            to analyze the models that uses only the vectorized features
    analysis > trans_feat_analysis.py           to analyze the models that uses only the 14 transformed features
    analysis > trans_feat_rank_analysis.py      to analyze the relevancy of the 14 transformed features
    analysis > resampling_order_analysis.py     to analyze the effect of resampling order before and during cross validation
    analyis  > other_analysis                   to analyze various other results obtained (listed in the comments) during training phase


Important Notes:
    Currently the tweets can be collected only for the selected users in the list of 20 users, and the period of collection is restricted from 01-01-2019 to 30-05-2019 for convenience.
    Currently prediction can be done only for the tweets present in default_results folder(not included here)
    Except tweets_collector.py and modeling.py classes, other classes need the documents/results from default_results folder(not included here).
    The newly produced results are stored in new_results folder(which should be renamed as default folders for running predictor and analysis classes)


Mandatory documents are stored in 'documents' folder
Training data: (Data from CodaLab)
    > 2018-E-c-En-train.txt, 2018-E-c-En-dev.txt, 2018-E-c-En-test-gold.txt,
    > 2018-EI-reg-En-anger-dev.txt, 2018-EI-reg-En-anger-test-gold.txt, EI-reg-En-anger-train.txt,
    > 2018-EI-reg-En-fear-dev.txt, 2018-EI-reg-En-fear-test-gold.txt, EI-reg-En-fear-train.txt,
    > 2018-EI-reg-En-joy-dev.txt, 2018-EI-reg-En-joy-test-gold.txt, EI-reg-En-joy-train.txt,
    > 2018-EI-reg-En-sadness-dev.txt, 2018-EI-reg-En-sadness-test-gold.txt, EI-reg-En-sadness-train.txt,
Other files needed:
    > stats folder (Ekphrasis)
    > hardvardDataverseDataset.csv (User name extraction)
    > anger.words, fear.words, joy.words, sadness.words (Query words from CodaLab)
    > NRC-AffectIntensity-Lexicon.txt, NRC-Emotion-Lexicon-Worldlevel-v0.92.txt, NRC-Hashtag-Emotion-Lexicon-v0.2.txt (Emotion Lexicon)
    > twitter_keys (property file to use twitter account)

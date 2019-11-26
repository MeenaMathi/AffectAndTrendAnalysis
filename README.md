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

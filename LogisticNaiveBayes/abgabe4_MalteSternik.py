"""
Aufgabenbeschreibung: 
Gegeben ist der Datensatz movie_reviews_10k.csv, der 10.000 Filmkritiken enthält, die jeweils als "positive" oder
"negative" klassifiziert wurden. Aufgabe: Erstellen Sie mithilfe der Library scikit-learn zwei Klassifikatoren,
die jeweils die ersten 80% des Datensatzes als Trainingsdaten und die restlichen Daten als Testdaten
(jeweils ohne Shuffle!) nutzen:

a)  einen Naive Bayes Klassifikator mit Bag-of-Word Features

b)  einen Logistic Regression Klassifikator mit den Features, die auf S. 81 in Jurafsky & Martin verwendet werden.
    Dies sind pro Dokument:
    - Anzahl positiver Lexikonwörter
    - Anzahl negativer Lexikonwörter
    - ob "no" enthalten ist
    - Anzahl Pronomen der 1. und 2. Person
    - ob "!" enthalten ist
    - Anzahl Tokens (logarithmiert mit der Funktion math.log)

Als Sentiment Lexikon steht die Datei WKWSCISentimentLexicon_v1.1.xlsx zur Verfügung. Darin stehen negative Werte für
ein negatives Sentiment und positive Werte für ein positives Sentiment.

Für jeden der beiden Klassifikatoren soll am Ende die Accuracy auf dem Testset berechnet und ausgegeben werden.

Neben scikit-learn dürfen beliebige weitere Libraries verwendet werden, z.B. für die Tokenisierung.

Zusatzaufgabe (Challenge): Optimieren Sie das Klassifikationsergebnis, indem Sie z.B. die Features verändern.
Achtung: Sie dürfen dazu mehrfach auf dem gleichen Testset testen, was in der Praxis eigentlich nicht erlaubt wäre.
Verwenden Sie für die Zusatzaufgabe ein neues Python-Skript, sodass das Skript mit den "Original-Features" als HA
gewertet werden kann.

Datenquellen:
movie_reviews_10k.csv: die ersten 10.000 Einträge aus diesem Datensatz: https://github.com/SK7here/Movie-Review-Sentiment-Analysis/blob/master/IMDB-Dataset.csv
WKWSCISentimentLexicon_v1.1.xlsx: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/DWWEBV
"""

################################
# Imports
################################

from pathlib import Path
import nltk
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd

# Eigene imports
import csv
import math

################################
# Functions
################################

#########################################
### TASK A: NAIVE BAYES CLASSIFICATOR ###
#########################################

def train_NB_classificator(data_filename:Path):
    """Returns trained NB Classificator (using Scikitlearn) and the 'remaining' 
       test_data (split into docs and y-values) extracted from a CSV File whereas
       80% of the data is used for training the classificator, as well as the bag-of-words
       vectorizer used to fit and transform the data.
    """

    # Reading csv data file
    df_movie_reviews = pd.read_csv(data_filename, sep=",")
    # Extract features
    docs_train, docs_test, y_train, y_test = train_test_split(
        df_movie_reviews['review'].to_list(), # docs (the movie reviews)
        df_movie_reviews['sentiment'].to_list(), # y (the sentiment which we try to classify/predict)
        shuffle=False, # Shuffle off according to task description
        test_size = 0.20, # 20% of the data is used for testing, 80% for training 
        random_state = 12 # Random state for replicable results
                        )
    
    # Intializing a vectorizer to create a bag-of-words
    movie_vectorizer = CountVectorizer()
    # Creating counts of a bag-of-words from our documents
    docs_train_counts = movie_vectorizer.fit_transform(docs_train)

    # Initializing Naive Bayes Classificator and train it with our data
    classificator = MultinomialNB()
    classificator.fit(
        docs_train_counts, # Setting 'training vector'
        y_train # Setting our target-values (the according sentiments)
                                )
    
    # return trained classificator as well as test_data
    return classificator, docs_test, y_test, movie_vectorizer



def task_a(movie_reviews:Path):
    """Creates, trains, tests and evaluates Scikitlearn Naive Bayes Classificator.
       Prints Evaluation report.
    """
    # Creating Scikit-Learn NB Classificator, trained on input movie_reviews
    # and unpack document_test data for testing as well as vectorizer
    classificator, docs_test, y_test, movie_vectorizer = train_NB_classificator(movie_reviews)

    # Testing the classificator
    docs_test_counts = movie_vectorizer.transform(docs_test)
    y_pred = classificator.predict(docs_test_counts)

    # Evaluate classifcator with comparing y-values to predicted values
    print(classification_report(y_test, y_pred))# Print evaluation

####################################
### TASK B: LOGISTICS REGRESSION ###
####################################

def extract_lexicon(lexicon:pd.DataFrame) -> list:
    """ Extracts positive and negative words from sentiment
        lexicon and returns both as respective lists.
    """

    # NOTE: Because for every token for each single review
    #       there has to be a check if a word is positive
    #       or negative, it saves a LOT of iteration time to
    #           1. Leave out the neutral sentiments (1/3 - 2/3 of the lexicon)
    #           2. Split positive and negative words for even shorter specific iterations
    #           3. Finally iterate through python lists rather than dataframes (a lot faster)
    
    pos_words = []
    neg_words = []

    # Iterating the data frames through each row
    for index, elt in lexicon.iterrows():

        # When the current element's sentiment is above 0 -> positive word
        if lexicon.at[index, 'sentiment'] > 0: pos_words.append(lexicon.at[index, 'term'])

        # When the current element's sentiment is below 0 -> negative word
        elif lexicon.at[index, 'sentiment'] < 0: neg_words.append(lexicon.at[index, 'term'])

    # Return values
    return pos_words, neg_words




def featurize_reviews(movie_reviews:Path, file_path:Path, sentiment_lexicon:str, save_as_csv:bool = True):
    """ Function that creates the features for the input reviews
        and returns featurized list of the reviews, each feature,
        and actual y value (positive or negative review)"""
    
    # Read movie review from filepath
    df_movie_reviews = pd.read_csv(movie_reviews, sep=",")

    # Initialize column names for featurized dataframe
    col_names = [
        "review",
        "count(pos_words)",
        "count(neg_words)",
        "contains(no)", 
        "count(1. 2. Pronoun)",
        "contains(!)",
        "log. Amount tokens",
        "sentiment"
                            ]
    
    # Initializing featurized dataframe 
    featurized_df = pd.DataFrame(columns=col_names)

    # Read lexicon for sentiment-reference
    lexicon = pd.read_excel(sentiment_lexicon)

    # Extract positive and negative word list from lexcion for faster iteration
    pos_words, neg_words = extract_lexicon(lexicon)
    

    # Iterate each row of the movie review dataframe
    for index, row in df_movie_reviews.iterrows():
        

        # Tokenize review
        tokenized_review = nltk.word_tokenize(row['review'])
        
        # Set features
        print("Featurizing.... (",index, " / ", len(df_movie_reviews),")") # print progress
        pronouns = ["i", "you", "we"] # Set 1. and 2. person pronouns for english
        x_1 = 0 # count positive words
        x_2 = 0 # count negative words
        x_3 = 0 # contains 'no'
        x_4 = 0 # count pronouns
        x_5 = 0 # contains '!'
        x_6 = math.log(len(tokenized_review)) # log amount of tokens

        # Iterate each token for feature - information
        for tok in tokenized_review:
            
            if tok in pos_words: x_1 += 1 # x_1
            if tok in neg_words: x_2 += 1 # x_2
            if tok.lower() == "no": x_3 = 1 # x_3 (.lower() to ignore case)
            if tok.lower() in pronouns: x_4 += 1 # x_4 (.lower() to ignore case)
            if tok == "!": x_5 = 1 # x_5
        
        # Concenate lists and append to dataframe
        featurized_row = [row['review']] + [x_1]+ [x_2]+ [x_3]+ [x_4] + [x_5] + [x_6] + [row['sentiment']]
        featurized_df.loc[len(featurized_df)] = featurized_row # Appending at the "end" (len(df)) of the dataframe

        
    if save_as_csv:
        featurized_df.to_csv(file_path, sep=",", encoding="utf-8", index=False) # Writing dataframe to .csv file
        featurized_df = None # save memory through returning 'none' since dataframe is stored as file
    
    # Print log:
    print(f"\nDONE. (featurized list and saved in {file_path}\n)")
    print(f"!!! Note: set 'write_list' in  task_b(write_list = FALSE)")
    print(f"so that the featurized list does not have to be created again")

    # Return dataframe ('None' if saved as csv)
    return featurized_df



def train_logistic_regression(
                            movie_reviews:str,
                            csv_filepath:str, 
                            sentiment_lexicon:str,
                            test_size:float = 0.2,
                            skip_featurization:bool = False
                                                            ):
    """Featurizes and splits movie reviews + sentiments to trainable data  
       saves it as .csv and fits a logistic regression to classify sentiments
       on test data.
    """

    # NOTE: Turn "skip_featurization" to 'false' if program already created .csv file -> saves a LOT of runtime!

    # If not skipped featurize movie reviews
    if not skip_featurization:
        # Featurize the movie reviews and generate CSV containing these features as X - Values
        # (Can also be stored in the variable if save_as_csv = False)
        feature_list = featurize_reviews(
                                        movie_reviews,
                                        file_path = csv_filepath, 
                                        sentiment_lexicon = sentiment_lexicon, 
                                        save_as_csv = True
                                                                                )
     
    # Import the featurized movie reviews to dataframe
    df = pd.read_csv(csv_filepath, index_col=False)

    # Extract the columns only containing the features
    feature_cols = df.columns[1:-1]

    X = df[feature_cols] # extract all x - values (features)
    y = df.sentiment # extract y - value (sentiment)

    # Split data to training and test set (80% of the data for training as default)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(1-test_size), shuffle=False)

    # Initialize and train logistic regression model with training data
    # (Setting a random_state for replicable data)
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)

    # Apply logistic regression and classify test data
    y_pred = logreg.predict(X_test)

    # Return y values of test data and the predicted y values
    return y_test, y_pred

def task_b(movie_reviews:str, csv_filepath:str, sentiment_lexicon:str, skip_featurization:bool):
    """
    Task b: Extracts movie reviews with sentiments, featurizes them and trains
    a Scikit-Learn Logisitc Regression Classificator with 80% of the data,
    and applies it on the other 20% of the data. Prints evaluation report.
    """

    # Train and apply logistic regression
    y_test, y_pred = train_logistic_regression(
        movie_reviews,
        csv_filepath = csv_filepath,
        sentiment_lexicon = sentiment_lexicon, 
        test_size = 0.2, 
        skip_featurization = skip_featurization
                                                )
    
    # Calculate accuracy score if necessary?
    #accuracy_score(y_test, y_pred)

    # Print classification report
    print(classification_report(y_test, y_pred))


# Funktion, die alle weiteren Funktionen aufruft
def run_script(movie_reviews, sentiment_lexicon):
    """Funktion, die alle weiteren Funktionen aufruft
    :param movie_reviews: Dateiname der Reviews
    :param sentiment_lexicon: Dateiname des Sentiment Lexikons
    """
    
    # TASK A
    task_a(movie_reviews)

    # TASK B

    # NOTE: Set 'skip_feauturization' to 'True' if the csv file has already been generated
    # (saves a LOT of runtime!). Mostly relevant if program runs multiple times

    # task_b(
    #     movie_reviews,
    #     csv_filepath = "featurized_reviews.csv",
    #     sentiment_lexicon= sentiment_lexicon,
    #     skip_featurization = False)


   
    

################################
# Hauptprogramm
################################

if __name__ == "__main__":

    movie_reviews = Path("movie_reviews_10k.csv")
    sentiment_lexicon = Path("WKWSCISentimentLexicon_v1.1.xlsx")
    
    # rufe Funktion auf, die alle weiteren Funktionen aufruft
    run_script(movie_reviews, sentiment_lexicon)




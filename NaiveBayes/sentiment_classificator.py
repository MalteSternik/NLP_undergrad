
"""
This program is a simple Naive Bayes Classificator.
I coded this program as an exercise for the Chapter "Naive Bayes" in Jurafsky & Martin 
It has simple + 1 Laplace-Smoothing and does not take unknown words into account.
(words that are not in the training set).

It takes training-data and test-document as user specific variables and is 
not properly evaluated and can be regarded as a simple exercise. I tried
to maximizie computing-speed and decrease complexity in comparison to the
pseudo-code from the book.

Malte Sternik
Ruhr-Universität Bochum
27.11.2024
"""

import math

################################
# Functions
################################


def train_sentiment_classifier(training_set:dict):
    """
    Function trains sentiment_classifier with a given training_set\n
    Input (* Optional):\n
    \ttraining_set (dict) -> a dictionary with documents as keys and classes as values\n
    Output:\n
    \tlogprior (dict) -> keys: all classes 'c', values: probabilites for a doc to be assigned class 'c'\n
    \tloglikelihood(dict) -> keys: word 'w' of document-vocabulary 'V', values: probability for 'w' to occur in 'c'\n
    \tC (list) -> set of all classes from training-set\n
    \tV (list) -> vocabulary of all documents
    """
    # initialize variables (cast most as list to escape potential type-handling errors)
    # (set() function collects all uniue elements to an unordered type, important to cast to list)
    D  = list(training_set.keys()) # all documents of the training set
    C = list(set(training_set.values())) # all different classes of the training_set
    V = list(set(" ".join(D).split())) # the vocabulary of ALL documents

    logprior = {}
    bigdoc = {}   
    loglikelihood = {}
    
    # iterate through all unique classes
    for c  in C:
        N_doc = len(D) #number of documents
        N_c = len([d for d in D if training_set[d] == c]) #number of documents with class 'c'

        #store values in dictionaries so that the currently-iterating 'c'
        #is a key and the probalities are stored as values
        logprior[c] = math.log(float(N_c / N_doc)) #probability that a document has the class 'c'
        bigdoc[c] = [d for d in D if training_set[d] == c] #all documents with class 'c'

        #Assign word-counts:
        #(because word counts need to be accessed multiple time later on )
        # -> counts how often a word 'w' occurs in all documents with the current class 'c'
        w_counts = {}
        for w in V: w_counts[w] = "".join(bigdoc[c]).split().count(w) #join all docs together, then count

        # iterate through all unique words to assign likelihood
        for w in V:

            # calculate likelihood for each word 'w'
            # numerator: all occurences of 'w' in docs (with class 'c')
            # denominator: all occurences of ALL words in docs(with class 'c')
            # (see w_counts), adding the length of V equals '+1' for each word (for laplace-smoothing)
            likelihood = math.log(
                w_counts[w] + 1
                 / 
                sum(w_counts.values())+len(V) 
            )
            
            #store each likelihood with respective word 'w' and current class 'c'
            loglikelihood[w,c] = likelihood

    return logprior, loglikelihood,C, V


def argmax(ls:list):
    """
    Returns the index of the max list element (with the highest 
    \nvalue).
    Input (* optional):\n
    \tls (list) -> input list
    Output:\n
    \tindex (int) -> index of argmax
    """
    # init 
    index = 0
    val = ls[0]

    # iterates list with index
    for i, v in enumerate(ls):
        if val < v:
        # if current val higher than all previously iterated elements set index as current argmax
            index = i
            val = v

    return index

def test_sentiment_classifier(test_doc, logprior, loglikelihood, C ,V):
    """
    Classifies document after training set\n
    Input (* Optional):\n
    \ttest_doc (str) -> document to be classified 
    \tlogprior (dict) -> learned probabilites for a doc to be assigned class 'c'\n
    \tloglikelihood(dict) -> learned probabilities for every word 'w' to occur in 'c'\n
    \tC (list) -> set of all classes from training-set\n
    \tV (list) -> vocabulary of all documents
    Output:\n
    \tC[argmax(sumc)] (any type) -> most likely class to be assigned to document (type depending on training-data)'\n
    """
    #stores likelihood-probability for all classes 'c' to be assigned to test_doc
    sumc = []
    
    #iterate through all classes C 
    for c_index, c in enumerate(C):

        #add base-probability (logprior) for class 'c' to be assigned to list
        sumc.extend([logprior[c]])
        
        #iterate through all tokens/words in test-document
        for w in test_doc.split():

            #if the word is known by our training set ... 
            if w in V:
                #... sum up probabilities that each word 'w' occurs in class 'c'
                sumc[c_index] += loglikelihood[w,c] 

    # calculate the class with the highest probability with argmax
    return C[argmax(sumc)]
    

###########################################
# FUNCTION THAT CALLS ALL OTHER FUNCTIONS # 
###########################################

def run_script(D:dict, testdoc:str):
    """
    main function for sentiment classifying
    """

    #unpacking variables from training-function, pass movie_reviews as the training set
    logprior, loglikelihood, C, V = train_sentiment_classifier(training_set =movie_reviews)
    #pass training-results to the classificator
    most_likely_class = test_sentiment_classifier(testdoc, logprior, loglikelihood, C, V)

    # print results
    print(f"Calculating most likely class for: \n'{testdoc}'\nAssigned class: '{most_likely_class}'")
    


################################
# Main program
################################

if __name__ == "__main__":
    # Example training-set (From Jurafsky&Martin S.61)
    # contains movie-reviews, either assigned with positive sentiment ('+') 
    # or negative sentiment ('-')
    movie_reviews = {
        "just plain boring" : "-",
        "entirely predictable and lacks energy" : "-",
        "no surprises and very few laughs" : "-",
        "very powerful" : "+",
        "the most fun film of the summer" : "+"
        }
    
    # Test-string to be classified
    test_document = "predictable with no fun"
    
    # calling main-function and passing arguments
    run_script(D = movie_reviews, testdoc = test_document)



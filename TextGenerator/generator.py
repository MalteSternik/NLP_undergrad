'''
Malte Sternik 19.11.2024

This program generates german text from an user input word. It uses bigram probabilities (nzz_transition_probs.dic, in
this case a small excerpt from german corpora for testing purposes.) to calculate the next respective word through simple
sampling.
'''


import ast
import random

###############
# Functions
###############


def sampling_pick(probabilities:dict, print_trace:bool):
    """
    Returns a pick from a dictionary containing strings as keys and probabilities as values,\n
    as an sampling algorithm\n
    Input (* optional):\n
    \tprobabilities (dict) input dictionary. (values must be floats 0.0-1.0)
    \tprint_trace* (bool) when 'true' print values for error tracing
    Output:\n
    \tpick (str) final pick from the dictionary
    """
    # initialize random float between 0 - 1
    # set border for sampling-algorithm to value of the FIRST element of the dict
    # set default pick for error-tracing
    r = random.uniform(0.0,1.0)
    border = float(next(iter(probabilities.values())))
    pick = "failed"

    # print values if necessary
    if print_trace: print("picking sample (r: ", r, ", border_start: ", border, "dict_len:", len(probabilities), end=" ")

    # Loop through the keys of the dict
    for index, key in enumerate(probabilities.keys()):
        #Check if it is the last loop (last element)
        if index == ( len(probabilities.keys())-1 ): 
            #if so the last border as been surpassed -> the last key must be the pick
            pick = key
            break
        #if the border is bigger than the 
        if r < border: # if the border is bigger than the random number..
            #.. it is the correct key to pick 
            pick = key
            break
        else:
            # if the border is smaller than the random number, increase the bordersize by actual probability
            border += probabilities[key]

    # print post-loop values if needed
    if print_trace: print("border_end: ", border, "final pick: ", pick, "(with prob:",  probabilities[pick],")]")
    # return pick
    return pick

def initialize_word(word_list:list):
    """
    Asks the user to pick an item from a specified list. If it is\n
    not in the list it returns a random item. \n
    Input (* optional):\n
    \tword_list (list) specified list to choose from
    Output:\n
    \tword (str) chosen word
    """
    # ask for user input
    in_word = input("Please enter starting word: ")
    # if input word is in the list pick the word
    if in_word in word_list:
        return in_word
    else:
        #if not return a random word from the list
        print(f"[!] Warning: '{in_word}' not found in corpus. Chose a random word instead.")
        return word_list[random.randint(0,len(word_list))]

def generate(start_word:str,input_dict:dict, iterations = 20, finishing_sequence = "<S>", showlog = False):
    """
    Generates a sentence based on probabilities dictionary and a\n
    specified starting_word. Finishes at specified seuence\n
    Input (* optional):\n
    \tword_list (list) specified list to choose from
    \tword_list (list) specified list to choose from
    \titerations* (int) max word limit for sentence
    \tfinishing_sequence* (str) specified list to choose from
    Output:\n
    \tsentence (str) final generated string sequence
    """

    #print log if needed
    if showlog: print(f"start_word: {start_word} , iterations: {iterations}, finishing_sequence: {finishing_sequence}")
    # initialize sentence and actual word
    sentence, actual_word = start_word, start_word

    
    #Iterate as often as specified
    for i in range(0,iterations):
        # in each iteration make a pick
        pick = sampling_pick(input_dict[actual_word], False)
        # add pick to sentence
        sentence += " " + pick
        # set the actual word to the pick
        actual_word = pick
        # when ShiShthe actual pick is a finishing sequence break the loop
        if pick == finishing_sequence: break

    # return created sentence
    return sentence


#######################################
# Initializing functions 
#######################################

def run_script() -> None:

    # 1. Einlesen der Übergangswahrscheinlichkeiten
    # dict transitions:
    # key: erstes Wort
    # val: dict with key: nächstes Wort, val: Wahrscheinlichkeit
    with open("nzz_transition_probs.dic", encoding="utf-8", mode="r") as f:
        file = f.read()

    transitions = ast.literal_eval(file)
    
    # call and print word generator, call "initialize_word()" to create starting_word
    print(generate(start_word = initialize_word(list(transitions.keys())),iterations=20, input_dict = transitions, showlog=True))




    
###########################################################
# main function
###########################################################

if __name__ == "__main__":

    print("Starting program... " + "\n")
    run_script()
    print("\nFinished.")



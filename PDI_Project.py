# coding: utf-8

import os
import math
from pickle import TRUE
from typing import Text
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

target_enxtension = "txt"
possible_categories = ["matematica", "historia", "algebra", "basedados"]
root_path = 'C:\\Users\\User\\Desktop\\ProjectoPDI'

####################
#   Read / Write   #
####################
def get_all_text_files(file_format="txt"):
    all_files_in_dir = os.listdir(root_path)
    has_correct_extension = lambda x: x[x.rfind('.')+1:] == file_format
    return list(filter(has_correct_extension, all_files_in_dir))

def read_file(filename, file_format="txt"):
    filepath = os.path.join(root_path, filename)
    if file_format == "txt":
        with open(filepath, 'r') as file: return file.read()
    else:
        raise AttributeError("O algoritmo não reconhece o ficheiro, experimente novamente ou tente com um ficheiro diferente")
        
def move_file(filename, category):
    initial_filename = os.path.join(root_path, filename)
    final_filepath = os.path.join(root_path, category)
    if not os.path.exists(final_filepath):
        os.makedirs(final_filepath)
    final_filename = os.path.join(final_filepath, filename)
    os.rename(initial_filename, final_filename)

####################
# Text Formatting  #
####################
def extract_words(text):
    punctuation = string.punctuation
    stop_words = stopwords.words('english')
    words_in_text = nltk.word_tokenize(text)
    is_insteresting_word = lambda word: word not in punctuation and word not in stop_words
    return list(filter(is_insteresting_word, words_in_text))


def extract_sentences(text):
    return nltk.sent_tokenize(text)

####################
#  Word Counting   #
####################
def get_word_count(document):
    word_list = extract_words(document)
    word_list = list(map(lambda x: x.lower(), word_list))
    return Counter(word_list)

####################
#    Auxiliary     #
####################
def filter_interesting_words(word_counts, interesting_words):
    return {word: count for word, count in word_counts.items() if word in interesting_words}

def get_document_category(tfidf_scores, possible_categories):
    if possible_categories is not None:
        tfidf_scores = filter_interesting_words(tfidf_scores,possible_categories)
    return max(tfidf_scores, key=tfidf_scores.get)

####################
#    Variables     #
####################
all_text_files = get_all_text_files(target_enxtension)
documents = list(map(lambda x: read_file(x, target_enxtension), all_text_files))
documents_word_count = list(map(get_word_count, documents))

print (documents_word_count)
print (all_text_files)
i = 0

for counter in documents_word_count:
    encontrou = False
    for letter, count in counter.most_common(3):
        #print (letter, " ",count)
        if not encontrou:
            for cat in possible_categories:
                if cat == letter:
                    category = cat
                    move_file(all_text_files[i], category)
                    i = i+1
                    encontrou = True
                    break

####################
#   TF-IDF SCORES  #
####################
# Iterate over all text files
for filename, word_count in zip(all_text_files, documents_word_count):
    # TF score
    n_terms_in_document = sum(word_count.values())
    tf_scores = {}
    for word, count in word_count.items():
        tf_scores[word] = count / n_terms_in_document
    # IDF score
    total_number_of_documents = len(documents)
    n_documents_with_word = lambda word: sum([word in counter for counter in documents_word_count])
    idf_scores = {}
    for word, count in word_count.items():
        print (word)
        print (count)
        print (n_documents_with_word(word))
        print (total_number_of_documents)
        idf_scores[word] = math.log(total_number_of_documents / n_documents_with_word(word))
    # TF-IDF score
    tfidf_scores = {}
    for word, count in word_count.items():
        tfidf_scores[word] = tf_scores[word] * idf_scores[word]
    # Assign category
    category = get_document_category(tfidf_scores, possible_categories)
    
    

    print (total_number_of_documents)
    print (n_documents_with_word)
    print (idf_scores)
    print (tfidf_scores)
# CSC 575 Group Project
# This file contains classes for terms and documents
import json
import sys
import nltk
import time
import pickle
import json
import time
import re
import sklearn

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer  # added for regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class Book:
    """
    Class representing the book object
    """

    def __init__(self, book_id, book_values, stop_words):
        self.book_id = book_id
        self.title = book_values['title']
        self.author = book_values['author']
        self.release_date = book_values['release_date']
        self.language = book_values['language']
        self.text = book_values['text']
        self.stopwords_to_exclude = stop_words
        # self.text_stemmed = self.get_stems(self.text) fails for sklearn
        # self.text_without_stopwords = None
        # self.text_term_counts = self.process_text()

    def get_stems(self, text_to_stem):
        "Parses documents for stems. Returns list."

        words = nltk.word_tokenize(text_to_stem)
        ps = nltk.PorterStemmer()

        chars = ["\\x90", "\\x91", "\\x92", "\\x93", "\\x94", "_", "\\x80", "\\x99", "â", "\.", "\,", "\(", "\)", "\:",
                 "\;", "\?", "\!", "\-\-", "\'"]

        filtered = []
        for w in words:
            w = w.lower()
            if w not in self.stopwords_to_exclude:
                for char in chars:
                    if re.search(char, w):
                        w = None
                        break

                if w != None:
                    filtered.append(ps.stem(w))

        return filtered

    def remove_stopwords(self, text):
        eng_stopwords = set(stopwords.words('english'))
        text_tokens = word_tokenize(text)
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')  # added for regex
        text_tokens = tokenizer.tokenize(text_tokens)  # added for regex
        stemmer = PorterStemmer()
        text_wo_stopwords = [stemmer.stem(word) for word in text_tokens if word not in eng_stopwords]
        # text_wo_stopwords = [word for word in text_tokens if word not in eng_stopwords]
        return text_wo_stopwords

    def process_text(self):
        # Processes text into a term:term_count dictionary
        term_counts_dict = dict()
        text_tokens = word_tokenize(self.text)
        text_wo_stopwords = [word for word in text_tokens if word not in self.stopwords_to_exclude]
        self.text_without_stopwords = text_wo_stopwords

        for term in text_wo_stopwords:
            if term in term_counts_dict:
                term_counts_dict[term] += 1
            else:
                term_counts_dict[term] = 1

        return term_counts_dict


class BookManager:

    def __init__(self, json_file):
        self.json_file = json_file
        self.stop_words = self.set_stop_words()
        self.books = self.process_file_to_books()
        self.book_id_list, self.book_text_list = self.book_text_lists()  # format for vectorizer.
        # Can change the text to meet our needs (stemming, stopwords, etc).
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)  # tfidf vectorizer
        self.book_tfidf_vector = self.vectorizer.fit_transform(self.book_text_list)

    def book_text_lists(self):
        book_ids = []
        books_text = []
        for book in self.books:
            book_ids.append(book.book_id)
            books_text.append(book.text)

        return book_ids, books_text

    def set_stop_words(self):
        eng_stopwords = set(stopwords.words('english'))
        return eng_stopwords

    def get_books(self):
        return self.books

    def process_file_to_books(self):
        # Processes the json file into a list of Books
        books_in_file = list()

        for book_id, book_values in self.json_file.items():
            new_book = Book(book_id, book_values, self.stop_words)
            books_in_file.append(new_book)

        return books_in_file

    def query_similarity(self, query, similarity_metric):

        query_vectorized = self.vectorizer.transform(query)
        metric_name = ""
        similarity_metric = int(similarity_metric)
        result_dict = dict()

        if similarity_metric == 0:  # Cosine Similarity
            metric_name = "Cosine Similarity"
            cos_sim_query = cosine_similarity(query_vectorized, self.book_tfidf_vector)
            # cos_sim_query = euclidean_distances(query_vectorized, self.book_tfidf_vector)
            greatest_sim = 0
            greatest_index = 0
            for index, sim in enumerate(cos_sim_query[0]):
                query_book_id = self.book_id_list[index]
                result_dict[query_book_id] = sim
                if sim > greatest_sim:
                    greatest_sim = sim
                    greatest_index = index
            result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
        elif similarity_metric == 1:  # Euclidean Distance
            metric_name = "Euclidean Distance"
            euc_sim_query = euclidean_distances(query_vectorized, self.book_tfidf_vector)
            greatest_sim = sys.maxsize
            greatest_index = 0
            for index, sim in enumerate(euc_sim_query[0]):
                query_book_id = self.book_id_list[index]
                result_dict[query_book_id] = sim
                if sim < greatest_sim:
                    greatest_sim = sim
                    greatest_index = index
            result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=False))

        query_book_id = self.book_id_list[greatest_index]
        for book in self.books:
            if book.book_id == query_book_id:
                return book, greatest_sim, metric_name, result_dict



if __name__ == '__main__':
    start_time = time.time()
    try:
        books_in_file = pickle.load(open("BookManager.pickle", "rb"))
    except:
        print("Done with NLTK, starting application.\n")
        json_book_file = json.load(open("books.json"))
        print("Creating BookManager")
        books_in_file = BookManager(json_book_file)
        print("Creating Term Manager")

    pickle.dump(books_in_file, open("BookManager.pickle", "wb"))

    print("--- %s seconds ---" % (time.time() - start_time))
    while True:
        test_user_query = [str(input("What is your query?\n"))]
        sim_book, sim_value, metric_name = books_in_file.query_similarity(test_user_query, 0)


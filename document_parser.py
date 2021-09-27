# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:13:45 2021
@author: jimmy
"""

from scraper_process import generate_sample
import nltk as nltk
import pandas as pd
import math
import json
import re

def generate_histogram(terms):
    "Generates a histogram of the term frequency distribution."

    import matplotlib.pyplot as plt

    d = {}
    for term in terms:
        key = terms[term]['doc frequency']
        if key not in d.keys():
            d[key] = 1
        else:
            d[key] += 1

    inv_map = {v: k for k, v in d.items()}

    plt.figure()
    plt.title(f"Trimmed 2.5% of Lower Frequencies. Num Terms = {len(terms)}")
    plt.hist(inv_map, bins = 30)

def get_stems(d):
    "Parses documents for stems. Returns list."

    words = nltk.word_tokenize(d)
    stopWords = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()

    chars = ["\\x90", "\\x91", "\\x92", "\\x93", "\\x94", "_", "\\x80", "\\x99", "â", "\.", "\,", "\(", "\)", "\:", "\;", "\?", "\!", "\-\-", "\'"]
    filtered = []
    for w in words[1000:2000]:
        if w.lower() not in stopWords:
            for char in chars:
                if re.search(char, w):
                    w = None
                    break

            if w != None:
                filtered.append(ps.stem(w.lower()))

    return filtered

def parse_documents(books):
    "Gets term frequencies. Returns dict."

    print(f"{len(books)} Books to Parse\n")

    i = 0
    dFinal = {}
    for key in books:
        stems = get_stems(books.get(key).get('text'))
        for stem in stems:
            stem = stem.lower()
            if stem not in dFinal.keys():
                dFinal[stem] = {'total frequency': 1,
                                  'doc frequency': 1,
                                  'doc occurance': {key:1}}
            else:
                dFinal[stem]['total frequency'] += 1
                if key in dFinal[stem]['doc occurance'].keys():
                    dFinal[stem]['doc occurance'][key] += 1
                else:
                    dFinal[stem]['doc occurance'].update({key:1})
                    dFinal[stem]['doc frequency'] += 1

        i += 1
        if i % 10 == 0:
            print(f"{i} Books Parsed")

    print("***** End of Document Parsing *****")
    return dFinal

def trim_terms(terms, books, hPerc = .8, lPerc = 0.025):
    "Trims terms from the data based on how frequenly they appear. Returns dict"

    N = len(books)
    high = hPerc * N
    low = lPerc * N

    for term in list(terms.keys()):
        freq = terms[term]['doc frequency']
        if (freq >= high) or (freq < low):
            del terms[term]

    return terms

def get_idf(terms, books):

    for term in terms:
        tf_idf = compute_tfidf(term, terms, books)
        terms[term]['tf_idf'] = tf_idf

    return terms

def compute_tfidf(term, terms, books):

    N = len(books)
    n = len(terms[term]['doc occurance'])
    idf = math.log(N/n)
    tf_idf = terms[term]['total frequency'] * idf

    return round(tf_idf, 4)

def process_query(query):
	Q = query.split( )
	l = []
	ps = nltk.PorterStemmer()
	stopWords = set(nltk.corpus.stopwords.words('english'))
	for word in Q:
		if word not in stopWords:
			stem = ps.stem(word) # remove stop words
			l.append(stem)
	return l

def return_query(query, dFinal):
	"An vector space model for an inverted index retrevial algorithim. Returns a dict."

	Q = process_query(query)
	sumSQ = 0
	R = {}
	for T in Q:

		if T not in dFinal.keys():
			continue

		I = dFinal[T]['tf_idf']
		K = dFinal[T]['total frequency']
		W = K * I
		L = dFinal[T]['doc occurance']
		for D in L.keys():
			C = L[D]
			if D in R.keys():
				R[D] += round(W*I*C, 3)
			else:
				R[D] = round(W*I*C, 3)

		sumSQ += W**2

		docLengths = {}
		for D in R.keys():
			if D in docLengths.keys():
				docLengths[D] += (I*C)**2
			else:
				docLengths[D] = (I*C)**2

		for D in docLengths.keys():
			docLengths[D] = docLengths[D]**0.5

	L = sumSQ**0.5
	for D in R.keys():
		S = R[D]
		Y = docLengths[D]
		R[D] = S/(L*Y)

	return R

def get_document_frequencies():
	"Parses three documents for term frequencies. Returns dict."

	print("***** Beginning to Parse Documents for Term Frequencies *****")

	books = json.load(open("sample_books.json"))
	terms = parse_documents(books)
	terms = trim_terms(terms, books, lPerc = .025)
	terms = get_idf(terms, books)

	return terms, books

def sort_rankings(rankings):
	"Sorts the rankings and returns a dict."

	dfRank = pd.DataFrame(list(rankings.values()), index = rankings.keys(), columns = ['Score'])
	dRank = dfRank.nlargest(10, columns = "Score").to_dict()
	dRank = dRank['Score']

	return dRank

def print_results(dRank, books, query):
	"Prints the results of the vector space retreival algorithim."

	print(f'\nThe Most Relevant Books to Your Query:\n"{query}"\n')
	for doc in dRank.keys():
		title = books[doc]['title']
		author = books[doc]['author']
		score = dRank[doc]

		print(f"Title: {title.title()}")
		print(f"Author: {author}")
		print(f"Similarity Score: {round(score, 3)}\n\n")

def get_query():
	"Gets and returns results of query"

	query = input("What is your query?")
	rankings = return_query(query, terms)
	dRank = sort_rankings(rankings)
	print_results(dRank, books, query)

def model_main():
	generate_sample(100)
	terms, books = get_document_frequencies()
	get_query()

if __name__ == "__main__":
	model_main()
import pandas as pd
import nltk
from nltk.corpus import gutenberg as gt
import gensim
import logging

#Training the word2vec model on pre-exisitng gutenburg corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = gt.sents()
model = gensim.models.Word2Vec(sentences, min_count=1)
model.most_similar('books',topn=5)
##End Model


#When a user gives a query we will also append the top 2 similar words to the query.

def newCosineScore(q,queryVector):
    scores=[0 for x in xrange(N)]  #scores is a list of length N and accumulates the score of each
                                   #document (initialize to zero)
    
    for term in q:                  
        if term in idf.keys(): #walk through the posting list of each term in the query that is present in the corpus 
            posLis=idf[term]
            for doc in posLis:
                docIndex=gt.fileids().index(doc)
                termIndex=idf.keys().index(term)
                scores[docIndex]+=U[termIndex,docIndex]*queryVector[termIndex]
            similarTerms=model.most_similar(term)[0:3]
            for simTerm in similarTerms: #also add weights to the words similar to the query vectors
                posLis=idf[simTerm[0]]
                for doc in posLis:
                    docIndex=gt.fileids().index(doc)
                    termIndex=idf.keys().index(simTerm[0])
                    scores[docIndex]+=(U[termIndex,docIndex]*queryVector[termIndex])
                                          #Note that we have added the weight to be half of tfidf weight
                
    SCORES=scores
    #sort the scores list and return the last three docs
    maxLis=sorted(SCORES)               
    maxSim=maxLis[-1]
    Maxindex=scores.index(maxSim)
    doc=gt.fileids()[Maxindex]     #most similar doc
    
    maxSim2=maxLis[-2]
    Maxindex=scores.index(maxSim2)
    doc1=gt.fileids()[Maxindex]    #second most similar doc
    
    maxSim3=maxLis[-3]
    Maxindex=scores.index(maxSim3)
    doc2=gt.fileids()[Maxindex]    #third most similar doc
    return [doc, doc1, doc2]
#preprocessing of query and conversion into vector

query = user_query

toks=nltk.word_tokenize(query)
for i in range(len(toks)):
    toks[i]=toks[i].lower()         #tokenize and normalize the query in the same way as the corpus
    




V=len(idf.keys())
qVector=np.zeros(V)             #Old query vector containing non zero dimensions 
                                #...only for words in the query
for term in toks:
    if term in idf.keys():
        qVector[idf.keys().index(term)]=idfs[term]
qVector=unitNorm(qVector)
    
#the new query vector qVector1 with appended dimensions of similar terms
qVector1=np.zeros(V)             
for term in toks:
    if term in idf.keys():
        qVector1[idf.keys().index(term)]=idfs[term]
        
for term in toks:
    if term in idf.keys():
        lis=model.most_similar(term)[0:3]
        for item in lis:
            qVector1[idf.keys().index(item[0])]=idfs[item[0]]/2 #halving weights of added terms                    
qVector1=unitNorm(qVector1)


lis=newCosineScore(toks,qVector1) 

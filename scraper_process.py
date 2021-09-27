# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:06:27 2021

"""

import requests
import regex as re
import json
from random import sample

def get_title(raw_text):
	"Scrapes book for title. Returns string."

	match = re.search('Title: .*', raw_text)
	title = match.group(0).strip()[7:].strip().lower()
	return title

def get_range(raw_text, pattern):
	"Gets the pattern range. Returns list of tuples."
	return [(m.start(0), m.end(0)) for m in re.finditer(pattern, raw_text)]

def get_book(raw_text):
	"Returns the full ebook without any licensing. Returns string"

	start = f"START OF (THE|THIS) PROJECT GUTENBERG EBOOK(.*|.*\n.*)\*\*\*"
	end = f"END OF (THE|THIS) PROJECT GUTENBERG EBOOK(.*|.*\n.*)\*\*\*"

	start_range = get_range(raw_text, start)
	end_range = get_range(raw_text, end)
	text = raw_text[start_range[0][1]:end_range[0][0]]
	return text

def scrape_book(raw_text, book_id):
	"Scrapes books for content. Returns dict."

	title = get_title(raw_text)
	book = get_book(raw_text)
	author = get_author(raw_text)
	release_date = get_release_date(raw_text)
	language = get_language(raw_text)

	d = {book_id: {'title':title,
				  'author':author,
				  'release_date':release_date,
				  'language':language,
				  'text':book}}
	return d

def get_author(raw_text):
	"Scrapes for book's author. Returns string."

	match = re.search('Author: .*', raw_text)
	author = match.group(0).strip()[8:]
	return author

def get_release_date(raw_text):
	"Scrapes for book's release date. Returns string."

	match = re.search('Release Date: .*', raw_text)
	date = match.group(0).strip()[14:]
	book_id = get_range(date, '\[.*\]')

	if date != []: # Removes formatting in release date: Ex. [EBook #500]
		date = date[:book_id[0][0]].strip()

	return date

def get_language(raw_text):
	"Scrapes for book's language. Returns string."

	match = re.search('Language: .*', raw_text)
	language = match.group(0).strip()[10:]

	return language

def generate_sample(sample_size):
	"Reads books.json and writes a sample"

	json_book_file = json.load(open("books.json"))
	keys = sample(list(json_book_file), sample_size)

	sampleDict = {}
	for key in keys:
		sampleDict[key] = json_book_file[key]

	with open('sample_books.json', 'w') as f:
	    json.dump(sampleDict, f)

def scraper_process():
	"Scrapes and parses the Project Gutenberg website for book data. Writes json file."

	print("***** Beginning Scraper Process *****")

	books = {}
	j = 0
	fail = 0
	for i in range(10,10000):

		book_id = str(i) #84 100 1342
		url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
		r = requests.get(url)
		if r.status_code == 200:

			try:
				d = scrape_book(r.text, i)
				books.update(d)
			except:
				fail += 1

			j += 1
			if j % 25 == 0:
				print(f"{str(j)} books scraped")
		else:
			continue

	print(f"Failed Scrapes: {str(fail)}")

	with open('books.json', 'w') as f:
	    json.dump(books, f)

	print("***** End Scraper Process *****")

if __name__ == "__main__":
	scraper_process()
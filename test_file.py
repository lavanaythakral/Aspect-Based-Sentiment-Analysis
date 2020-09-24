import aspect_based_sentiment_analysis as absa
import pandas as pd
import time
import re 
import csv
import logging
import os

def find_sentence(word,test_sentences):
	print("Finding sentences")
	sentence =[]
	for sen in test_sentences:
		if(word in sen.lower()):
			sentence.append(sen.lower())
	print("Sentences found\n")
	print("----------------------")
	return sentence

def sentim(word,test_sentences,nlp):
	sentences = find_sentence(word,test_sentences)
	print("Calculating sentiments")
	print("----------------------")
	all_sentiments = []
	for sen in sentences:
		print(sen)
		sentiment = nlp(sen, aspects=[word])
		all_sentiments.append(sentiment.subtasks[word].sentiment.value)
		# sentiment = 0
		# all_sentiments.append(0)
	print("----------------------")
	print("Sentiments calculated\n")
	return all_sentiments

def make_row(word,all_sentiments):
	print("Making rows")
	cnt_pos = 0
	cnt_neg = 0
	cnt_neu = 0
	for sen in all_sentiments:
		if(sen == 0):
			cnt_neu = cnt_neu+1
		elif(sen == 1):
			cnt_neg = cnt_neg+1
		elif(sen == 2):
			cnt_pos = cnt_pos+1

	row = [word,cnt_pos,cnt_neu,cnt_neg]
	return row

def main():
	nlp = absa.load()
	print("Model Loaded")
	df = pd.read_csv('/home/luv/Downloads/Thesis project-20200829T070727Z-001/Thesis project/datasets/WOF_split_into_sentences.csv')
	test_sentences = list(df['Sentences'])
	keys = ['time','stage','minister','research','defence','slv','missiles','launch','technology','work','rocket','rameswaram','sarabhai','development','project','space','brahm']
	with open('/home/luv/Aspect-Based-Sentiment-Analysis/Custom/Results.csv',"w") as f:
		writer = csv.writer(f)
		writer.writerow(['word','Pos','Neu','Neg'])
		for key in keys:
			print("Using key : ",key)
			all_sentiments = sentim(key,test_sentences,nlp)
			row = make_row(key,all_sentiments)
			writer.writerow(row)	

	df = pd.read_csv('/home/luv/Aspect-Based-Sentiment-Analysis/Custom/Results.csv')
	print(df)

if __name__ == "__main__":
	main()
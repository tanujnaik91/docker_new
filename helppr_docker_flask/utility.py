from boilerpy3 import extractors
import spacy
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(nlp.create_pipe('sentencizer'))
REGEX_FOR_DOT = "."
REGEX_FOR_QUESTION_MARK = "?"

def extractTextFromURLNumWordsRules(url):
    extractor = extractors.NumWordsRulesExtractor()
    data = extractor.get_doc_from_url(url)
    return data

def extractTextFromContentNumWordsRules(text):
    extractor = extractors.NumWordsRulesExtractor()
    try:
        text = text.replace('\n', ' ').replace('\r', '')
        document = extractor.get_doc(text)
        #data = document.get_text(True, False)
    except:
        document = None
    return document


def removeSingleNoiseFromText(text, noise_keywords, sentence_length=4):
  doc = nlp(text)
  sent_list = []
  token_length = []
  for sent in doc.sents:
    sentence = sent.text
    if sentence.endswith(REGEX_FOR_DOT or REGEX_FOR_QUESTION_MARK):
      #print('new_sent', sentence)
      token = sentence.split()
      if len(token)>sentence_length:
        #print('new_sent', sentence)
        sent_list.append(sentence)
  para = ' '.join(sent_list)
  #keyword = '📣 The Indian Express is now on Telegram.'
  para = para.split(noise_keywords)[0]
  return para


def getFacts(text):
  doc = nlp(text)
  fact_list = []
  for sent in doc.sents:
    sentence = sent.text
    if bool(re.search(r'\d+', sentence))== True:
      fact_list.append(sentence)
  #facts = ' '.join(fact_list)
  return fact_list

##give HTML content here.
def getImage(html_text):
  soup = BeautifulSoup(html_text)
  metas = soup.find_all('meta')
  image = soup.find("meta",  property="og:image").attrs['content']
  return image

def pre_process(html_text):
  document = extractTextFromContentNumWordsRules(html_text)
  try:
      without_html = document.content
      title = document.title
      content = removeSingleNoiseFromText(without_html,INDIANEXPRESS_NOISE, 4)
      facts = getFacts(content)
      image = getImage(html_text)
  except:
    pass
  return  content , title, facts, image

def splitByDelimiter(x,text):
    return text.split(x)

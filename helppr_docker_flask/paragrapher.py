from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from beautifier import *

def break_into_sentences(txt):
    sentencizer = Sentencizer()
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    doc = nlp(txt)
    span_list = list(doc.sents)
    sentence_list = [t.text for t in span_list]
    return sentence_list

SENTENCE_MAX_LENGTH_WORDS = 10
TOTAL_WORDS_MAX_LENGTH = 40

#this is still not used.
MAX_SENTENCES_PER_PARAGRAPH = 3

def generate_distribution(my_list):
    list_size = len(my_list)
    distribution_dict = {}
    position = 0
    for l in my_list:
        nwc, ncc = get_sentence_distribution(l)
        distribution_dict[position] = [nwc,ncc]
        position = position +1
    return distribution_dict, list_size


def get_sentence_distribution(txt):
    word_array = txt.split(" ")
    naive_word_count = len(word_array)
    naive_char_count = []
    for word in word_array:
        naive_char_count.append(len(word))

    return naive_word_count, naive_char_count


def add_new_line(txt,**kwargs):
     return txt + "<br><br>"

def generate_paragraphs(text):
    my_list = break_into_sentences(text)
    distribution_dict, list_size = generate_distribution(my_list)

    for key in distribution_dict:
      if key < list_size-1:

        if distribution_dict[key][0] >= SENTENCE_MAX_LENGTH_WORDS or sum(distribution_dict[key][1]) >= TOTAL_WORDS_MAX_LENGTH:
            my_list[key] = add_new_line(my_list[key])

        links = check_if_hyperlink_exists(my_list[key])

        if links is not None:
            for link in links:
                my_list[key] = add_anchor_tag(my_list[key],link)

    return ' '.join(my_list)

def capitalise_first_sentence(text):
    text_list = break_into_sentences(text)
    finals = []
    for entry in text_list:
        finals.append(entry.capitalize())
    return ' '.join(finals)

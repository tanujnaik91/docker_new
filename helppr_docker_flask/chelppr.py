from flask import Flask, request
from boilerpy3 import extractors
from gensim.summarization import summarize
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration,T5Tokenizer
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from celery import Celery
from dotenv import load_dotenv
from googleapiclient import discovery, errors
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from paragrapher import generate_paragraphs
from absummary import custom_summarize
from beautifier import preface_output_merger
import time
from utility import *
from pre_process import pre_process
# from flask_crontab import Crontab

#from waitress import serve     #for WSGI and threading when in production

app = Flask(__name__)
load_dotenv('.flaskenv')
INDIANEXPRESS_NOISE = '📣 The Indian Express is now on Telegram.'
#bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', output_past=True)
#bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', output_past=True)
#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
youtube = discovery.build('youtube', 'v3', developerKey=os.environ['GOOGLE_API_KEY'])
device = "cpu"

def get_env_variable(name):
    try:
        return os.environ[name]
    except KeyError:
        message = "Expected environment variable '{}' not set.".format(name)
        raise Exception(message)

#setting the values from the environment file
POSTGRES_URL = get_env_variable("POSTGRES_URL")
POSTGRES_USER = get_env_variable("POSTGRES_USER")
POSTGRES_PW = get_env_variable("POSTGRES_PW")
POSTGRES_DB = get_env_variable("POSTGRES_DB")


#Now let us set the db engine
DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # silence the deprecation warning

db = SQLAlchemy(app)
db.Model.metadata.reflect(db.engine)

class SummarisePage(db.Model):
    __table__ = db.Table('mycore_summarisepage', db.Model.metadata,
            # You'll need to "override" the timestamp fields to get flask-sqlalchemy to update them automatically on creation and modification respectively.
            # db.Column('created_timestamp', db.DateTime, default=datetime.datetime.now),
            # db.Column('modified_timestamp', db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now),
            autoload=True,
            extend_existing=True,
            autoload_with=db.engine,
        )
    # You need to do model relationships manually in SQLAlchemy...
    # related = db.relationship(SomeOtherFlaskSQLAlchemyModel, backref=db.backref('related_backref', lazy='dynamic'))

class ParentPage(db.Model):
    __table__ = db.Table('mycore_parentpage', db.Model.metadata,
            # You'll need to "override" the timestamp fields to get flask-sqlalchemy to update them automatically on creation and modification respectively.
            # db.Column('created_timestamp', db.DateTime, default=datetime.datetime.now),
            # db.Column('modified_timestamp', db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now),
            autoload=True,
            extend_existing=True,
            autoload_with=db.engine,
        )

class Simplify(db.Model):
    __table__ = db.Table('mycore_simplifyselection', db.Model.metadata,
            # You'll need to "override" the timestamp fields to get flask-sqlalchemy to update them automatically on creation and modification respectively.
            # db.Column('created_timestamp', db.DateTime, default=datetime.datetime.now),
            # db.Column('modified_timestamp', db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now),
            autoload=True,
            extend_existing=True,
            autoload_with=db.engine,
        )

class Explain(db.Model):
    __table__ = db.Table('mycore_explainselection', db.Model.metadata,
            # You'll need to "override" the timestamp fields to get flask-sqlalchemy to update them automatically on creation and modification respectively.
            # db.Column('created_timestamp', db.DateTime, default=datetime.datetime.now),
            # db.Column('modified_timestamp', db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now),
            autoload=True,
            extend_existing=True,
            autoload_with=db.engine,
        )
#class Page_facts(db.Model):
#    __table__ = db.Table('mycore_pagefacts', db.Model.metadata,
#    autoload=True,
#    extend_existing=True,
#    autoload_with=db.engine,
    #id = Column(Integer, primary_key=True)
    #page_id = Column(uuid, ForeignKey('mycore_parentpage.id')
    #facts = relationship("mycore_parentpage")
    #)

def extractTextFromURL(url1):
    extractor = extractors.ArticleExtractor()
    extractedText = extractor.get_content_from_url(url1)
    return extractedText

def extractText(text):
    extractor = extractors.ArticleExtractor()
    try:
        text = text.replace('\n', ' ').replace('\r', '')
        document = extractor.get_doc(text)
        extractedText = document.get_text(True, False)
    except:
        extractedText = text
    return extractedText

def remove_duplicates(txt):

    #first check via "\n" to remove duplicates
    new_txt = unique(txt.split("\n"))

    sentencizer = Sentencizer()
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    doc = nlp(new_txt)
    span_list = list(doc.sents)
    sentence_list = [t.text for t in span_list]
    summary = unique(sentence_list)
    return summary

def unique(sequence):
    seen = set()
    q =  [x for x in sequence if not (x in seen or seen.add(x))]
    finalString = ' '.join(q)
    print(finalString,"\n")
    return finalString

def gensimSum(text):
    try:
        total_words = text.split()
        if len(total_words) < 200:
            wc = min(len(total_words),50)
        else:
            wc= 200
        summary = remove_duplicates(summarize(text, word_count=wc))
    except:
        summary = ""
    return summary

# def summary_t5(text):
#     try:
#         text = str(text)
#         preprocess_text = text.strip().replace("\n","")
#         t5_prepared_Text = "summarize: "+ preprocess_text
#         device = "cpu"
#
#         tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)
#
#         # summmarize
#
#         summary_ids = model.generate(tokenized_text,
#                                               num_beams=4,
#                                               no_repeat_ngram_size=2,
#                                               min_length=75,
#                                               max_length=500,
#                                               early_stopping=True)
#
#         output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     except:
#         output = ""
#     return output


def videoSearch(q, salt="Explain"):
    request = youtube.search().list(
        part="snippet",
        maxResults=1,
        q=q + salt
    )
    response = request.execute()

    try:
        videoId = response['items'][0]['id']['videoId']
    except:
        return "https://www.youtube.com/watch?v=sDP3SDaSf4c"

    return "https://www.youtube.com/watch?v="+videoId

def query_video_helppr(q):
    print(q)
    chunks = q.split(" ")
    if len(chunks) < 4:
        pass
    else:
        q = summarize(q)
    return q


@app.route('/')
def index():
    return 'Hello'

#@app.route('/pre_process')
def pre_process(html_text):
    document = extractTextFromContentNumWordsRules(html_text)
    try:
        without_html = document.content
        # Extracting title from extracted but not processed text
        title = document.title
        # removing noise from text
        content = removeSingleNoiseFromText(without_html,INDIANEXPRESS_NOISE, 4)
        # collecting data with numbers in a list
        facts = getFacts(content)
        # extracting image link from html text
        image = getImage(html_text)
    except:
        pass
    return content , image, facts, title

# Main function and expects a url as input
# @app.route('/x', methods=["GET"])
def markasSummaryPending(id):
    #SummarisePage.query.filter_by(id = id)
    sp = SummarisePage.query.filter_by(id = id).first()
    sp.status = 1
    db.session.commit()
    try:
        generateSummary(sp)
    except:# Exception as e:
        sp.status = 3
        #print("Hello")
        #print(e, sp.id)
        #app.logger.info(e + "/n For: /n"+sp)

# Main function and expects a url as input
# @app.route('/y', methods=["GET"])
def generateSummary(sp):
    try:
        page_id = sp.page_id
        page = ParentPage.query.get(page_id)
        format = page.format
        if format== "html":
          content = page.page_content
          print(content)

        elif format== "pdf":
          print(content)

        elif format== "png":
          print(content)

        else:
          None


        try:
            #modifying mycore_parentpage by adding data to columns named page_content, image_link, facts, title
            page.page_content, page.image_link, page.facts, page.title = pre_process(html_content)
        except:
            pass
        article_text = page.page_content
        gensimOut = gensimSum(article_text)
        out = gensimOut
        if out == "":
            sp.status = 3
        else:
            preface = custom_summarize(out,5,200)
            out = generate_paragraphs(out)
            out = preface_output_merger(preface,out)
            sp.status = 2
        sp.data = out

    except Exception as e:
        sp.status = 3

    db.session.commit()

@app.route('/wakeupcall', methods=["GET"])
def wakeSummarize():
    id = request.args.get('q')
    markasSummaryPending(id)
    return 'ok'

@app.route('/simplify', methods=["GET"])
def simplify():
    #first pick ALL ROWS and mark them pending
    Simplify.query.filter_by(status = 0).update(dict(status = 1))
    db.session.commit()

    #Now let us pick All Rows to Simplify
    sps = Simplify.query.filter_by(status = 1).all()
    for sp in sps:
        selection = sp.page_selection
        out = gensimSum(selection)
        if out == "":
            sp.status = 3
        else:
            sp.status = 2
        sp.data = out
        db.session.commit()
    return "ok"

@app.route('/explain', methods=["GET"])
def explain():
    #first pick ALL ROWS and mark them pending
    Explain.query.filter_by(status = 0).update(dict(status = 1))
    db.session.commit()

    #Now let us pick All Rows to Simplify
    sps = Explain.query.filter_by(status = 1).all()
    for sp in sps:
        selection = sp.page_selection
        out = videoSearch(selection)
        if out == "":
            sp.status = 3
        else:
            sp.status = 2
        sp.data = out
        db.session.commit()
    return "ok"

@app.route('/rest', methods=["GET"])
def T():
    q = "Recently, neural network trained language models, such as ULMFIT, BERT, and GPT-2, have been remarkably successful when transferred to other natural language processing tasks. As such, there's been growing interest in language models. Traditionally, language model performance is measured by perplexity, cross entropy, and bits-per-character (BPC). As language models are increasingly being used as pre-trained models for other NLP tasks, they are often also evaluated based on how well they perform on downstream tasks. The GLUE benchmark score is one example of broader, multi-task evaluation for language models.Counterintuitively, having more metrics actually makes it harder to compare language models, especially as indicators of how well a language model will perform on a specific downstream task are often unreliable. One of my favorite interview questions is to ask candidates to explain perplexity or the difference between cross entropy and BPC. While almost everyone is familiar with these metrics, there is no consensus: the candidates’ answers differ wildly from each other, if they answer at all. One point of confusion is that language models generally aim to minimize perplexity, but what is the lower bound on perplexity that we can get since we are unable to get a perplexity of zero? If we don’t know the optimal value, how do we know how good our language model is? Moreover, unlike metrics such as accuracy where it is a certainty that 90% accuracy is superior to 60% accuracy on the same test set regardless of how the two models were trained, arguing that a model’s perplexity is smaller than that of another does not signify a great deal unless we know how the text is pre-processed, the vocabulary size, the context length, etc. For instance, while perplexity for a language model at character-level can be much smaller than perplexity of another model at word-level, it does not mean the character-level language model is better than that of the word-level. Therefore, how do we compare the performance of different language models that use different sets of symbols? Or should we? Despite the presence of these downstream evaluation benchmarks, traditional intrinsic metrics are, nevertheless, extremely useful during the process of training the language model itself. In this article, we will focus on those intrinsic metrics. We will accomplish this by going over what those metrics mean, exploring the relationships among them, establishing mathematical and empirical bounds for those metrics, and suggesting best practices with regards to how to report them."
    print(len(q.split()))
    st = time.time()
    preface = custom_summarize(q,5,200)
    out = generate_paragraphs(q)
    output = preface_output_merger(preface,out)
    app.logger.info("Logger" + " Sdo")
    print(time.time() - st," : is the time taken")
    return output


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='80')
    #serve(app, host='0.0.0.0', port=5000)

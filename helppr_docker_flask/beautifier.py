from regexer import RegexType
import re

def preface_output_merger(preface, out):
    IDEA_IN_BREIF= "<b><u>Idea in Brief</u></b><br><br>"
    preface = preface +"<br><br>"
    FACTS_FROM_ARTICLE = "<b><u>Main Points from Article</u></b><br><br>"

    return IDEA_IN_BREIF + preface + FACTS_FROM_ARTICLE + out

def check_if_hyperlink_exists(text):
    print(text)
    regex = RegexType.REGEX_TO_EXTRACT_LINK.value
    if (len(re.findall(regex, text))) <= 0:
        return None
    links = re.findall(regex, text)
    print(regex)
    return links

def add_anchor_tag(text, link):
    return text.replace(link,"<br><br><a href='"+link+"'>"+link+"</a><br><br>")

def add_new_line(txt,**kwargs):
     return txt + "<br><br>"
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

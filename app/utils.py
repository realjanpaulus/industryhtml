import re
from typing import Dict, List, Optional, Tuple, Union
from unicodedata import normalize

import lxml
from lxml.html.clean import Cleaner
from lxml.cssselect import CSSSelector
from lxml import html, etree
import numpy as np
import pandas as pd

import syntok.segmenter as segmenter
from flair.data import Sentence
from flair.models import SequenceTagger

import logging
logging.getLogger("flair").setLevel(logging.ERROR)

try:
    TAGGER = SequenceTagger.load("pos-multi-fast")
    TOKEN_PATTERN = re.compile(r"\w+")
except:
    print("Loading of Sequence Tagger Model failed!")

def filter_nouns(text: str):
    tokens = TOKEN_PATTERN.findall(text)
    sentence = Sentence(tokens, use_tokenizer=False)
    TAGGER.predict(sentence)
    return " ".join(filter_tokens(sentence))
    
def filter_tokens(sentence: Sentence) -> List[str]:
    nouns = [token.text for token in sentence if token.get_tag("upos").value == "NOUN"]
    return [
        token.lower() for token in nouns if len(token) > 1 and not any(c.isdigit() for c in token)
    ]


# ================ #
# HTML & XML utils #
# ================ #

def clean_boilerplate(
    string: str,
    url: str,
    cleaner: Optional[lxml.html.clean.Cleaner] = None,
    markup_type: Optional[str] = None,
) -> str:
    """Cleans boilerplate tags, attributes etc. from a valid website as string
        (HTML, XHTML or XML). The markup type will be detected.
        A lxml html Cleaner will be set. If the detection fails, another cleaning
        attempt will be made by setting another markup type.

    Parameters
    ----------
    string : str
        String which contains the HTML, XHTML or XML.
    url : str
        String which contains the websites URL.
    markup_type : str, default=None
        Indicate the markup type ('xml' or another).

    Returns
    -------
    clean : str
        Cleaned HTML, XHTML or XML string.
    """

    if not cleaner:
        cleaner = Cleaner(
            scripts=True,
            javascript=True,
            comments=True,
            style=True,
            inline_style=True,
            links=True,
            meta=True,
            page_structure=False,
            processing_instructions=True,
            embedded=True,
            frames=True,
            forms=True,
            annoying_tags=True,
            remove_tags=None,
            allow_tags=None,
            kill_tags=["img"],
            remove_unknown_tags=True,
            safe_attrs_only=True,
            safe_attrs=[],  # No Attribute saving
            add_nofollow=False,
            host_whitelist=(),
            whitelist_tags=set(["embed", "iframe"]),
        )

    string = string[string.find("<") :]

    if not markup_type:
        markup_type = detect_XML(string)
    try:
        clean = clean_website(string, cleaner, markup_type)
    except:
        # try the other (not detected) website type
        try:
            if markup_type == "xml":
                clean = clean_website(string, cleaner, "html")
            else:
                clean = clean_website(string, cleaner, "xml")
        except:
            clean = string
            print(
                f"Website '{url}' couldn't be cleaned (variable type: {type(string)})."
            )

    return clean


def clean_string(string: str):
    """ Cleans the following things from string:
        - new line '\n'
        - carriage return '\r'
        - tab '\t'
        - duplicate whitespace
        - special characters
    """
    string = string.replace("\n", "")
    string = string.replace("\r", "")
    string = string.replace("\t", "")
    string = reduce_whitespace(string)
    string = remove_special_characters(string)

    return string


def clean_website(
    string: str, cleaner: lxml.html.clean.Cleaner, markup_type: str
) -> str:
    """ Cleans boilerplate tags, attributes etc. from a valid website as string
        (HTML, XHTML or XML). A lxml html Cleaner has to be passed.

    Parameters
    ----------
    string : str
        String which contains the HTML, XHTML or XML.
    cleaner : str
        String which sets the lxml html Cleaner object.
    markup_type : str, default=None
        Indicate the markup type ('xml' or another).

    Returns
    -------
    clean : str
        Cleaned HTML, XHTML or XML string.
    """
    tree = extract_tree(string, markup_type)
    string = etree.tostring(tree, encoding="unicode")
    clean = cleaner.clean_html(string)

    return normalize("NFKD", clean)


def detect_XML(string: str) -> str:
    """ Detect XML by XML declaration and returns a markup type string."""
    if string.startswith("<?xml"):
        return "xml"
    else:
        return "html"


def extract_meta_informations(string: str, meta_type: str) -> list:
    """ Extracts meta information from 'title'-, 'keyword'- and description'- 
        meta elements (by choice) and returns the content in a list.
    """
    # title already in text
    if meta_type == "title":
        tags = ['meta[property="og:title"]', 'meta[name="title"]']
    elif meta_type == "keywords":
        tags = ['meta[property="og:keyword"]', 'meta[name="keyword"]']
    elif meta_type == "description":
        tags = ['meta[property="og:description"]', 'meta[name="description"]']
    
    else:
        tags = ['meta[property="og:description"]',
                'meta[name="description"]',
                'meta[property="og:keyword"]',
                'meta[name="keyword"]', 
                'meta[property="og:title"]', 
                'meta[name="title"]']

    tags = ", ".join(tags)

    
    try:      
        tree = extract_tree(string, "html")
        select = CSSSelector(tags, translator="html")
    except:
        tree = extract_tree(string, "xml")
        select = CSSSelector(tags, translator="xml")
        
    results = [element.get('content') for element in select(tree)]
    results = [x for x in results if x is not None]
    return " ".join(list(set(results)))

def extract_tagtexts(string: str, tag: str):
    """ Extract text content from all elements inside given tag."""
    try:      
        tree = extract_tree(string, "html")
        select = CSSSelector(tag, translator="html")
    except:
        tree = extract_tree(string, "xml")
        select = CSSSelector(tag, translator="xml")
        
    results = [element.text_content() for element in select(tree)]
    results = [x for x in results if x is not None]
    return " ".join(list(set(results)))

def extract_tree(string: str, markup_type: str) -> lxml.etree._Element:
    """ Extracts tree from string.

    Parameters
    ----------
    string : str
        String which contains the HTML, XHTML or XML.
    markup_type : str, default=None
        Indicate the markup type ('xml' or another).

    Returns
    -------
    tree : lxml.etree._Element
        Extracted lxml.etree Element.
    """
    # XML
    if markup_type == "xml":
        parser = etree.XMLParser(
            encoding="utf-8", ns_clean=True, recover=True, remove_comments=True
        )
        tree = etree.fromstring(string.encode("utf-8"), parser=parser)
    # HTML and XHTML
    else:
        parser = html.HTMLParser(encoding="utf-8")
        tree = html.fromstring(string.encode("utf-8"), parser=parser)
    return tree


def reduce_whitespace(string: str) -> str:
    """ Reduces all whitespace of a string to a single whitespace."""
    return " ".join(string.split())


def remove_special_characters(text: str, remove_digits: Optional[bool]=False):
    """ Removes special characters. """
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_tags(tree):
    """ Remove all tags of lxml tree and return string."""
    return " ".join(tree.itertext())

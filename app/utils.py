import re
from typing import Dict, List, Optional, Tuple, Union

from lxml.html.clean import Cleaner
from lxml import html, etree
import numpy as np
import pandas as pd


# TODO
# in clean_html evlt automatische detection von xml/html einbauen!!


# TODO docstring
# TODO: https://stackoverflow.com/questions/15304229/convert-python-elementtree-to-string
#       vllt verbessern in try?
def clean_html_boilerplate(row: pd.Series) -> str:
    """ Cleans boilerplate HTML tags from HTML."""

    string = row.html
    url = row.url

    cleaner = Cleaner(
        page_structure=False, meta=True, safe_attrs=[], style=True, kill_tags=["img"]
    )
    clean = string

    try:
        clean = cleaner.clean_html(string)
    except:
        try:
            parser = etree.XMLParser(ns_clean=True, recover=True, encoding="utf-8")
            parsed_xml = etree.fromstring(string.encode("utf-8"), parser=parser)
            string = etree.tostring(parsed_xml)
            string = string.decode("utf-8")
            clean = cleaner.clean_html(string)
        except:
            print(f"Website '{url}' couldn't be cleaned.")

    clean = clean.replace("\n", "")
    clean = clean.replace("\r", "")
    clean = clean.replace("\t", "")

    return clean


# TODO: docstring
# TODO: accept also strings
def remove_tags(tree):
    """ Remove all tags of lxml tree and return string."""
    return ' '.join(tree.itertext())


# TODO docstring
def tokenizing_html(text: str, token_list: Optional[List[str]] = []) -> List[str]:
    """ Tokenizes a HTML document by keeping the HTML tags with angle brackets 
        and the text tokens. If a token_list is given, only tokens of the list
        will be used, the others will be removed.
    """
    token_pattern = r"(?u)\b\w\w+\b"
    tag_pattern = r"</{0,1}[A-Za-z][A-Za-z0-9]*\s{0,1}/{0,1}>"
    regex = re.compile(token_pattern + "|" + tag_pattern)
    tokens = regex.findall(text)

    # create html tags from token list
    updated_token_list = []

    for token in token_list:
        updated_token_list.append(f"<{token}>")
        updated_token_list.append(f"<{token}/>")
        updated_token_list.append(f"</{token}>")
        updated_token_list.append(f"</ {token}>")
    
    if token_list:           
        return [token for token in tokens if token in updated_token_list or token[0] != "<"]
    else:
        return tokens


# TODO docstring
# TODO: allow xml
def trim_html(html_string, tag_list=[], tagless_output_string=False, return_tree=False):
    """ Trim a html string file by removing all tags which are not inside the tag list."""

    tree = html.fromstring(html_string)
    unique_tags = list(np.unique([element.tag for element in tree.iter()]))
    unique_tags = [element for element in unique_tags if element not in tag_list]
    
    etree.strip_tags(tree, unique_tags)
    
    if return_tree:
        return tree
    else:
        if tagless_output_string:
            return remove_tags(tree)
        else:
            return etree.tostring(tree, encoding="unicode", method="html")
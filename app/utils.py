import re
from typing import Dict, List, Optional, Tuple, Union

from lxml.html.clean import Cleaner
from lxml import html, etree
import numpy as np  # todo: nötig?
import pandas as pd  # todo: nötig?


def clean_boilerplate(string: str, url: str, website_type: Optional[str] = None) -> str:
    """Cleans boilerplate tags, attributes etc. from HTML, XHTML or XML.

    Parameters
    ----------
    string : str
        String which contains the HTML, XHTML or XML.
    url : str
        String which contains the websites URL.
    website_type : str, default=None
        Pass website type directly.

    Returns
    -------
    clean : str
        Cleaned HTML, XHTML or XML string.
    """

    cleaner = Cleaner(
        scripts=True,
        javascript=True,
        comments=True,
        style=True,
        inline_style=True,
        links=True,
        meta=True,  # TODO!!!
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

    if not website_type:
        website_type = detect_XML(string)

    # todo außerhalb?
    # todo docstring
    def clean_website(string, cleaner, website_type):
        """ TODO"""
        # XML
        if website_type == "xml":
            parser = etree.XMLParser(encoding="utf-8", recover=True)
            tree = etree.fromstring(string.encode("utf-8"), parser=parser)
        # HTML and XHTML
        else:
            parser = html.HTMLParser(encoding="utf-8")
            tree = html.fromstring(string.encode("utf-8"), parser=parser)
        string = etree.tostring(tree, encoding="unicode")
        clean = cleaner.clean_html(string)
        return clean

    try:
        clean = clean_website(string, cleaner, website_type)
    except:
        # try the other (not detected) website type
        try:
            if website_type == "xml":
                clean = clean_website(string, cleaner, "html")
            else:
                clean = clean_website(string, cleaner, "xml")
        except:
            clean = string
            print(
                f"Website '{url}' couldn't be cleaned (variable type: {type(string)})."
            )

    clean = clean.replace("\n", "")
    clean = clean.replace("\r", "")
    clean = clean.replace("\t", "")

    return clean


def detect_XML(string):
    """ Detect XML by XML declaration."""
    if string.startswith("<?xml"):
        return "xml"
    else:
        return "html"


# TODO: docstring
# TODO: accept also strings
def remove_tags(tree):
    """ Remove all tags of lxml tree and return string."""
    return " ".join(tree.itertext())


# TODO docstring
def tokenizing_html(text: str, token_list: Optional[List[str]] = []) -> List[str]:
    """Tokenizes a HTML document by keeping the HTML tags with angle brackets
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
        return [
            token for token in tokens if token in updated_token_list or token[0] != "<"
        ]
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

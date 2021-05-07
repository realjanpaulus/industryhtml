import re
from typing import Dict, List, Optional, Tuple, Union
from unicodedata import normalize

import lxml
from lxml.cssselect import CSSSelector
from lxml.html.clean import Cleaner
from lxml import html, etree
import numpy as np


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

    clean = clean.replace("\n", "")
    clean = clean.replace("\r", "")
    clean = clean.replace("\t", "")

    return clean


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


# TODO: DOCSTRING
def extract_meta_informations(string, markup_type):
    """ TODO"""
    # title bereits in text dabei
    tags = ['meta[property="og:description"]',
            'meta[name="description"]',
            'meta[property="og:keyword"]',
            'meta[name="keyword"]']

    tags = ", ".join(tags)

    # TODO: bessere Lösung?
    try:
        tree = extract_tree(string, markup_type)
        select = CSSSelector(tags, translator=markup_type)
        results = [element.get('content') for element in select(tree)]
        results = [x for x in results if x is not None]
        return list(set(results))
    except:
        return [""]

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

# TODO: docstring
# TODO: accept also strings
def remove_tags(tree):
    """ Remove all tags of lxml tree and return string."""
    return " ".join(tree.itertext())


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
        return [
            token for token in tokens if token in updated_token_list or token[0] != "<"
        ]
    else:
        return tokens


# TODO: allow xml. ICH: ?
def trim_html(
    html_string: str,
    clean_html: Optional[bool] = True,
    element_list: Optional[list] = None,
    keep_tags: Optional[bool] = False,
    return_tree: Optional[bool] = False,
):
    """Trim a html string file by removing all tags which are not inside the tag list.

    Parameters
    ----------
    html_string : str
        String which contains the HTML.
    clean_html : bool, default=True
        Indicates if the html string should be cleaned. Could prevent errors.
    element_list : list, default=None
        List with elements which should be kept.
    keep_tags : bool, default=False
        Indicates if tags should be removed or kept.
    return_tree : bool, default=False
        Indicates if lxml tree object or a string should be returned.

    Returns
    -------
    lxml.html.HtmlElement/string
        Returns trimmed tree or string.

    """

    if element_list is None:
        element_list = []

    if clean_html:
        html_string = clean_boilerplate(html_string, "")
    tree = html.fromstring(html_string)
    unique_tags = list(np.unique([element.tag for element in tree.iter()]))
    unique_tags = [element for element in unique_tags if element not in element_list]

    etree.strip_tags(tree, unique_tags)

    if return_tree:
        return tree
    else:
        if keep_tags:
            return etree.tostring(tree, encoding="unicode", method="html")
        else:
            return remove_tags(tree)

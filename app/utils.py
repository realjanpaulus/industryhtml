from lxml.html.clean import Cleaner
from lxml import html, etree


def clean_html_boilerplate(row):
    """ Cleans boilerplate HTML tags from HTML."""
    
    string = row.html
    url = row.url
    
    cleaner = Cleaner(page_structure=False, meta=True, style=True, kill_tags=["img"])
    clean = string
    
    try:
        clean = cleaner.clean_html(string)
    except:
        try:
            parser = etree.XMLParser(ns_clean=True, recover=True, encoding='utf-8')
            parsed_xml = etree.fromstring(string.encode('utf-8'), parser=parser)
            string = etree.tostring(parsed_xml)
            string = string.decode("utf-8")
            clean = cleaner.clean_html(string)
        except:
            print(f"Website '{url}' couldn't be cleaned.")
    
    clean = clean.replace("\n", "")
    clean = clean.replace("\r", "")
    clean = clean.replace("\t", "")
    
    return clean
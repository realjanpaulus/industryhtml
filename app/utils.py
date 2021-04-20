from lxml.html.clean import Cleaner

def clean_html_boilerplate(string):
    """ Cleans boilerplate HTML tags from HTML."""
    
    cleaner = Cleaner(page_structure=False, meta=True, style=True, kill_tags=["img"])
    clean = cleaner.clean_html(string)
    clean = clean.replace("\n", "")
    clean = clean.replace("\r", "")
    clean = clean.replace("\t", "")

    return clean
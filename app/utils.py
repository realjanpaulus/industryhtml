import pandas as pd
import spacy
from tqdm.notebook import tqdm
import typing
from typing import List, Optional


# TODO
# identifier einbauen, der wenigstens guckt, ob der text in einer bestimmten sprache ist!
# TODO: country codes (alle nur einmal):
#       BA, AD, EE, CX, HK, HT, PH, LT, LB, IE, FI, 
#       LI, RW, SG, TR, NZ, AO, UA, AR, PT, RS
def language_identifier(iso2):
    """ Returns the correct spacy model, given a iso2 string.

    """
    # TODO: weg?
    no_models = {"BE" : "xx_ent_wiki_sm",
                "CZ" : "xx_ent_wiki_sm",
                "HR" : "xx_ent_wiki_sm",
                "IN" : "xx_ent_wiki_sm",
                "MA" : "xx_ent_wiki_sm",
                "SE" : "xx_ent_wiki_sm",
                "UNKNOWN" : "xx_ent_wiki_sm"}

    languages = {"AT" : "de_core_news_sm",
                "AU" : "en_core_web_sm",
                "BR" : "pt_core_news_sm",
                "CA" : "en_core_web_sm",
                "CH" : "de_core_news_sm",
                "CN" : "zh_core_web_sm",
                "CO" : "es_core_news_sm",
                "DE" : "de_core_news_sm",
                "DK" : "da_core_news_sm",
                "ES" : "es_core_news_sm",
                "FR" : "fr_core_news_sm",
                "GB" : "en_core_web_sm",
                "GR" : "el_core_news_sm",
                "IT" : "it_core_news_sm",
                "MX" : "es_core_news_sm",
                "NL" : "nl_core_news_sm",
                "NO" : "nb_core_news_sm",
                "PL" : "pl_core_news_sm",
                "RO" : "ro_core_news_sm",
                "RU" : "ru_core_news_sm"}

    if iso2 in languages.keys():
        return languages[iso2]
    else:
        return "xx_ent_wiki_sm"


def remove_pos(df: pd.DataFrame, 
                pos_tags: Optional[List[str]] = ["VERB", "ADJ", "NOUN"]) -> List[str]:
    """ Remove every part of speach except the specified exceptions.
    
    Parameters
    ----------
    df
        DataFrame with at least the columns "text" and "country".
    pos_tags
        List of POS tags which should not be removed from the strings.
    

    Returns
    -------
    list
        List of strings with only the pos tag which are in `pos_tags.
    """

    df = df.sort_values(by=['country'])

    text_array = []

    unique_countries = list(df.country.unique())

    for country_code in tqdm(unique_countries, 
                                total = len(unique_countries), 
                                desc = "Country codes"):
        country_df = df[df["country"] == country_code]

        lang = language_identifier(country_code)

        try:
            nlp = spacy.load(lang)
        except OSError:
            nlp = spacy.load("xx_ent_wiki_sm")

        nlp.max_length = 3000000

        for index, row in tqdm(country_df.iterrows(), 
                                total = country_df.shape[0], 
                                desc = f"{country_code} texts"):

            text = row["text"]
            pos_str = nlp(text)
            new_text = ""
            for token in pos_str: 
                if token.pos_ in pos_tags:
                    new_text = new_text + token.text + " "
            text_array.append(new_text)

        
    return text_array

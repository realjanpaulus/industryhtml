def remove_pos(lst, used_pos = ["VERB", "ADJ", "NOUN"]):
    """ Remove every part of speach except the specified exceptions.
    """

    l = ["test test test test", "test test test test", "test test test test", "test test test test", 
        "test test test test", "test test test test", "test test test test", "test test test test", 
        "test test test test", "test test test test", "test test test test", "test test test test", 
        "test test test test", "test test test test", "test test test test", "test test test test", 
        "test test test test", "test test test test", "test test test test", "test test test test", 
        ["test test test test", "test test test test", "test test test test", "test test test test", 
            "test test test test", "test test test test", "test test test test", "test test test test"
            "test test test test", "test test test test", "test test test test", "test test test test", ]]

    nlp = spacy.load('de')
    nlp.max_length = 3000000
    new_lst = []
    for s in lst:
        posstr = nlp(s)
        new_string = ""
        for token in posstr:  
            if token.pos_ in used_pos:
                new_string = new_string + token.text + " "
        new_lst.append(new_string)
        
    return new_lst
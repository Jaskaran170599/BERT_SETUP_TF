def jaccard(ext,targets): 
    score=[]
    for str1,str2 in zip(ext,targets):
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        c = a.intersection(b)
        score.append(float(len(c)) / (len(a) + len(b) - len(c)))
    return score
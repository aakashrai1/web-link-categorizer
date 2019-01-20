"""
Created on Tue Nov 13 15:13:24 2018

@author: akash
"""

import re
from urllib.parse import urlparse

with open('./data/stop_words.txt', 'r') as f:
    temp = f.read()
    stopWords = temp.split('\n')

class URLParser():
    
    
    def __init__(self, url, title = ""):
        self.data = {
            'scheme': '',
            'netloc': '',
            'path': '',
            'query': '',
            'params': '',
            'extension': '',
            'pathWords' : [],
            'tokenizedURL': [],
            'tokens': [],
            'titleTokens': []
        }
        self.url = url.lower()
        self.title = title.lower()
        self.processURL()
    
    
    def segmentizeWord(self, w):
        try:
            sp = re.split(r'_|-', w)
            res = []
            for c in sp:
                if (c and c!= ""):
                    if (c.isalnum()):
                        striped = re.sub(r'\d+', '', c)
                        if (striped != ""):
                            res.append(striped)
                    else:
                        res.append(c)
            return res
        except:
            return []

        
    def union(self, a, b):
        return list(set(a) | set(b))
    
    
    def findNgrams(self, inputList, n):
        return zip(*[inputList[i:] for i in range(n)])
    
    
    def generateNgrams(self, IL, m, n):
        tokens = []
        for i in range(m, n+1):
            for j in list(self.findNgrams(IL, i)):
                tokens.append(' '.join(j))
        return tokens
        
    
    def processURL(self):
        try:
            tokens = set()
            titleTokens = set()
            # Tokenized is complete tokenized url
            self.data['tokenizedURL'] = re.split(r"[/:\.?=&_-]+", self.url)
            
            cleanedURL = []
            for w in re.split(r"[\d/:\.?=&_-]+", self.url):
                if (w not in stopWords):
                    cleanedURL.append(w)
                    tokens.add(w)
                    
            cleanedTitle = []
            t = re.sub(r'[^\w\s]','', self.title)
            for w in re.split(r"[\d\s/:\.?=&_-]+", t):
                if (w not in stopWords):
                    cleanedTitle.append(w)
                    titleTokens.add(w)
                
            """
            # Creating n grams
            for i in self.generateNgrams(cleanedURL, 2, 5):
                tokens.add(i)
              
            # Title
            for i in self.generateNgrams(cleanedTitle, 2, 5):
                titleTokens.add(i)
            """
            
            # Url processing
            u = urlparse(self.url)
            self.data['scheme'] = u.scheme
            self.data['netloc'] = u.netloc
            self.data['path'] = u.path
            self.data['query'] = u.query
            self.data['params'] = u.params
            extSplit = u.path.rsplit('.', 1)
            
            urlExtension = "" if (len(extSplit) == 1) else extSplit[1]
            self.data['extension'] = urlExtension
            
            # Words extracted from the path
            # It's different from tokenized url in a sense that it is cleaned
            # after removing numeric characters from alphanumeric word
            for w in extSplit[0].split('/')[1:]:
                for i in self.segmentizeWord(w):
                    self.data['pathWords'].append(i)
            
            self.data['tokens'] = list(tokens)
            self.data['titleTokens'] = list(titleTokens)
                
            #print(self.data)
        except:
            #print("Error occured ")
            pass

    def getParsedData(self):
        return self.data
        
   
if __name__ == "__main__":
    pass
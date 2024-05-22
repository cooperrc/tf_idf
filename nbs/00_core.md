---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.2
  kernelspec:
    display_name: python3
    language: python
    name: python3
---

# core

> Fill in a module description here

```python
#| default_exp core
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
import nltk
nltk.download('stopwords', quiet=True)

nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

```
https://medium.com/@mifthulyn07/comparing-text-documents-using-tf-idf-and-cosine-similarity-in-python-311863c74b2c

```python
#| hide
from nbdev.showdoc import *
```

```python
#| export
def foo(): pass
```

```python
#| hide
import nbdev; nbdev.nbdev_export()
```
```python
def preprocess_text(text):
    # lowercasing
    lowercased_text = text.lower()

    # cleaning 
    import re 
    remove_punctuation = re.sub(r'[^\w\s]', '', lowercased_text)
    remove_white_space = remove_punctuation.strip()

    # Tokenization = Breaking down each sentence into an array
    # from nltk.tokenize import word_tokenize
    tokenized_text = word_tokenize(remove_white_space)

    # Stop Words/filtering = Removing irrelevant words
    # from nltk.corpus import stopwords
    # stopwords = set(stopwords.words('english'))
    stopwords_removed = [word for word in tokenized_text if word not in stopwords.words()]

    # Stemming = Transforming words into their base form
    #from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    stemmed_text = [ps.stem(word) for word in stopwords_removed]
    
    # Putting all the results into a dataframe.
    df = pd.DataFrame({
        'DOCUMENT': [text],
        'LOWERCASE' : [lowercased_text],
        'CLEANING': [remove_white_space],
        'TOKENIZATION': [tokenized_text],
        'STOP-WORDS': [stopwords_removed],
        'STEMMING': [stemmed_text]
    })

    return df
```

```python
AI = 'For instance, in the design phase of a structural engineering project, Monte Carlo simulations can help evaluate the performance of a proposed design under different loading conditions and material properties, providing valuable insights into its reliability and safety'
ME = 'For instance, Monte Carlo simulations can simulate hundreds or thousands of different combinations of loading conditions and material properties to create statistical predictions of structure stiffness'
# word_tokenize(AI.lower().split())
# preprocess_text(AI)
```

```python
compare = preprocess_text(AI)
```

```python
compare = pd.concat([compare, preprocess_text(ME)], ignore_index=True)
compare
```

```python
def calculate_tfidf(df):
    # Call the preprocessing result
    #df = preprocessing(corpus)
        
    # Make each array row from stopwords_removed to be a sentence
    stemming = df['STEMMING'].apply(' '.join)
    
    # Count TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(stemming)
    
    # Get words from stopwords array to use as headers
    feature_names = vectorizer.get_feature_names()

    # Combine header titles and weights
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    df_tfidf = pd.concat([df, df_tfidf], axis=1)

    return df_tfidf
```

```python
compare_tfidf = calculate_tfidf(compare)
compare_tfidf
```

```python
def cosineSimilarity(df):
    # Call the TF-IDF result
    df_tfidf = calculate_tfidf(df)
    
    # Get the TF-IDF vector for the first item (index 0)
    vector1 = df_tfidf.iloc[0, 6:].values.reshape(1, -1)

    # Get the TF-IDF vector for all items except the first item
    vectors = df_tfidf.iloc[:, 6:].values
    
    # Calculate cosine similarity between the first item and all other items
    from sklearn.metrics.pairwise import cosine_similarity
    cosim = cosine_similarity(vector1, vectors)
    cosim = pd.DataFrame(cosim)
    
    # Convert the DataFrame into a one-dimensional array
    cosim = cosim.values.flatten()

    # Convert the cosine similarity result into a DataFrame
    df_cosim = pd.DataFrame(cosim, columns=['COSIM'])

    # Combine the TF-IDF array with the cosine similarity result
    df_cosim = pd.concat([df_tfidf, df_cosim], axis=1)

    return df_cosim[['DOCUMENT', 'STEMMING', 'COSIM']]
```

```python
cosineSimilarity(compare)
```

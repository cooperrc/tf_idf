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

```python
#| hide
from tf_idf.core import *
```

# tf_idf

> This is a short set of functions meant to help analyze cosine similarity between texts


This file will become your README and also the index of your documentation.


## Install

<!-- #region -->
```sh
pip install tf_idf
```
<!-- #endregion -->

## How to use


Fill me in please! Don't forget code examples:
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
compare_tfidf = calculate_tfidf(compare)
compare_tfidf
```


```python
cosineSimilarity(compare)
```

```python

```

```python
1+1
```

```python

```

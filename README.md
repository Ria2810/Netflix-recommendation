# Netflix-recommendation
<h1 align="center">PROJECT</h1>
<h2 align="center">PREDICT THE INTEREST OF USERS USING NETFLIX DATASET</h2>

<h2 align="center">PROBLEM STATEMENT</h2>
Netflix provided a lot of anonymous rating data, and a prediction accuracy bar on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)

### LIBRARIES USED
- NUMPY: NumPy is a Python library used for working with arrays.It also has functions for working in domain of linear algebra, fourier transform, and matrices.

- PANDAS: Pandas is an open-source, BSD-licensed Python library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.

- MATPLOTLIB: Matplotlib is a low level graph plotting library in python that serves as a visualization utility.Matplotlib is open source and we can use it freely.Matplotlib is mostly written in python, a few segments are written in C, Objective-C and Javascript for Platform compatibility.

- SEABORN: Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

- SCIKIT-LEARN: Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.

```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```
### CONCEPTS USED
Machine learning is a large field of study that overlaps with and inherits ideas from many related fields such as artificial intelligence.The focus of the field is learning, that is, acquiring skills or knowledge from experience.

There are four types of machine learning:
- SUPERVISED LEARNING: Supervised Learning is the one, where you can consider the learning is guided by a teacher. Once the model gets trained it can start making a prediction or decision when new data is given to it.
- UNSUPERVISED LEARNING: The model learns through observation and finds structures in the data. Once the model is given a dataset, it automatically finds patterns and relationships in the dataset by creating clusters in it.
- SEMI-SUPERVISED LEARNING: As the name suggests, its working lies between Supervised and Unsupervised techniques. We use these techniques when we are dealing with a data which is a little bit labelled and rest large portion of it is unlabeled.
- REINFORCEMENT LEARNING: It is the ability of an agent to interact with the environment and find out what is the best outcome. It follows the concept of hit and trial method.
### Types of Supervised Learning:
- Classification : It is a Supervised Learning task where output is having defined labels(discrete value).

- Regression : It is a Supervised Learning task where output is having continuous value.

### Types of Unsupervised Learning :-
- Clustering: Broadly this technique is applied to group data based on different patterns, our machine model finds.

- Association: This technique is a rule based ML technique which finds out some very useful relations between parameters of a large data set.

### [Source code with output](https://github.com/Ria2810/Netflix-recommendation/blob/main/Edu_project_netflix_Copy1.ipynb)

### Dataset: 
[Netflix.csv](https://github.com/Ria2810/Netflix-recommendation/blob/main/netflix%20dataset.csv)

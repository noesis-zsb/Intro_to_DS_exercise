# %% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 1

# %% [markdown]
# ## Exercise 1 | Matrix warm-up
# <span style="background-color: #ccfff2"> *Note: You can find tutorials for NumPy and Pandas under 'Useful tutorials' in the course material.*</span>

# %% [markdown]
# One of the most useful properties of any scientific programming language (Python with NumPy, R, Julia, etc) is that they allow us to work with matrices efficiently. Let's learn more about these features!
# 
# ### 1.1 Basics
# 
# 1. Let's start by creating two arrays <span style="background-color: #ccfff2"> A</span> and <span style="background-color: #ccfff2"> B</span> which each have the integers <span style="background-color: #ccfff2"> 0, 1, 2, ..., 1e7-1</span>. Use the normal arrays or lists of the programming language you are using, e.g. *list* or *[ ]* or *numpy.array()* in Python.

# %%
# Use this cell for your code
import numpy as np
N = int(1e7)
a = np.arange(N)
b = np.arange(N)

# %% [markdown]
# 2. Create a function that uses a <span style="background-color: #ccfff2"> for loop</span> or equivalent to return a new array <span style="background-color: #ccfff2"> C</span>, which contains the <span style="background-color: #ccfff2"> element-wise sum of *A* and *B*</span>, e.g. C should contain the integers <span style="background-color: #ccfff2"> 0, 2, 4, etc</span>.

# %%
import numpy as np
def add_with_for(A,B):
    C = []
    for i in range(A.size):
        C.append(A[i] + B[i])
    # add your code here
    return C

# %% [markdown]
# 3. Next, let's create another function that uses NumPy (or equivalent) to do the same. To try it out, allocate two arrays (e.g. using <span style="background-color: #ccfff2"> np.array</span> in NumPy) and add the arrays together using your function. Don't use loops, instead, find out how to add the two arrays directly. What do you notice in comparison to the previous function?

# %%
# Use this cell for your code
# This function takes less time than using a loop.
def alternative_add(A, B):
    return A + B

# %% [markdown]
# ### 1.2 Array manipulation

# %% [markdown]
# <span style="background-color: #ccfff2"> *Note: for the following exercises, only use NumPy or equivalent functions. Don't use any loops.* </span>
# 1. Create the following array:
# 
# *[hint: <span style="background-color: #ccfff2"> np.reshape</span>]*

# %%
# array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
#        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
#        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
#        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
#        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
#        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])

# %%
# Use this cell for your code
import numpy as np
c = np.arange(100)
d = c.reshape(10, 10)
print(d)

# %% [markdown]
# 2. Create the following array:

# %%
# array([[0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]])

# %%
# Use this cell for your code
import numpy as np
e = np.array([0, 1])
f = np.tile(e, (10, 5))
print(f)
# %% [markdown]
# 3. Create the following array (D):

# %%
# array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]])

# %%
# Use this cell for your code
import numpy as np
D = np.fromfunction(lambda i, j: (i!=j).astype(int), (10, 10), dtype=int)
print(D)

# %% [markdown]
# 4. Create the following array (E):

# %%
# array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

# %%
# Use this cell for your code
import numpy as np
E = np.fromfunction(lambda i, j: (i+j!=9).astype(int), (10, 10), dtype=int)
print(E)

# %% [markdown]
# 5. Call the last two matrices <span style="background-color: #ccfff2">D</span> and <span style="background-color: #ccfff2">E</span>, respectively. Show that the determinant of their product (matrix multiplication) is the same as the product of their determinants. That is calculate both <span style="background-color: #ccfff2">det(DE)</span> and <span style="background-color: #ccfff2">det(D) * det(E)</span>, and show that they are the same. Is it a coincidence? (I think not) The product of the determinants (or the determinant of the product) should be -81.

# %%
# Use this cell for your code
import numpy as np
D = np.fromfunction(lambda i, j: (i!=j).astype(int), (10, 10), dtype=int)
E = np.fromfunction(lambda i, j: (i+j!=9).astype(int), (10, 10), dtype=int)

det_DE = np.linalg.det(D@E)
print(round(det_DE)==round(np.linalg.det(D)*np.linalg.det(E)))

# %% [markdown]
# -- *Use this markdown cell for your written answer* --
# It's not a coincidence. This is an axiom.
# The determinant of the product of two matrices is equal to the product of their individual determinants.


# %% [markdown]
# ### 1.3 Slicing
# 
# Array slicing is a powerful way to extract data from an array. Let's practice array slicing with the following exercises!
# 
# 1. Load the [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). The data should be a matrix of shape <span style="background-color: #ccfff2">(20640, 8)</span>, that is 20640 rows and 8 columns. Use the <span style="background-color: #ccfff2">.shape</span> attribute of NumPy arrays to verify this. Here's a [description of the fields](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

# %%
# Use this cell for your code
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
a = housing.data
print(a.shape)

# %% [markdown]
# 2. Select rows where the average number of bedrooms <span style="background-color: #ccfff2">(AveBedrms)</span> is higher than 2. The first few row indices should be <span style="background-color: #ccfff2">710,  1023,  1024, ...</span> (zero-indexed). Count these houses - how many rows are selected? *[hint: <span style="background-color: #ccfff2">np.where</span>]*

# %%
# Use this cell for your code
import pandas as pd
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

filtered = df[df["AveBedrms"] > 2]
print("Number of selected rows:", filtered.shape[0])
print(filtered)

# %% [markdown]
# 3. Select the rows where the median house age (i.e. median in each block group) <span style="background-color: #ccfff2">(HouseAge)</span> is between 1 and 3 years (inclusive). There should be **124** of these.

# %%
# Use this cell for your code
import pandas as pd
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

filtered = df[(df["HouseAge"]>=1) & (df["HouseAge"]<=3)]
print("number of selected rows: ", filtered.shape[0])
print(filtered)

# %% [markdown]
# 4. Find the mean of the block group population <span style="background-color: #ccfff2">(Population)</span> for homes whose median value is more than 25000 USD (the target variable). It should be around **1425.68**.

# %%
# Use this cell for your code
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame

filtered = df[df["MedHouseVal"]>0.25]
print(filtered["Population"].mean())

# %% [markdown]
# ## Exercise 2 | Working with text data
# 
# Next, let's look into some text data. We will be looking into Amazon reviews, and the necessary steps to transform a raw dataset into a format more suitable for prediction tasks.
# 
# 1. Download the automotive 5-core dataset from [here](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz). Next, you can extract the data in <span style="background-color: #ccfff2">JSON</span> format. You can also download one of the bigger ones, if you are feeling ambitious. Open the JSON file. Access the <span style="background-color: #ccfff2">reviewText</span> field, which contains the unstructured review text written by the user.
# 
# For instance, the first review reads as follows: 
# 
# *'After I wrote the below review, the manufacturer contacted me and explained how to use this.  Instead of the (current) picture on Amazon where the phone is placed vertically, you actually use the stand with the phone placed horizontally. [...]'*

# %%
# Use this cell for your code
# The following code assumes that the file has been downloaded to the same directory
import json, gzip

data = []
with gzip.open("Automotive_5.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
print(data[0]["reviewText"])

# %% [markdown]
# 2. Next, let's follow some steps to normalize the text data.
# 
# When dealing with natural language, it is important to notice that while, for example, the words "Copper" and "copper" are represented by two different strings, they have the same meaning. When applying statistical methods on this data, it is useful to ensure that words with the same meaning are represented by the same string.
# 
# * <span style="background-color: #ccfff2">Downcasing</span>: Let's first downcase the contents of the <span style="background-color: #ccfff2">reviewText</span> field.
# 
# Now the first review should be:
# 
# *'after i wrote the below review, the manufacturer contacted me and explained how to use this.  instead of the (current) picture on amazon where the phone is placed vertically, you actually use the stand with the phone placed horizontally.'*

# %%
# Use this cell for your code
# The following code assumes that the file has been downloaded to the same directory
import json, gzip

data = []
with gzip.open("Automotive_5.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
review = data[0]["reviewText"]
print(review.lower())

# %% [markdown]
# 3. Let's continue with punctuation and stop word removal. Stop words are words like "and", "the", etc. They are usually very common words that have little to do with the actual content matter. There's plenty openly available lists of stop words for almost any (natural) language.
# 
# * <span style="background-color: #ccfff2">Punctuation and stop-word removal</span>: Let's now remove all punctuation, as well as the stop words. You can find a stop word list for English, e.g. [here](https://gist.github.com/xldrkp/4a3b1a33f10d37bedbe0068f2b4482e8#file-stopwords-en-txt).*(use the link to download a txt of english stopwords)* Save the stopwords in the file as "stopwords-en.txt".
# 
# First review at this point reads as: 
# 
# *'wrote review manufacturer contacted explained current picture amazon phone vertically stand phone horizontally'*

# %%
# Use this cell for your code
# The following code assumes that the file has been downloaded to the same directory
import json, gzip, string

data = []
with gzip.open("Automotive_5.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
review = data[0]["reviewText"].lower()

#review = "after i wrote the below review, the manufacturer contacted me and explained how to use this.  instead of the (current) picture on amazon where the phone is placed vertically, you actually use the stand with the phone placed horizontally."
print("That is original review: ", review)

# get stopwords
with open("stopwords-en.txt", "r", encoding="utf-8") as f:
    stopwords = set(word.strip() for word in f.readlines())

import nltk
original_words = nltk.word_tokenize(review)
# delete punctuation
filtered_words = [word for word in original_words if word.isalpha()]
# delete stopwords
filtered_words = [word for word in filtered_words if word not in stopwords]
print("This is filtered text: ")
print(" ".join(filtered_words))

# %% [markdown]
# 4. Let's continue with stemming. For example, while the words "swims" and "swim" are different strings, they both refer to swimming. [Stemming](https://en.wikipedia.org/wiki/Stemming) refers to the process of mapping words from their inflected form to their base form, for instance: swims -> swim.
# 
# * <span style="background-color: #ccfff2">Stemming</span>: Apply a stemmer on the paragraphs, so that inflected forms are mapped to the base form. For example, for Python the popular natural language toolkit [nltk](http://www.nltk.org/howto/stem.html) has an easy to use stemmer. In case you are using R, you can try the [Snowball stemmer](https://www.rdocumentation.org/packages/corpus/versions/0.10.2/topics/stem_snowball). You can find out how to install nltk from [here](https://www.nltk.org/install.html). It will take a while to run! So, grab a coffee and wait :D
# 
# Finally, after stemming: 
# 
# *'wrote review manufactur contact explain current pictur amazon phone vertic stand phone horizont'*

# %%
# Use this cell for your code
import json, gzip, string

data = []
with gzip.open("Automotive_5.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
review = data[0]["reviewText"].lower()

#review = "after i wrote the below review, the manufacturer contacted me and explained how to use this.  instead of the (current) picture on amazon where the phone is placed vertically, you actually use the stand with the phone placed horizontally."
#print("This is original review: ", review)

# get stopwords
with open("stopwords-en.txt", "r", encoding="utf-8") as f:
    stopwords = set(word.strip() for word in f.readlines())

import nltk
original_words = nltk.word_tokenize(review)
# delete punctuation
filtered_words = [word for word in original_words if word.isalpha()]
# delete stopwords
filtered_words = [word for word in filtered_words if word not in stopwords]
print("This is filtered text: ")
print(" ".join(filtered_words))

# stemming
porter = nltk.PorterStemmer()
stemmed_words = [porter.stem(word) for word in filtered_words]
print("after stemming: ", stemmed_words)

# %% [markdown]
# 5. Finally, filter the data by selecting reviews where the field <span style="background-color: #ccfff2">overall</span> is 4 or 5, and store the review texts in a file named <span style="background-color: #ccfff2">pos.txt</span>. Similarly, select reviews with rating 1 or 2 and store them in a file named <span style="background-color: #ccfff2">neg.txt</span>. Ignore the reviews with overall rating 3. Each line in the two files should contain exactly one preprocessed review text without the rating.

# %%
# Use this cell for your code
import json, gzip, string
import nltk

# get stopwords
with open("stopwords-en.txt", "r", encoding="utf-8") as f:
    stopwords = set(word.strip() for word in f.readlines())


def preprocess(text):

    if not isinstance(text, str):
        return ""
    text = text.lower()
    original_words = nltk.word_tokenize(text)
    # delete punctuation
    filtered_words = [word for word in original_words if word.isalpha()]
    # delete stopwords
    filtered_words = [word for word in filtered_words if word not in stopwords]
    #print("after removing punctuation and stopwords: ")
    #print(" ".join(filtered_words))

    # stemming
    porter = nltk.PorterStemmer()
    stemmed_words = [porter.stem(word) for word in filtered_words]
    #print("after stemming: ", stemmed_words)

    return " ".join(stemmed_words)

data = []
with gzip.open("Automotive_5.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
#review = data[0]["reviewText"]

import pandas as pd
df = pd.DataFrame(data)
pos_df = df[df['overall'].isin([4, 5])]
neg_df = df[df['overall'].isin([1, 2])]

pos_review = pos_df['reviewText'].str.replace('\n', ' ', regex=False).apply(preprocess)
neg_review = neg_df['reviewText'].str.replace('\n', ' ', regex=False).apply(preprocess)

pos_review.to_csv('pos.txt', index=False, header=False)
neg_review.to_csv('neg.txt', index=False, header=False)

# %% [markdown]
# ## Exercise 3 | SQL basics
#
# Next, let's take a refresher on the basics of SQL. In this exercise, you will be working on the simplified Northwind 2000 SQLite database. You can download the database from Kaggle here: https://courses.mooc.fi/api/v0/files/course/f92ffc32-2dd4-421d-87f3-c48800422cc5/files/VEKX2bxGCDGyojG902gmYZTXCnrAQw.zip
#
# To test your SQL queries and complete the exercise, you can download and install SQLite if you don't yet have it installed.
# 
# Please write SQL queries for the tasks on the simplified Northwind 2000 SQLite database.
# 
# 1. List the first name, last name, and hire date of all employees hired after January 1st, 1994.
# select LastName, FirstName, HireDate from Employees where HireDate > '1994-01-01' order by HireDate;

# 2. Count how many orders each customer has placed.
# SELECT CustomerID, COUNT(OrderID) AS OrderNum FROM Orders GROUP BY CustomerID ORDER BY OrderNum DESC;

# 3. Find the names of all customers who have ordered the product "Chai".
# select distinct c.CompanyName
# from Customers c
# join Orders o on o.CustomerID = c.CustomerID
# join OrderDetails od on od.OrderID = o.OrderID
# join Products p on p.ProductID = od.ProductID
# where p.ProductName = 'Chai';

# 4. Find all orders that have been placed but not yet shipped.
# select * from Orders where ShippedDate is null;

# 5. Find the customer who has placed the most orders.
# select c.CompanyName, count(o.OrderID) as OrderNum
# from Customers c
# join Orders o on o.CustomerID = c.CustomerID
# group by c.CustomerID, c.CompanyName
# order by OrderNum desc
# limit 1;

# %% [markdown]
# **Remember to submit your solutions. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences. Remember to also submit your SQL queries. No need to submit the text files for the programming exercises.**



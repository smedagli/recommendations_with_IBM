# Recommendations with IBM
Udacity project: recommendations with IBM


_categorizing tweets in case of disaster and catastrophic events_
###### Author: Stefano Medagli
###### date: 13.07.2020
###### ver: 0.1
## prerequisites
create a conda environment using the file in `environment/pkg.txt`

```bash
conda create --name disaster --file disaster_response/environment/pkg.txt
```
#### @TODO:

# Introduction
For this project you will analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles you think they will like
Below you can see an example of what the dashboard could look like displaying articles on the IBM Watson Platform.

# Tasks
I. Exploratory Data Analysis

Before making recommendations of any kind, you will need to explore the data you are working with for the project. Dive in to see what you can find. There are some basic, required questions to be answered about the data you are working with throughout the rest of the notebook. Use this space to explore, before you dive into the details of your recommendation system in the later sections.

II. Rank Based Recommendations

To get started in building recommendations, you will first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

III. User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. You will implement this next.

IV. Content Based Recommendations (EXTRA - NOT REQUIRED)

Given the amount of content available for each article, there are a number of different ways in which someone might choose to implement a content based recommendations system. Using your NLP skills, you might come up with some extremely creative ways to develop a content based recommendation system. You are encouraged to complete a content based recommendation system, but not required to do so to complete this project.

V. Matrix Factorization

Finally, you will complete a machine learning approach to building recommendations. Using the user-item interactions, you will build out a matrix decomposition. Using your decomposition, you will get an idea of how well you can predict new articles an individual might interact with (spoiler alert - it isn't great). You will finally discuss which methods you might use moving forward, and how you might test how well your recommendations are working for engaging users.


## folder structure
```bash
|   .gitignore
|   README.md
|           
+---analysis
|       users.py
|       __init__.py
|       
+---data
|       articles_community.csv
|       user-item-interactions.csv
|       
+---info
|       paths.py
|       tree.txt
|       version.py
|       __init__.py
|       
+---processing
|   |   common.py
|   |   __init__.py
|           
+---read_data
|   |   reader.py
|   |   __init__.py
|           
+---recommendations
|       collaborative.py
|       matrixFactorization.py
|       __init__.py
|       
\---verification
        verify.py
        __init__.py

```

### components
#### analysis
* *users.py*:
contains method to plot basic data
#### data
the folder contains the .csv data (paths defined anyway in `info/paths.py`)
#### info
* *paths.py*:
contains the default paths for the project
* *version.py*:
contains versioning information
#### processing
* *common.py*:
contains common funcions 
#### read_data
* *reader.py*:
contains the class to read and perform initial operations on the data
#### recommendations
* *collaborative.py*:
contains method to do recommendations based on collaborative filtering
* *matrixFactorization.py*:
contains method to do recommendations based on matrix factorization (i.e. _FunkSVD_)
 
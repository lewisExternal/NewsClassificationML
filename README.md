
# News Classification ML

This is a machine learning solution to categorize news headlines into their respective categories.  The models used in this example were; Multinomial Naive Bayes classifier & Support-vector machine (SVM) with SGD training. Using the the bag-of-words approach, where each unique word in a text will be represented by one number.

## Data 

The data is sotred as a JSON file.

"News_Category_Dataset_v2.json"

This data was sourced from Kaggle, the link for which can be seen below.  

[https://www.kaggle.com/rmisra/news-category-dataset](https://www.kaggle.com/rmisra/news-category-dataset)

Excerpt included for reference. 

"""

### Context

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from  [HuffPost](https://www.huffingtonpost.com/). The model trained on this dataset could be used to identify tags for untracked news articles or to identify the type of language used in different news articles.

### Content

Each news headline has a corresponding category. Categories and corresponding article counts are as follows:

-   `POLITICS`:  `32739`
    
-   `WELLNESS`:  `17827`
    
-   `ENTERTAINMENT`:  `16058`
    
-   `TRAVEL`:  `9887`
    
-   `STYLE & BEAUTY`:  `9649`
    
-   `PARENTING`:  `8677`
    
-   `HEALTHY LIVING`:  `6694`
    
-   `QUEER VOICES`:  `6314`
    
-   `FOOD & DRINK`:  `6226`
    
-   `BUSINESS`:  `5937`
    
-   `COMEDY`:  `5175`
    
-   `SPORTS`:  `4884`
    
-   `BLACK VOICES`:  `4528`
    
-   `HOME & LIVING`:  `4195`
    
-   `PARENTS`:  `3955`
    
-   `THE WORLDPOST`:  `3664`
    
-   `WEDDINGS`:  `3651`
    
-   `WOMEN`:  `3490`
    
-   `IMPACT`:  `3459`
    
-   `DIVORCE`:  `3426`
    
-   `CRIME`:  `3405`
    
-   `MEDIA`:  `2815`
    
-   `WEIRD NEWS`:  `2670`
    
-   `GREEN`:  `2622`
    
-   `WORLDPOST`:  `2579`
    
-   `RELIGION`:  `2556`
    
-   `STYLE`:  `2254`
    
-   `SCIENCE`:  `2178`
    
-   `WORLD NEWS`:  `2177`
    
-   `TASTE`:  `2096`
    
-   `TECH`:  `2082`
    
-   `MONEY`:  `1707`
    
-   `ARTS`:  `1509`
    
-   `FIFTY`:  `1401`
    
-   `GOOD NEWS`:  `1398`
    
-   `ARTS & CULTURE`:  `1339`
    
-   `ENVIRONMENT`:  `1323`
    
-   `COLLEGE`:  `1144`
    
-   `LATINO VOICES`:  `1129`
    
-   `CULTURE & ARTS`:  `1030`
    
-   `EDUCATION`:  `1004`
    

### Acknowledgements
This dataset was collected from  [HuffPost](https://www.huffingtonpost.com/).

  """

## How to run

Use the Jupyter Notebook provided ( .ipynb file extension). 
 
## Pipeline 

The script goes through the steps below. 

 1. remove punctuation & lower 
 2. remove numeric characters 
 3. [Porter stemmer](https://tartarus.org/martin/PorterStemmer/) for word stemming  
 4. remove [stopwords](https://en.wikipedia.org/wiki/Stop_word)
 5. count vectorizer, strings to token integer counts
 6. [TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/#:~:text=TF%2DIDF%20is%20a%20statistical,across%20a%20set%20of%20documents.) transformer, integer counts to weighted TF-IDF scores
 7. train on TF-IDF vectors w/ Naive Bayes classifier/ SVM with SGD training
 8. A classification report will then be produced containing the f1-score to test the validity of the model 

## Requirements

Relevant package installs are the first cell within the Jupyter Notebook. 

from clean import CleanData
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import operator
# Libraries for  LDA
import gensim
from gensim import corpora 
import pyLDAvis
import pyLDAvis.gensim_models
# from textblob import TextBlob
import seaborn as sns

# This file does the calculations and plotting for the models 
# In CompareStars() the column "Star Rating" is coded in so with new files it will need to be changed

# Key-word Extraction using CountVectorizer along with the RankTerms and PlotTerms functions
def TermExtration(raw_text):
    """
    Turned the review data into numeric values. 

    returns the top 15 words (features)
    
    calls a plotting function and returns a plot
    
    """
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'sink','also','good','like','great','come','would','faucet','faucets','shower','delta','water','amazon','delta','the','Delta','want','wanted','wish','desired','one','glass','rinser','head','seat','toilet','kitchen']
    # Calls fucntions from Sklearn to give numeric values to each word
    vectorizer = CountVectorizer(analyzer='word',max_features=5000,stop_words=stop_words,lowercase=True,token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    x_counts = vectorizer.fit_transform(raw_text)
    terms = vectorizer.get_feature_names()
    # Calls plotting function
    PlotTerms(RankTerms(x_counts,terms))
# Ranks term from the Doc-term-matrix
def RankTerms(A, terms):
    # get the sum over each column 
    sums = A.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documets
    return sorted(weights.items(),key=operator.itemgetter(1), reverse=True)
# Plots ranked terms 
def PlotTerms(rankings):
    # Initialize list 
    top_terms = []
    top_weights = []
    # get the top terms and their weights
    for i, pair in enumerate(rankings[0:15]):
        top_terms.append(pair[0])
        top_weights.append(pair[1])
    # note we reverse the ordering for the plot
    top_terms.reverse()
    top_weights.reverse()
    # create the plot
    fig = plt.figure(figsize=(13,8))
    # add the horizontal bar chart
    ypos = np.arange(len(top_terms))
    ax = plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)
    plt.title('Non-negative matrix factorization top 15')
    plt.xlabel("Term Weight",fontsize=14)
    plt.tight_layout()
    # plt.savefig("RENAME.png")
    plt.show()
# Executes and runs LDA model
def LDA(raw_text,num_topics):
    """
    Runs the LDA model from the gensim libray.
    
    Returns an interactive display from pyLDAvis 
    
    """
    # call clean_data function (removes punction, stop words, tokenizes, and lemmatizes)
    clean_text = CleanData(raw_text)
    # Term dictionary from corpus
    dictionary = corpora.Dictionary(clean_text)
    # convert list of reviews into Document Term Matrix using dictionary (convert dictionary into bag of words)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in clean_text]
    # Build LDA Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix,id2word=dictionary,num_topics=num_topics,random_state=123,chunksize=50,passes=10)
    # Visualize the topics
    prep = pyLDAvis.gensim_models.prepare(lda_model, doc_term_matrix,dictionary)
    html_string = pyLDAvis.prepared_data_to_html(prep)
    return html_string
# Allows users to see how a single word impacts R&R
def CompareStars(df,searchwords):
    # For nicer formatting in web app
    word = ", "
    word = word.join(searchwords)
    # Extracts all rows in the data frame that contain keywords
    keyword_comments = df[df['Review Text'].str.contains('|'.join(searchwords), na = False)]
    # Prints length of dataframe and the new filtered data frame to see if sample size is big enough to create meaningful results.
    st.write('The length of all the products is',len(df),'vs the length of reviews that mention [',word,']',len(keyword_comments))
    # Calculates and prints the average star review
    st.write('\nThe avearge star for comments with',word,'is',round(keyword_comments['Star Rating'].mean(),2))
    st.write('\nThe avearge star rating for all comments is',round(df['Star Rating'].mean(),2))
    st.write('\nThis is a difference of',round(df['Star Rating'].mean()-keyword_comments['Star Rating'].mean(),2),'stars')
    # Allows user to print and read some reviews
    print_reviews=st.radio("Do you want to print reviews containing these keywords?",['Yes','No'],index=1)
    # If yes, let user select how many reviews they want to see
    if print_reviews == 'Yes' and len(keyword_comments != 0):
        if len(keyword_comments) >= 5: 
            num = st.slider("How many reviews do you want to see?",min_value=5,max_value=len(keyword_comments),step=1)
        elif len(keyword_comments) < 5: 
            num = st.slider("How many reviews do you want to see?",min_value=1,max_value=len(keyword_comments),step=1)
        for i in range(num):
            st.write(i+1,keyword_comments['Review Text'].iloc[i],'(Rating:', keyword_comments["Star Rating"].iloc[i],')')
       
    
# Finds all color ways for any SKU
def FindAllSKUs(DF,MN):
    """
    Uses the model number to extract the reviews of the same product in all of the color ways.
    
    Parameters
        DF(dataframe): Dataframe must contain column "Model Number" 
        MN(string): Model number for product of interest
        
    Returns 
        Dataframe of all reviews of intrest
        
    Useful to increase the sample size you are dealing with.
    """
    # Initialize list to store model numbers and inputing the first one
    all_MN = [MN]
    # If the model number contains a '-' it gets broken up and the first part is extracted. 
    # Thats the first part is the parents for all sub SKUs
    if '-' in MN:
        beg = MN.split('-',1)
        beg = beg[0]
    # Handles the cases when the Model number does not have a '-' 
    else:
        beg=MN
    for num in DF['Model Number']:
        if beg in str(num):
            # Store all model numbers in list
            all_MN.append(num)
    # Extract all reviews with model numbers in the list
    filter = list(set(all_MN))
    # return DF[DF['Model Number'].isin(filter)]
    return filter
# Orgainzes and plots star ratings
def PlotRatings(rating):
    """
    Plots the star rating distribution (1-5).
    
    Parameters: 
        rating(pandas series (int)): Star Rating column of DF
        
    Returns:
        Barplot of ratings
        
    Useful for an easy way to plot and see the rating distribution for a given product.
    """
    # Counts how many of each rating their is
    score = rating.value_counts()
    # Sorts the ratings
    score=score.sort_index()
    # Plots the ratings 
    fig, ax = plt.subplots()
    sns.barplot(score.index, score.values, alpha=0.8)
    plt.title("Star Rating Distribution")
    plt.ylabel('count')
    plt.xlabel('Star Ratings')
    rects = ax.patches
    labels = score.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    st.pyplot(fig)

# To have work need to also use textblob (idk if I want more packages)
# Rethink keeping this
# def OverallSentiment(reviews):
#     """
#     Takes in review text and outputs a pie chart of the sentiment around all of the given review.    
    
#     Parameters:
#         reviews(pandas series (string)): review text of product of intrest
        
#     Returns:
#         Pie chart of the Negatice, Neutral and Positive sentiment of the reviews
        
#     Useful for a quick graphic around general sentiment of a product / feature
#     """
#     # Categorize polarity into positive, neutral or negative
#     labels = ['Negative','Neutral','Positive']
#     # Initialize cont array
#     values = [0,0,0]
#     # Categorize each review
#     for review in reviews:
#         sentiment = TextBlob(review)
#         # Custom formula to convert polarity
#         # 0 - (Negative) 1-(Netural) 2-(Positive)
#         polarity = round((sentiment.polarity+1)*3)%3
#         # add the summary array
#         values[polarity] = values[polarity] + 1
#     colors=['tomato','Orange','palegreen']
#     # explsion
#     explode = (0.05,0.05,0.05)
#     # Plot a pie chart
#     plt.pie(values, labels=labels, colors=colors,\
#            autopct='%1.1f%%', shadow=True, startangle=140, explode = explode,pctdistance=0.85)
#     # draw circle, for donut shape
#     centre_circle = plt.Circle((0,0),0.7,fc='white')
#     fig = plt.gcf()
#     fig.gca().add_artist(centre_circle)
#     plt.axis('equal') 
#     plt.savefig('Sentiment_pie_chart.png') #  uncomment to save image
#     plt.show()
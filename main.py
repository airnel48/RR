# Imports
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit import components
from streamlit_tags import st_tags, st_tags_sidebar
from models import TermExtration, LDA, CompareStars,FindAllSKUs, PlotRatings

#   Notes:
#   ----------------------------------------------------------------------------------
# - This code needs to have 2 other files in the same directory: models.py and clean.py
# - When running type "streamlit run .\main.py" for a streamlit webapp to open 
# - Some Code assumes column headers to be "Star Rating", "Review Text","Brand", "Model Number",and "Title"
#       so if the final df has different column heads the code will need to be changed
# - Must install spacy. might also have to run the following command: python -m spacy download en_core_web_sm
# - Performace can be more optimized with @st.cache (didn't get to impliment it lots of errors)
# - Streamlit Documentation: https://docs.streamlit.io/en/stable/api.html 
# - Code can be combined or split up more it'll work either way 


# Read in delta product list (This is used in lots of places in the code) file must be in the same folder
product_line = pd.read_csv("mtrl_mstr.csv")
# Drop down boxes for prouduct filters 
def FaucetFilter():
    """
    Collection of if statements to properly filter any delta df by model number
    """
    # Initialize variables so if statements work
    function, sub = '<Select>', '<Select>'
    # Sidebar explanation                            
    st.sidebar.title("Filter Options") 
    st.sidebar.write("Leave as <Select> if you don't want to filter the dataset.")
    # Slider to let the user pick the star rating
    rating = st.sidebar.slider("Select Star Rating Range (X and under)",min_value=1,max_value=5,step=1,value=5)
    # Selection box for the product type (This choice opens up other selection boxes)
    product_type = st.sidebar.selectbox("Select product type",['<Select>', 'Faucet', 'Bathing'],index=0)
    # If statements for product type (basesd of mtrl_mstr.csv)
    if product_type == 'Faucet':
        function = st.sidebar.selectbox("select function",
            ['<Select>','Kitchen','Lavatory','Showering Component','Tub / Shower']) # product_line[product_line['sub]]
        if function == 'Kitchen':
            sub = st.sidebar.selectbox("Select Sub Function",
                ['<Select>','Pull-Down','Deck-Mount','Pull-Down Prep','Pull-Out'])
        elif function == 'Lavatory':
            sub = st.sidebar.selectbox("Select Sub Function",
                ['<Select>','Centerset','Single Hole','Widespread'])
    elif product_type == 'Bathing':
          function = st.sidebar.selectbox("select function",
            ['<Select>','Shower','Bathtub'])
    # Text input to filter SKU
    SKU = st_tags_sidebar(label='Enter SKU of a Detla product',text='Type Here')
    return product_type, function, sub, rating,SKU
# Function to filter reviews df
def FilterDF(df,type,function,sub,rating,collection,SKU):
    """
    Filters df based off of user inputs to product type. All based on "mtrl_mstr" dataset
    """
    # Filter rating (COULD CAUSE ERROR IF DATASET HAS DIFF COL NAME)
    df = df[df["Star Rating"]<=rating]
    # Filters SKU (calls FindAllSKUs function to get all colors ways, Only Delta products)
    if SKU != []:
        AllMN = FindAllSKUs(df,SKU[0])
        df = df[df['Brand']=='Delta']
        df = df[df['Model Number'].isin(AllMN)]
    # Filters Product Type (Probably not the most effective but works fine)
    if collection != '<Select>':
        key_MN = product_line[(product_line["SERIES"] == collection)]
        df = df[df['Model Number'].isin(list(key_MN.NM.values))]
    if type == '<Select>' and function == '<Select>' and sub == '<Select>':
        return df
    elif function == '<Select>' and sub == '<Select>':
        key_MN = product_line[(product_line["TYP_DESC"] == type)]
        filtered_df = df[df['Model Number'].isin(list(key_MN.NM.values))]
        return filtered_df
    elif sub == '<Select>':
        key_MN = product_line[(product_line["TYP_DESC"] == type) & (product_line["FX_DESC"] == function)]
        filtered_df = df[df['Model Number'].isin(list(key_MN.NM.values))]
        return filtered_df
    else:
        key_MN = product_line[(product_line["TYP_DESC"] == type) & (product_line["FX_DESC"] == function) & (product_line["SUBFX_DESC"] == sub)]
        filtered_df = df[df['Model Number'].isin(list(key_MN.NM.values))]
        return filtered_df
# Function to run generic models
# @st.cache() Doesn't work yet 
def RunningModels(df):
    # Have User Select model to run 
    model = st.selectbox("What model would you like to run?",
                        ['<Select>','Key Word Extraction', 'Topic Modeling'],index=0)
    if model is not None:
        # Calls and Filters DF 
        df = FilterDF(df,product_type,function,sub,rating,collection,SKU)
        # Warning check to make sure filtered DF has over 20 samples 
        if len(df) < 20:
            st.warning("Small Sample Size, take results lightly")
        # Error check to make sure DF is strings (CAN DELETE THIS IF CONNECTED TO CDB)
        if df[select_col].dtypes == 'int64':
            st.warning("Please select a column that has textual data.")
        # Runs key word extration model
        if model == 'Key Word Extraction':
            # To prevent error warning from showing up
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(TermExtration(df[select_col]))
            with st.expander("See explanation"):
                st.write("""
                The graph above is the top 15 words by term weight.
                """)
        # Runs Topic Model
        elif model == 'Topic Modeling':
            # Progress bar
            st.spinner()
            with st.spinner(text = 'In progress...'):
                html_string = LDA(df[select_col],5)
                components.v1.html(html_string,width=1300,height=800,scrolling=True)
                # To write LDA explatnation
                with st.expander("See explanation"):
                    st.write("""
                    This model is a Latent Dirichlet Allocation model. 
                    The right hand side shows the 30 most frequent terms within the text. The terms are then split into clusters that you can see on the left.
                    You can click on a word and see what clusters it's apart of or click on a cluster and see what words are grouped together. 
                    The adjust relevance metric on the top changes the frequency of the terms shown. Ideal metric is around 0.4-0.6.
                    """)
        if model != "<Select>":
             DetailedModel(df) 
    # Prints dataframe
    st.markdown("## Filtered DataFrame with length {}".format(len(df)))
    st.dataframe(df)
# Functon to run detailed models to build off of generic models
def DetailedModel(df):
    """
    This function is called after user picks and runs a model.
    This is used to allow a deeper analysis.
    """
    st.markdown("# Below you can dive deeper into the reviews.")
    ####### Commented out code is for running sentiment model
    # Pick Model
    # detailed_model = st.selectbox("Select a model to run",
    #                  ['<Select>','Compare Stars of key words / Read Reviews','Overall Sentiment'],
    #                  index = 0)
    # if detailed_model != '<Select>':
        # if detailed_model == 'Compare Stars of key words / Read Reviews':
            # User input for CompareStars
    # If uncomment other code this will need to be indented
    user_input = st_tags(label='Enter Keywords (all lower case)',text='Type Here')
    if user_input is not "": # If user enters anything -> run
        # Function call from models.py notebook
        CompareStars(df,list(user_input))

        # elif detailed_model == 'Overall Sentiment':
        #     # Function call from models.py
        #     st.pyplot(OverallSentiment(df[select_col]))
# Function to plot star rating distibution
def PlotRating(df):
    df = FilterDF(df,product_type,function,sub,rating,collection,SKU)
    # Warning if sample size is small and it won't plot
    if len(df) < 20:
        st.warning("Sample size to small to plot")
    else:  
        # Call function from model.py
        PlotRatings(df['Star Rating'])     
# Has 500 collecitons (probably a better way to do this)
def SortCollection(df):
    # Uses product list to find all collections (which is named 'SERIES')
    every_collection = product_line['SERIES'].value_counts()
    # Slider for user to pick collection
    collection = st.sidebar.selectbox("Select Collection (Type to search):", every_collection.index.insert(0,'<Select>'),index = 0)
    return collection
# EXtra filters to get more specfic
def ExtraFilters(df):
    # st.sidebar.markdown('## Other Filters')
    # Sidebar selections
    all_finishes = product_line['FINSH'].value_counts().index
    finish = st.sidebar.selectbox("Finish (Type to search):", all_finishes.insert(0,'<Select>'),index=0) # Find best way top filter this
    touch = st.sidebar.radio("Touch feature:",['Yes','No'],index=1)
    voice = st.sidebar.radio("Voice Feature:",['Yes','No'],index=1)
    st.sidebar.markdown('## Non Delta Specific Filters')
    retailer = st.sidebar.selectbox("Retailer:",['<Select>','Amazon','Build','Home Depot','Lowes','Wayfair'],index=0)    
    brand = st.sidebar.selectbox("Brand:",['<Select>','Delta','Moen','Vigo Industries','Kraus','Kohler','Glacier Bay','Kingston Brass','DreamLine','Peerless'])
    open_search = st_tags_sidebar(label = "Title Search",text = "Type Here")
    # Takes user input and filters df
    if finish != '<Select>':
        key_MN = product_line[(product_line["FINSH"] == finish)]
        df = df[df['Model Number'].isin(list(key_MN.NM.values))]
    if touch == 'Yes':
        key_MN = product_line[(product_line["FIT_TCH_TECH"] == 'T')]
        df = df[df['Model Number'].isin(list(key_MN.NM.values))]
    if voice == 'Yes':
        key_MN = product_line[(product_line["FIT_TCH_20_VOICE"] == 'T')]
        df = df[df['Model Number'].isin(list(key_MN.NM.values))]
    if retailer != '<Select>':
        df = df[df['Retailer'].str.contains(retailer)]
    if brand != '<Select>':
        df=df[df['Brand']==brand]
    if open_search is not None:
        df = df[df['Title'].str.contains('|'.join(open_search),na=False)]
    return df


# MAIN BLOCK OF CODE WHERE ALL The WORK is DONE 
  
# Writes title / describes tool
st.title("Consumer Review Analyzer")
st.markdown("""This tool is meant to analyze consumer ratings and review data.
            After importing the rating and review data you will be able to filter the data in a variety of ways on the left hand side of the web page.
            When you apply filters it will be applied to the dataset at the bottom of the page to show you the amount of reviews within the filter.
            After filting the data you can select between two main models.""")
            # \n- __Key Word Extration:__ This model will take all of the reviews from the filtered dataframe and plot the 15 words with the largest weights.
            # \n- __Topic Modelings:__ This model takes all of the review text and organizes it into 5 different topics (clusters). This is an interative model where the right will show the most common words and the left will be 5 clusters which will be constructed with words used in a similar contex, creating 5 distinct topics.
            # \n  - This model is computationally expensive so it works best if the data is filtered first, it might take a couple minutes.
            # \nAfter running one of these models a new section will open up allowing for a more detailed analysis.
            # \n- __Compare Star Rating:__ This model will allow you to enter any words you want and it will report back the star rating difference for when that word is used in reviews vs all reviews of that filtered dataset.
            #   If you want to dive deeper you can then look click a button to read all reviews containing the words you typed.
            # \n- __Overall Sentiment:__ Pie chart displaying the sentiment of all the reviews in the filtered dataframe.
            # \nHave Fun! :sunglasses:""")
# Lets user pick a file and prints the dataframe (THIS WILL CHANGE TO AZURE PULL WITH PYODBS WILL JUST CALL azure_conection.py)
filename = st.file_uploader("Select a file with text you want to visualize",type=['csv']) 
# Checks to make sure user picks a file then runs rest of code
if filename is not None:
    # Read in and print dataframe
    df = pd.read_csv(filename)
    st.dataframe(df)
    # Lets user pick what column to analyze (IF WE USE AZURE CONNECTION PRE SET IT TO REVIEW TEXT)
    # select_col = st.selectbox("What column would you like to analyze?",(df.columns), index = 0)
    select_col = "Review Text" # Un comment this, VERY HARD CODED FOR DEMO
    # Creates sidebar and gives options for user to filter the DF
    product_type, function, sub, rating, SKU = FaucetFilter()
    # Allows user to sort by collection
    collection = SortCollection(df)
    # Other filters (Finish, Touch, Voice, Retailer, Brand)
    df = ExtraFilters(df)
    # Plots star rating 
    PlotRating(df)
    # Runs all of the models  
    RunningModels(df)
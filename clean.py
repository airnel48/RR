import pandas as pd
import en_core_web_sm

# Main function, cleans sereies of text
def CleanData(text):
    # remove unwanted characters, numbers and symbols
    text = text.str.replace("[^a-zA-Z#]", " ",regex = True)
    # remove short words (length < 3)
    text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    # remove stop words
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'sink','also','good','like','great','come','would','faucet','faucets','shower','delta','water','amazon','delta','the','Delta','want','wanted','wish','desired','one','glass','rinser','head','seat','toilet','kitchen']
    reviews = [remove_stopwords(r.split(),stop_words) for r in text]
    # make entire text lowercase
    reviews = [r.lower() for r in reviews]
    # tokenzies text
    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    # call lemmatization function
    reviews = lemmatization(tokenized_reviews)
    return reviews

# picking english for cleaning models to use (using lematizer, pos tagging, and flitering nouns and adjectives)
nlp = en_core_web_sm.load(disable=['parse','ner'])
# function to lemmatize the data 
def lemmatization(text,tags=['NOUN','ADJ']):# filter noun and adjective
    output = []
    for sent in text:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

# function to remove stopwords
def remove_stopwords(rev,stop_words):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new
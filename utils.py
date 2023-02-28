import spacy
import pickle
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)
model = spacy.load("en_core_web_sm")
topics ={
    0 : 'bank account services',
    1 :'credit_card',
    2:'mortgage/loans',
    3:'theft/dispute reporting',
    4:'others'
}


def pre_process_text(text):
  """
  This function is to clean the text and remove punctuations,digits, text in square brackets and masked characters.
  """
  text = text.lower()
  text = re.sub(r'\[.*?\]','',text)
  text = re.sub(r'[^\w\s]', '',text)
  text = re.sub(r'[\d]','',text)
  text = re.sub(r'[*x]?','',text)
  logging.info(f"Pre-processed text: {text}")

  return text

def get_lemma(text):
  """
  This function returns only noun words
  """
  modified_text =" "
  tokens = model(text)
  for token in tokens :
    if(token.tag_=='NN'):
       modified_text = modified_text + token.lemma_ +" "

  logging.info(f"After lemmatization : {text}")
  return modified_text[1:]

def get_tf_idf_features(text):
  logging.info("Applying tf_idf transformation")
  with open('tf_idf.pkl','rb') as f:
    tf_idf = pickle.load(f)
  X = tf_idf.transform([text])
  return X.toarray()

def get_predict(X):
  with open('model.pkl','rb') as f:
    model = pickle.load(f)
    y=model.predict(X)
    logging.info(f"prediction: {y}")
    return topics[y[0]]
  
def predict_class(text):
  preprocessedText=pre_process_text(text)
  lemmatizedText=get_lemma(preprocessedText)
  X = get_tf_idf_features(lemmatizedText)
  category=get_predict(X)
  logging.info(f"Input Text: {text} \n")
  logging.info(f"Final Prediction: {category}")
  return category
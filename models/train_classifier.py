import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    """Loads data from database to dataframe and extracts X and Y values from it"""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df[['id', 'message', 'original', 'genre']]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, df

def tokenize(text):
    """Takes plaintext messages and returns the key words in lower case"""
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """Constructs the ml pipeline and performs grid search for the best parametres"""
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'moc__estimator__n_estimators': [10],
        'moc__estimator__min_samples_split': [2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Calculates and displays various performance metrics"""
    Y_pred = model.predict(X_test['message'])

    tp = np.logical_and((Y_test == 1), (Y_pred == 1)).sum()
    fp = np.logical_and((Y_test == 0), (Y_pred == 1)).sum()
    fn = np.logical_and((Y_test == 1), (Y_pred == 0)).sum()

    precision_score = tp / (tp + fp).mean()
    recall_score = tp / (tp + fn).mean()

    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    f1_score = f1_score.mean()

    accuracy_score = (Y_pred == Y_test).mean()
    
    print("Accuracy", accuracy_score.mean())
    print("Precision", precision_score.mean())
    print("Recall", recall_score.mean())
    print("F1", f1_score.mean())

    # The reason I hand wrote my evaluation is because I keep getting this error,
    # ValueError: multiclass-multioutput is not supported,
    # when I write things like this: 
    # print(classification_report(Y_test, Y_pred))

def save_model(model, model_filepath):
    """Saves the trained model in a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

import pandas as pd
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pickle
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

class LabelTopics(object):
    
    def __init__(self, data=None, captions=None, session_ids=None):
        '''
        This object will take the output of a well-formed API call from OMF 
        and allow the user to manipulate the data to evenutally end up with assigned 
        topics for each session in the response. The simplest use case of this functionality
        looks something like:
        
        lt = LabelTopics(data=data)
        lt.clean_data()
        lt.predict_fluff()
        lt.predict_topic_labels()
        session_topics = lt.session_probs
        
        Each of the functions can be somewhat customized with different hyperparameters
        based on what you find works best. The Delta team has tried to document
        what these different parameters control and whether it requires model retraining
        but if you have any questions feel free to contact gcmac[at]fastmail[.]com
        
        Args:
            data: a json like string or json object that contains the response of a successful
                  call to the current OMF API. This json should have a top level key 'results'
                  and within that object have the keys 'caption' and 'session_id'
            captions: an iterable of string types containing the text you want to classify
            session_ids: an iterable of session_ids where session_ids[i] corresponds to the entry
                         of captions[i]
            
        Either (data) or (captions AND session_ids) must be included for the prediction process
        to be executed.
        '''
        
        self.data_type = 'list' if data is None else 'other'
        self.data = self.check_data(data, captions, session_ids)
        
        # handle json data  
        if self.data_type == 'other':
            self.results = self.data['results']
            self.captions = [res['caption'] for res in self.results]
            self.sessions = [res['session_id'] for res in self.results]
        
        else:
            if session_ids is None:
                raise ValueError("If passing in list of topics, need corresponding session ids")
            self.captions = captions
            self.sessions = session_ids
            
        self.fluff_labels = []
        self.topic_labels = []
        self.clean_captions = []
        
        self.fluff_vectorizer = pickle.load(open('tfidf_vectorizer_obj.pkl','rb'))
        self.fluff_model = pickle.load(open('fluff_model','rb'))
        
        self.lda_vectorizer = pickle.load(open('tfidf_model_for_lda.pkl','rb'))
        self.lda_model = pickle.load(open('lda_model.pkl', 'rb')) 
        
        self.topicid_2_label = {0:'water, transportation',
                               1:'transit',
                               2: 'service',
                               3:"health",
                               4:"zoning",
                               5:"license",
                               6:'crime',
                               7: 'transit',
                               8:'law',
                               9:'public_space',
                               10:'housing',
                               11:'community',
                               12:'education',
                               13: 'procedural',
                               14:'budget',
                               15: 'procedural',
                               16:'zoning',
                               17:'espanol',
                               18:'procedural',
                               19:'housing'}
        
    def clean_data(self, stop_word_set=set(stopwords.words('english')),
                   remove_punc=True, lemmatize=True):
        '''
        Function that takes the raw captions and transforms them into 
        their cleaned counterpart. 
        
        **IMPORTANT**
        The default values for this function are set based on what choices the Delta 
        Analytics team made when building this application. They can be changed, but
        if they are you will need to generate a new TfidfVectorizer object and subsequently
        train a new fluff model and LDA model.
        
        args:
            stop_word_set (iterable): list of words for the model to remove
                                      typically words like 'a', 'the', 'and', etc.
            remove_punc (boolean): flag for whether punctuation should be removed
            lemmatize (boolean): flag for whether captions should be tokenized
            
        '''
        
        stop_word_set = stop_word_set if isinstance(stop_word_set, set) else set(stop_word_set)
        lemmatizer = WordNetLemmatizer()
        
        self.clean_captions = [self._clean(caption, stop_word_set, remove_punc, lemmatize, lemmatizer) 
                               for caption in self.captions]
        
    def predict_fluff(self, threshold=.5, ):
        '''
        Method to predict the probability that each caption is fluff and does not contain 
        text relate to any specific policy topic. 
        
        args:
            threshold (float): Floating point number between 0 and 1 that represents
                               the probability above which a caption is labeled as fluff.
                               The default value is .5 so any predictions > .5 are labeled 
                               as fluff.
            fluff_vec_path (str): Path to the tidf vectorizer that will transform the clean
                                  captions to numbers to be consumed by the model.
        '''
        fluff_vectorized_captions = self._text2mat(self.clean_captions, 
                                                   self.fluff_vectorizer)
        self.fluff_prob_preds = self.fluff_model.predict_proba(fluff_vectorized_captions)[:,1]
        self.fluff_bin_preds = [1 if p > threshold else 0 for p in self.fluff_prob_preds]
    
    def check_data(self, data, captions, session_ids):
        '''
        Function that makes sure input data is suitable for task
        '''
        # Make sure we either have json data or captions and sessions
        if data is None and (captions is None or session_ids is None):
            raise ValueError("Must pass in either json response or iterables of captions and session ids")
        
        if self.data_type == 'other':
            if isinstance(data, str):
                data = json.loads(data)
                return data

            if not isinstance(data, dict):
                raise ValueError('Data must be a JSON like object')

            if 'results' not in data.keys() or len(data['results']) == 0:
                raise ValueError('Passed in data with no results')

            return data
        
        else:
            for i,cap in enumerate(captions):
                if not isinstance(cap, str):
                    raise ValueError(f'Non string value ({type(cap)}) in list of captions at position {i}')
            return data
        
    def predict_topic_labels(self, prob_th=.65, topic_th=.1):
        '''
        Function that predicts the labels for each topic. 
        
        args:
            prob_th: Float between 0 and 1 that determines when a label is assigned
                             as a 1. Similar logic to threshold in predict_fluff method.
            topic_th: Float between 0 and 1 that determines when a session is assigned
                      a certain topic.
        '''
        
        lda_vectorized_captions = self._text2mat(self.clean_captions, 
                                                 self.lda_vectorizer)
        self.topic_preds_by_cap = self.lda_model.transform(lda_vectorized_captions)
        
        high_prob_preds = defaultdict(list)
        self.sess_labels = {sid:[] for sid in set(self.sessions)}
        
        i = 0
        for pred_list, sid in zip(self.topic_preds_by_cap, self.sessions):
            if np.max(pred_list) > prob_th and self.fluff_bin_preds[i] == 0:
                high_prob_preds[sid].append(np.round(pred_list))
            i += 1
        
        for sid in high_prob_preds.keys():
            mean_probs = np.mean(high_prob_preds[sid], axis=0)
            common_topics = [(mp,i) for (i,mp) in zip(mean_probs, range(len(mean_probs))) if i >= topic_th]
            common_topics = sorted(common_topics, key=itemgetter(1), reverse=True)
            self.sess_labels[sid].extend([self.topicid_2_label[i] for (i,j) in common_topics])
            
    def get_w2v_predictions(self, w2v_model, session_thresh=.1, 
                            pickle_path='topics_dict.pkl'):
        '''
        Function that generates the word2vec predictions. Given a word2vec model and a path
        to a pickled dictionary where keys are topics and values are list of words associated
        with that topic, match sessions to topics.
        '''
        self.w2v_model = w2v_model
        self.topics_dict = pickle.load(open(pickle_path,'rb'))
        self.topic_labels = list(self.topics_dict.keys())
        self.topic_word_vecs = list(self.topics_dict.values())
        
        self.caption_w2v_preds = np.array([self._max_cosine_similarity(cc, self.topic_word_vecs)
                                           for cc in self.clean_captions])
 
        cap, label = np.where(self.caption_w2v_preds > .75)
        
        self.cap_pos_labels = defaultdict(list)
        for c,l in zip(cap, label):
            self.cap_pos_labels[c].append(self.topic_labels[l])
        
        self.sess_w2v_labels = defaultdict(list)
        for i, sid in enumerate(self.sessions):
            self.sess_w2v_labels[sid].extend(self.cap_pos_labels[i])

        self.sess_w2v_labels = {sid:(len(labels), Counter(labels)) \
                                for sid, labels in self.sess_w2v_labels.items()}
        
        for sid, label_tup in self.sess_w2v_labels.items():
            count_total = label_tup[0]
            count_dict = label_tup[1]
            pct_dict = {topic:count_topic/count_total for topic, count_topic in count_dict.items()}
            self.sess_w2v_labels[sid] = pct_dict
        
        high_pct_dict = {}
        for sid, topic_dict in self.sess_w2v_labels.items():
            high_prob_topics = [topic for (topic, label_prob) in topic_dict.items() 
                                if label_prob > session_thresh and topic != 'plenary']
            high_pct_dict[sid] = high_prob_topics
        
        self.sess_w2v_labels = high_pct_dict

    def _clean(self, cap, stop_word_set, remove_punc, lemmatize, lemmatizer):
        '''
        Delegated function for cleaning a single caption. Should not
        be accessed directly but rather through the clean_data() function.
        Arguments are described in that function
        '''
        cleaned_cap = ' '.join([word for word in cap.lower().split() if word not in stop_word_set])
        
        if remove_punc:
            cleaned_cap = ''.join([char for char in cleaned_cap if char not in set(string.punctuation)])

        if lemmatize:
            cleaned_cap = ' '.join([self._lemmatize(word, lemmatizer) for word in cleaned_cap.split()])
        
        return cleaned_cap

    def _lemmatize(self, word, lemma):
        '''
        This function will take a single word and return the lemma of it
        Examples:
            running -> run
            awkwardly -> awkward
        
        Transforming text in this way allows for better topic prediction
        as topics will probably contain the same root words and we don't 
        care about the tense the topic is being discussed in.
        
        args:
            word (str): word that we want to transform
            lemma (WordNetLemmatizer): Lemmatizer object for transforming word
                                       to root of itself
        '''
        lemmatized = lemma.lemmatize(word, "n")
        if lemmatized == word:
            lemmatized = lemma.lemmatize(word, "v")
        if lemmatized == word:
            lemmatized = lemma.lemmatize(word, "r")
        return lemmatized
    
    def _max_cosine_similarity(self, caption, topics):
        '''
        Find the captions related to topics too small for LDA to catch by measuring the
        caption's word similarities to common words from other topics.
        '''
        # Convert captions to word2vec
        caption = [word for word in caption.split() if word in self.w2v_model.vocab]
        caption = [self.w2v_model[word] for word in caption]

        # Calculate max cosine similarity
        try:
            sim = [np.amax(cosine_similarity(caption, t)) for t in topics]
            if len(sim) == 0:
                sim = [0]*15
        except:
            sim = [0]*15

        return np.array(sim)
    
    def combine_preds(self):
        '''
        Combine the word2vec and LDA predictions
        '''
        self.topic_labels = defaultdict(list)
        
        for k in self.sess_labels.keys():
            try:
                self.topic_labels[k] = set([lab for lab in self.sess_labels[k] if lab != 'procedural'] \
                                       + [lab for lab in self.sess_w2v_labels[k] if lab != 'procedural'])
            except:
                self.topic_labels[k] = set([lab for lab in self.sess_labels[k] if lab != 'procedural'])
    
    @staticmethod
    def _text2mat(clean_caps, vectorizer):
        return vectorizer.transform(clean_caps).todense()
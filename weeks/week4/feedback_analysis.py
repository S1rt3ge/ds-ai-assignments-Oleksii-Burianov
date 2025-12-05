import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


class CustomerFeedbackAnalyzer:
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.lda_model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # stopwords for social media
        self.custom_stopwords = set(ENGLISH_STOP_WORDS).union({
            'rt', 'amp', 'http', 'https', 'co', 'im', 'dont', 'cant', 'wont',
            'aint', 'gonna', 'gotta', 'u', 'ur', 'ya', 'yall', 'em', 'da',
            'dat', 'dis', 'dey', 'dem', 'lol', 'lmao', 'lmfao', 'smh', 'tbh',
            'idk', 'imo', 'btw', 'omg', 'wtf', 'like', 'just', 'got', 'get',
            'know', 'want', 'think', 'look', 'make', 'way', 'come', 'say',
            'let', 'need', 'try', 'ask', 'tell', 'feel', 'give', 'call'
        })
        
    def load_data(self) -> pd.DataFrame:
        print("Loading data")
        
        self.df = pd.read_csv(self.data_path)
        
        class_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
        self.df['class_name'] = self.df['class'].map(class_mapping)
        
        print(f"Dataset: {self.df.shape[0]} samples")
        print(f"Classes: {dict(self.df['class_name'].value_counts())}")
        
        return self.df
    
    def preprocess_text(self, text: str) -> dict:
        """Text preprocessing: lowercase, remove URLs/mentions, tokenize, remove stopwords"""
        if pd.isna(text) or not isinstance(text, str):
            return {'clean_text': '', 'tokens': []}
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'&amp;', 'and', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        
        clean_tokens = []
        for token in tokens:
            if token not in self.custom_stopwords and len(token) > 2 and token.isalpha():
                if token.endswith('ing') and len(token) > 5:
                    token = token[:-3]
                elif token.endswith('ed') and len(token) > 4:
                    token = token[:-2]
                clean_tokens.append(token)
        
        return {'clean_text': ' '.join(clean_tokens), 'tokens': clean_tokens}
    
    def apply_preprocessing(self):
        print("Text preprocessing")
        
        processed = self.df['tweet'].apply(self.preprocess_text)
        self.df['clean_text'] = processed.apply(lambda x: x['clean_text'])
        self.df['tokens'] = processed.apply(lambda x: x['tokens'])
        
        original_len = len(self.df)
        self.df = self.df[self.df['clean_text'].str.len() > 0].reset_index(drop=True)
        
        print(f"Samples: {original_len} -> {len(self.df)} (removed {original_len - len(self.df)})")
            
    def analyze_sentiment_vader(self, text: str) -> dict:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
            
        return {
            'vader_compound': compound,
            'vader_pos': scores['pos'],
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_label': label
        }
    
    def analyze_sentiment_textblob(self, text: str) -> dict:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = 'POSITIVE'
        elif polarity < -0.1:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
            
        return {
            'textblob_polarity': polarity,
            'textblob_subjectivity': subjectivity,
            'textblob_label': label
        }
    
    def apply_sentiment_analysis(self):
        print("Sentiment analysis")
        
        # VADER
        vader_results = self.df['tweet'].apply(self.analyze_sentiment_vader)
        for col in ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_label']:
            self.df[col] = vader_results.apply(lambda x: x[col])
        
        # TextBlob
        textblob_results = self.df['clean_text'].apply(self.analyze_sentiment_textblob)
        for col in ['textblob_polarity', 'textblob_subjectivity', 'textblob_label']:
            self.df[col] = textblob_results.apply(lambda x: x[col])
        
        # Ensemble
        def ensemble_sentiment(row):
            vader = row['vader_label']
            textblob = row['textblob_label']
            if vader == textblob:
                return vader
            if vader == 'NEUTRAL':
                return textblob
            if textblob == 'NEUTRAL':
                return vader
            return vader  # VADER for social media
        
        self.df['sentiment_ensemble'] = self.df.apply(ensemble_sentiment, axis=1)
        
        print(f"VADER: {dict(self.df['vader_label'].value_counts())}")
        print(f"TextBlob: {dict(self.df['textblob_label'].value_counts())}")
        print(f"Ensemble: {dict(self.df['sentiment_ensemble'].value_counts())}")
        
    def build_tfidf_features(self):
        print("TF-IDF vectorization")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            max_df=0.85,
            min_df=10,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['clean_text'])
        
        print(f"TF-IDF matrix: {self.tfidf_matrix.shape}")
        
        # top terms
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(self.tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_scores.argsort()[-10:][::-1]
        
        print("Top TF-IDF terms:", [feature_names[i] for i in top_indices])
            
        return [(feature_names[i], tfidf_scores[i]) for i in top_indices]
    
    def train_lda_topics(self, n_topics: int = 5):
        print(f"LDA topic modeling ({n_topics} topics)")
        
        count_vectorizer = CountVectorizer(
            max_features=1500,
            max_df=0.85,
            min_df=10,
            ngram_range=(1, 2)
        )
        count_matrix = count_vectorizer.fit_transform(self.df['clean_text'])
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            random_state=42,
            max_iter=25,
            n_jobs=-1
        )
        
        self.lda_model.fit(count_matrix)
        
        feature_names = count_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_name = f"Topic {topic_idx + 1}"
            topics[topic_name] = top_words
            print(f"{topic_name}: {', '.join(top_words[:6])}")
        
        # assign topics to documents
        doc_topics = self.lda_model.transform(count_matrix)
        self.df['dominant_topic'] = doc_topics.argmax(axis=1) + 1
        self.df['topic_confidence'] = doc_topics.max(axis=1)
            
        return topics
    
    def create_visualizations(self, output_dir: str):
        print("visualizations")
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # sentiment analysis dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # class distribution
        class_counts = self.df['class_name'].value_counts()
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, 
                       autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Original Class Distribution')
        
        # sentiment distribution
        sent_counts = self.df['sentiment_ensemble'].value_counts()
        colors_sent = {'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6', 'POSITIVE': '#27ae60'}
        axes[0, 1].bar(sent_counts.index, sent_counts.values, 
                       color=[colors_sent.get(x, '#3498db') for x in sent_counts.index])
        axes[0, 1].set_title('Sentiment Distribution (Ensemble)')
        axes[0, 1].set_ylabel('Count')
        
        # VADER scores
        axes[1, 0].hist(self.df['vader_compound'], bins=50, color='#3498db', edgecolor='white')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].set_title('VADER Compound Score Distribution')
        axes[1, 0].set_xlabel('Compound Score')
        
        # sentiment by class heatmap
        cross_tab = pd.crosstab(self.df['class_name'], self.df['sentiment_ensemble'], normalize='index')
        sns.heatmap(cross_tab, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1, 1])
        axes[1, 1].set_title('Sentiment by Original Class')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # word clouds
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (class_name, cmap) in enumerate([
            ('hate_speech', 'Reds'),
            ('offensive_language', 'Oranges'),
            ('neither', 'Greens')
        ]):
            text = ' '.join(self.df[self.df['class_name'] == class_name]['clean_text'])
            if text:
                wc = WordCloud(width=600, height=400, background_color='white',
                              colormap=cmap, max_words=100).generate(text)
                axes[idx].imshow(wc, interpolation='bilinear')
                axes[idx].set_title(f'Word Cloud: {class_name}')
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wordclouds.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # topic distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        topic_counts = self.df['dominant_topic'].value_counts().sort_index()
        ax.bar([f'Topic {i}' for i in topic_counts.index], topic_counts.values, 
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(topic_counts))))
        ax.set_title('Document Distribution Across Topics')
        ax.set_ylabel('Number of Documents')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # class vs sentiment
        fig, ax = plt.subplots(figsize=(10, 8))
        confusion = pd.crosstab(self.df['class_name'], self.df['sentiment_ensemble'])
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Class vs Predicted Sentiment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_vs_sentiment.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Saved: sentiment_analysis.png, wordclouds.png, topic_distribution.png, class_vs_sentiment.png")
        
    def save_results(self, output_dir: str, topics: dict):
        print("saving results")
        
        # processed data
        output_cols = ['tweet', 'clean_text', 'class', 'class_name', 
                      'vader_compound', 'vader_label', 
                      'textblob_polarity', 'textblob_label',
                      'sentiment_ensemble', 'dominant_topic', 'topic_confidence']
        self.df[output_cols].to_csv(f'{output_dir}/processed_data.csv', index=False)
        
        # sentiment stats
        with open(f'{output_dir}/sentiment_stats.txt', 'w') as f:
            f.write("SENTIMENT ANALYSIS RESULTS\n\n")
            
            f.write("VADER:\n")
            for label, count in self.df['vader_label'].value_counts().items():
                f.write(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)\n")
            f.write(f"  Mean compound: {self.df['vader_compound'].mean():.3f}\n\n")
                
            f.write("TextBlob:\n")
            for label, count in self.df['textblob_label'].value_counts().items():
                f.write(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)\n")
            f.write(f"  Mean polarity: {self.df['textblob_polarity'].mean():.3f}\n\n")
                
            f.write("Ensemble:\n")
            for label, count in self.df['sentiment_ensemble'].value_counts().items():
                f.write(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)\n")
        
        # topics
        with open(f'{output_dir}/topics.txt', 'w') as f:
            f.write("KEY THEMES (LDA Topics)\n\n")
            for topic_name, words in topics.items():
                f.write(f"{topic_name}: {', '.join(words)}\n")
        
        # summary report
        with open(f'{output_dir}/analysis_report.txt', 'w') as f:
            f.write("CUSTOMER FEEDBACK ANALYSIS REPORT\n\n")
            
            f.write("1. DATASET\n")
            f.write(f"   Samples: {len(self.df)}\n")
            for cls, count in self.df['class_name'].value_counts().items():
                f.write(f"   {cls}: {count} ({count/len(self.df)*100:.1f}%)\n")
            
            f.write("\n2. SENTIMENT\n")
            for label, count in self.df['sentiment_ensemble'].value_counts().items():
                f.write(f"   {label}: {count} ({count/len(self.df)*100:.1f}%)\n")
            
            f.write("\n3. TOPICS\n")
            for topic_name, words in topics.items():
                f.write(f"   {topic_name}: {', '.join(words[:5])}\n")
            
        print("Saved: processed_data.csv, sentiment_stats.txt, topics.txt, analysis_report.txt")
        
    def run_full_analysis(self):
        
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        self.load_data()
        self.apply_preprocessing()
        self.apply_sentiment_analysis()
        self.build_tfidf_features()
        topics = self.train_lda_topics(n_topics=5)
        self.create_visualizations(output_dir)
        self.save_results(output_dir, topics)
        
        return self.df


if __name__ == "__main__":
    analyzer = CustomerFeedbackAnalyzer('labeled_data.csv')
    results = analyzer.run_full_analysis()

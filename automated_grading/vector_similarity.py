"""
Vector-Based Similarity Module
Implements Latent Semantic Analysis (LSA) for text similarity.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import re


class LSAModel:
    """Latent Semantic Analysis model for text similarity."""
    
    def __init__(self, n_components=100):
        """
        Initialize LSA model.
        
        Args:
            n_components (int): Number of dimensions for LSA
        """
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000
        )
        self.svd = None
        self.is_fitted = False
    
    def fit(self, documents):
        """
        Fit the LSA model on a corpus of documents.
        
        Args:
            documents (list): List of text documents
        """
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Apply SVD (LSA)
        n_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components > 0:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
            self.is_fitted = True
    
    def transform(self, documents):
        """
        Transform documents into LSA space.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            numpy.ndarray: LSA representations
        """
        if not self.is_fitted:
            return None
        
        tfidf_matrix = self.vectorizer.transform(documents)
        lsa_matrix = self.svd.transform(tfidf_matrix)
        return lsa_matrix
    
    def similarity(self, doc1, doc2):
        """
        Calculate LSA-based cosine similarity between two documents.
        
        Args:
            doc1 (str): First document
            doc2 (str): Second document
            
        Returns:
            float: Cosine similarity score
        """
        if not self.is_fitted:
            return 0.0
        
        lsa_vecs = self.transform([doc1, doc2])
        sim = cosine_similarity(lsa_vecs[0:1], lsa_vecs[1:2])[0][0]
        return float(sim)


def simple_tfidf_similarity(text1, text2):
    """
    Calculate TF-IDF based cosine similarity between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Cosine similarity score
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except:
        return 0.0


def compute_vector_similarity(student_answer, model_answer, lsa_model=None):
    """
    Compute vector-based similarity measures.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model (correct) answer
        lsa_model (LSAModel, optional): Pre-fitted LSA model
        
    Returns:
        dict: Dictionary of similarity scores
    """
    results = {
        'tfidf_cosine': simple_tfidf_similarity(student_answer, model_answer)
    }
    
    if lsa_model and lsa_model.is_fitted:
        results['lsa_similarity'] = lsa_model.similarity(student_answer, model_answer)
    else:
        results['lsa_similarity'] = 0.0
    
    return results

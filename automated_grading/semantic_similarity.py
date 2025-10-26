"""
Semantic Similarity Module
Implements WordNet-based semantic similarity measures.
"""

import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


def download_nltk_resources():
    """Download necessary NLTK resources if not already available."""
    resources = ['wordnet', 'punkt', 'stopwords', 'wordnet_ic', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}' if resource in ['wordnet', 'stopwords', 'wordnet_ic'] else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def preprocess_for_semantic(text):
    """
    Preprocess text for semantic analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of words (lowercase, no stopwords)
    """
    # Tokenize
    words = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return words


def get_wordnet_synsets(word):
    """
    Get WordNet synsets for a word.
    
    Args:
        word (str): Input word
        
    Returns:
        list: List of synsets
    """
    return wordnet.synsets(word)


def path_similarity_score(word1, word2):
    """
    Calculate path similarity between two words using WordNet.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: Path similarity score (0-1)
    """
    synsets1 = get_wordnet_synsets(word1)
    synsets2 = get_wordnet_synsets(word2)
    
    if not synsets1 or not synsets2:
        return 0.0
    
    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            sim = syn1.path_similarity(syn2)
            if sim and sim > max_sim:
                max_sim = sim
    
    return max_sim


def lch_similarity_score(word1, word2):
    """
    Calculate Leacock-Chodorow similarity between two words.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: LCH similarity score (normalized 0-1)
    """
    synsets1 = get_wordnet_synsets(word1)
    synsets2 = get_wordnet_synsets(word2)
    
    if not synsets1 or not synsets2:
        return 0.0
    
    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            # Only compare synsets of the same POS
            if syn1.pos() == syn2.pos():
                try:
                    sim = syn1.lch_similarity(syn2)
                    if sim and sim > max_sim:
                        max_sim = sim
                except:
                    pass
    
    # Normalize (max LCH is around 3.6)
    return min(max_sim / 3.6, 1.0) if max_sim > 0 else 0.0


def lin_similarity_score(word1, word2):
    """
    Calculate Lin similarity between two words using WordNet.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: Lin similarity score (0-1)
    """
    # Load information content
    try:
        brown_ic = wordnet_ic.ic('ic-brown.dat')
    except:
        return 0.0
    
    synsets1 = get_wordnet_synsets(word1)
    synsets2 = get_wordnet_synsets(word2)
    
    if not synsets1 or not synsets2:
        return 0.0
    
    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            if syn1.pos() == syn2.pos() and syn1.pos() in ['n', 'v']:
                try:
                    sim = syn1.lin_similarity(syn2, brown_ic)
                    if sim and sim > max_sim:
                        max_sim = sim
                except:
                    pass
    
    return max_sim


def average_word_similarity(words1, words2, similarity_func):
    """
    Calculate average similarity between two lists of words.
    
    Args:
        words1 (list): First list of words
        words2 (list): Second list of words
        similarity_func: Similarity function to use
        
    Returns:
        float: Average similarity score
    """
    if not words1 or not words2:
        return 0.0
    
    total_sim = 0.0
    count = 0
    
    for w1 in words1:
        max_sim = 0.0
        for w2 in words2:
            sim = similarity_func(w1, w2)
            if sim > max_sim:
                max_sim = sim
        total_sim += max_sim
        count += 1
    
    return total_sim / count if count > 0 else 0.0


def compute_semantic_similarity(student_answer, model_answer):
    """
    Compute all semantic similarity measures.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model (correct) answer
        
    Returns:
        dict: Dictionary of similarity scores
    """
    # Ensure NLTK resources are available
    download_nltk_resources()
    
    # Preprocess
    student_words = preprocess_for_semantic(student_answer)
    model_words = preprocess_for_semantic(model_answer)
    
    # Calculate similarities
    path_sim = average_word_similarity(student_words, model_words, path_similarity_score)
    lch_sim = average_word_similarity(student_words, model_words, lch_similarity_score)
    lin_sim = average_word_similarity(student_words, model_words, lin_similarity_score)
    
    return {
        'path_similarity': path_sim,
        'lch_similarity': lch_sim,
        'lin_similarity': lin_sim
    }

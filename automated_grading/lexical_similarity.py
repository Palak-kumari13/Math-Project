"""
Lexical Similarity Module
Implements word overlap-based similarity measures for text comparison.
"""

import re
from collections import Counter


def preprocess_text(text):
    """
    Preprocess text by converting to lowercase and removing punctuation.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of words
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text)
    return words


def jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Jaccard similarity score (0-1)
    """
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def overlap_coefficient(text1, text2):
    """
    Calculate overlap coefficient between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Overlap coefficient (0-1)
    """
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    min_size = min(len(words1), len(words2))
    
    return len(intersection) / min_size if min_size > 0 else 0.0


def cosine_word_similarity(text1, text2):
    """
    Calculate word-level cosine similarity between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Cosine similarity score (0-1)
    """
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    if not words1 or not words2:
        return 0.0
    
    # Create word frequency vectors
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    
    # Get all unique words
    all_words = set(counter1.keys()).union(set(counter2.keys()))
    
    # Create vectors
    vec1 = [counter1.get(word, 0) for word in all_words]
    vec2 = [counter2.get(word, 0) for word in all_words]
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def compute_lexical_similarity(student_answer, model_answer):
    """
    Compute all lexical similarity measures.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model (correct) answer
        
    Returns:
        dict: Dictionary of similarity scores
    """
    return {
        'jaccard': jaccard_similarity(student_answer, model_answer),
        'overlap': overlap_coefficient(student_answer, model_answer),
        'cosine_word': cosine_word_similarity(student_answer, model_answer)
    }

"""
Quick test script for the automated grading modules.
"""

import sys

def test_lexical():
    """Test lexical similarity module."""
    print("Testing lexical similarity...")
    from lexical_similarity import compute_lexical_similarity
    
    text1 = "The cat sat on the mat"
    text2 = "A cat sat on a mat"
    
    result = compute_lexical_similarity(text1, text2)
    print(f"  Jaccard: {result['jaccard']:.3f}")
    print(f"  Overlap: {result['overlap']:.3f}")
    print(f"  Cosine: {result['cosine_word']:.3f}")
    print("  ✓ Lexical similarity working\n")


def test_vector():
    """Test vector similarity module."""
    print("Testing vector similarity...")
    from vector_similarity import compute_vector_similarity, LSAModel
    
    text1 = "The cat sat on the mat"
    text2 = "A cat sat on a mat"
    
    result = compute_vector_similarity(text1, text2)
    print(f"  TF-IDF Cosine: {result['tfidf_cosine']:.3f}")
    
    # Test LSA
    corpus = [
        "The cat sat on the mat",
        "A dog ran in the park",
        "Birds fly in the sky"
    ]
    lsa_model = LSAModel(n_components=2)
    lsa_model.fit(corpus)
    
    result_with_lsa = compute_vector_similarity(text1, text2, lsa_model)
    print(f"  LSA Similarity: {result_with_lsa['lsa_similarity']:.3f}")
    print("  ✓ Vector similarity working\n")


def test_semantic():
    """Test semantic similarity module."""
    print("Testing semantic similarity...")
    print("  (This may take a moment to download NLTK resources...)")
    
    from semantic_similarity import compute_semantic_similarity
    
    text1 = "The cat is sleeping"
    text2 = "The feline is resting"
    
    result = compute_semantic_similarity(text1, text2)
    print(f"  Path similarity: {result['path_similarity']:.3f}")
    print(f"  LCH similarity: {result['lch_similarity']:.3f}")
    print(f"  Lin similarity: {result['lin_similarity']:.3f}")
    print("  ✓ Semantic similarity working\n")


def test_grading_system():
    """Test main grading system."""
    print("Testing complete grading system...")
    from grading_system import AutomatedGradingSystem
    
    grader = AutomatedGradingSystem(use_lsa=False)  # Disable LSA for speed
    
    model = "Photosynthesis is the process by which plants use sunlight"
    student = "Plants use the sun through photosynthesis"
    
    result = grader.grade_answer(student, model)
    print(f"  Final score: {result['final_score']:.3f}")
    print(f"  Lexical avg: {result['category_averages']['lexical']:.3f}")
    print(f"  Semantic avg: {result['category_averages']['semantic']:.3f}")
    print(f"  Vector avg: {result['category_averages']['vector']:.3f}")
    print("  ✓ Grading system working\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Automated Grading System - Module Tests")
    print("=" * 60)
    print()
    
    try:
        test_lexical()
        test_vector()
        test_semantic()
        test_grading_system()
        
        print("=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

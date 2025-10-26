"""
Example Usage Script for Automated Grading System
Demonstrates various ways to use the system.
"""

import pandas as pd
from grading_system import AutomatedGradingSystem


def example_1_single_answer():
    """Example 1: Grade a single student answer."""
    print("=" * 70)
    print("Example 1: Grade a Single Answer")
    print("=" * 70)
    
    grader = AutomatedGradingSystem(use_lsa=False)
    
    model_answer = """
    Photosynthesis is the process by which plants use sunlight, water, and 
    carbon dioxide to create oxygen and energy in the form of sugar.
    """
    
    student_answer = """
    Photosynthesis is when plants convert light energy into chemical energy.
    They use sunlight to make glucose and release oxygen.
    """
    
    result = grader.grade_answer(student_answer, model_answer)
    
    print(f"\nModel Answer: {model_answer.strip()[:80]}...")
    print(f"Student Answer: {student_answer.strip()[:80]}...")
    print(f"\nPredicted Score: {result['final_score']:.3f}")
    print("\nDetailed Scores:")
    print(f"  Lexical Average: {result['category_averages']['lexical']:.3f}")
    print(f"  Semantic Average: {result['category_averages']['semantic']:.3f}")
    print(f"  Vector Average: {result['category_averages']['vector']:.3f}")
    print()


def example_2_multiple_answers():
    """Example 2: Grade multiple answers and evaluate performance."""
    print("=" * 70)
    print("Example 2: Grade Multiple Answers")
    print("=" * 70)
    
    # Sample dataset with student answers and human scores
    data = {
        'student_answer': [
            "Photosynthesis is the process where plants convert light into energy using water and CO2.",
            "Plants use the sun to make food. This is photosynthesis.",
            "Through photosynthesis, plants create glucose and oxygen using sunlight, water, and carbon dioxide.",
            "Plants absorb sunlight and this helps them grow.",
            "Photosynthesis occurs in chloroplasts where light energy is converted to chemical energy.",
        ],
        'model_answer': [
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
        ],
        'human_score': [0.9, 0.5, 0.95, 0.4, 0.85]
    }
    
    df = pd.DataFrame(data)
    grader = AutomatedGradingSystem(use_lsa=True)
    
    # Grade all answers
    results = grader.grade_multiple_answers(df, 'student_answer', 'model_answer', 'human_score')
    
    # Show results
    print("\nGrading Results:")
    print("-" * 70)
    for idx, row in results.iterrows():
        print(f"\nAnswer {idx + 1}:")
        print(f"  Human Score: {row['human_score']:.2f}")
        print(f"  Predicted Score: {row['predicted_score']:.2f}")
        print(f"  Difference: {abs(row['human_score'] - row['predicted_score']):.2f}")
    
    # Evaluate performance
    metrics = grader.evaluate_performance(results)
    print("\n" + "=" * 70)
    print("Performance Metrics:")
    print("-" * 70)
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.4f}")
    print(f"RMSE: {metrics['root_mean_squared_error']:.4f}")
    print()


def example_3_with_csv():
    """Example 3: Load data from CSV and grade."""
    print("=" * 70)
    print("Example 3: Process CSV Dataset")
    print("=" * 70)
    
    # Check if test.csv exists
    try:
        df = pd.read_csv('test.csv')
        print(f"\nLoaded dataset with {len(df)} essays")
        
        # Define model answer
        model_answer = """
        Photosynthesis is the process by which plants use sunlight, water, and 
        carbon dioxide to create oxygen and energy in the form of sugar. This 
        process occurs in chloroplasts and is essential for life on Earth.
        """
        
        # Add model answer to dataframe
        df['model_answer'] = model_answer
        
        # Normalize scores to 0-1 range
        df['normalized_score'] = (df['domain1_score'] - df['domain1_score'].min()) / \
                                 (df['domain1_score'].max() - df['domain1_score'].min())
        
        # Grade essays
        grader = AutomatedGradingSystem(use_lsa=True)
        results = grader.grade_multiple_answers(
            df, 'essay', 'model_answer', 'normalized_score'
        )
        
        # Show sample results
        print("\nSample Results (first 5):")
        print("-" * 70)
        for idx in range(min(5, len(results))):
            row = results.iloc[idx]
            print(f"\nEssay {idx + 1}:")
            print(f"  Original Score: {df.iloc[idx]['domain1_score']}")
            print(f"  Predicted (normalized): {row['predicted_score']:.3f}")
            print(f"  Human (normalized): {row['human_score']:.3f}")
        
        # Evaluate
        metrics = grader.evaluate_performance(results)
        print("\n" + "=" * 70)
        print("Performance:")
        print(f"  Correlation: {metrics['pearson_correlation']:.4f}")
        print()
        
    except FileNotFoundError:
        print("\ntest.csv not found. Skipping this example.")
        print()


def example_4_custom_weights():
    """Example 4: Use custom weights for similarity categories."""
    print("=" * 70)
    print("Example 4: Custom Similarity Weights")
    print("=" * 70)
    
    # Create grader with custom weights
    grader = AutomatedGradingSystem(use_lsa=False)
    
    # Modify weights to emphasize semantic similarity
    grader.weights = {
        'lexical': 0.2,
        'semantic': 0.6,  # Emphasize semantic
        'vector': 0.2
    }
    
    model_answer = "The cell membrane controls what enters and exits the cell."
    student_answer = "The cellular membrane regulates the passage of substances into and out of the cell."
    
    result = grader.grade_answer(student_answer, model_answer)
    
    print(f"\nModel: {model_answer}")
    print(f"Student: {student_answer}")
    print(f"\nWeights: Lexical=0.2, Semantic=0.6, Vector=0.2")
    print(f"Final Score: {result['final_score']:.3f}")
    print("\nNote: These answers use different words but similar meaning,")
    print("so semantic similarity should be high.")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Automated Grading System - Usage Examples")
    print("=" * 70)
    print()
    
    example_1_single_answer()
    example_2_multiple_answers()
    example_3_with_csv()
    example_4_custom_weights()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nFor more information, see README.md")
    print()


if __name__ == "__main__":
    main()

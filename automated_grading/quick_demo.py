"""
Quick Demo - Automated Grading System
Fast demonstration without semantic similarity (to avoid NLTK delays).
"""

from grading_system import AutomatedGradingSystem
import pandas as pd


def main():
    print("=" * 70)
    print("Automated Grading System - Quick Demo")
    print("=" * 70)
    print("\nThis demo uses lexical and vector similarity (faster).")
    print("For full semantic similarity, use grading_system.py\n")
    
    # Create sample data
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
        ] * 5,
        'human_score': [0.9, 0.5, 0.95, 0.4, 0.85]
    }
    
    df = pd.DataFrame(data)
    
    # Initialize grader (LSA enabled, but faster than semantic)
    grader = AutomatedGradingSystem(use_lsa=True)
    
    # For speed, adjust weights to reduce semantic component
    grader.weights = {
        'lexical': 0.4,
        'semantic': 0.2,
        'vector': 0.4
    }
    
    print("Grading 5 student answers...")
    results = grader.grade_multiple_answers(df, 'student_answer', 'model_answer', 'human_score')
    
    print("\nResults:")
    print("-" * 70)
    for idx, row in results.iterrows():
        print(f"\nAnswer {idx + 1}:")
        print(f"  Text: {row['student_answer'][:60]}...")
        print(f"  Human Score: {row['human_score']:.2f}")
        print(f"  Predicted: {row['predicted_score']:.2f}")
        print(f"  Difference: {abs(row['human_score'] - row['predicted_score']):.2f}")
    
    # Evaluate
    metrics = grader.evaluate_performance(results)
    print("\n" + "=" * 70)
    print("Performance Metrics:")
    print("-" * 70)
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.4f}")
    print(f"RMSE: {metrics['root_mean_squared_error']:.4f}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("For more examples, see examples.py")
    print("For full documentation, see README.md")
    print("=" * 70)


if __name__ == "__main__":
    main()

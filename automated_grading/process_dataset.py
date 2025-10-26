"""
Process Hewlett Foundation Dataset
Script to apply the automated grading system to the Hewlett dataset.
"""

import pandas as pd
import numpy as np
from grading_system import AutomatedGradingSystem


def normalize_score(score, min_score=0, max_score=5):
    """
    Normalize score to 0-1 range.
    
    Args:
        score: Original score
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        float: Normalized score
    """
    return (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0


def denormalize_score(normalized_score, min_score=0, max_score=5):
    """
    Convert normalized score back to original scale.
    
    Args:
        normalized_score: Score in 0-1 range
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        float: Score in original scale
    """
    return normalized_score * (max_score - min_score) + min_score


def process_dataset(csv_path, model_answer, output_path=None):
    """
    Process the Hewlett dataset and apply automated grading.
    
    Args:
        csv_path (str): Path to the CSV file
        model_answer (str): The reference/model answer for the question
        output_path (str, optional): Path to save results
        
    Returns:
        tuple: (results_df, metrics)
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded: {len(df)} essays")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare data
    # Use 'essay' column as student answers
    # domain1_score as the human score
    
    if 'essay' not in df.columns:
        raise ValueError("'essay' column not found in dataset")
    
    if 'domain1_score' not in df.columns:
        raise ValueError("'domain1_score' column not found in dataset")
    
    # Add model answer column
    df['model_answer'] = model_answer
    
    # Normalize human scores to 0-1 for comparison
    min_score = df['domain1_score'].min()
    max_score = df['domain1_score'].max()
    print(f"Score range: {min_score} to {max_score}")
    
    df['normalized_human_score'] = df['domain1_score'].apply(
        lambda x: normalize_score(x, min_score, max_score)
    )
    
    # Initialize grading system
    print("\nInitializing automated grading system...")
    grader = AutomatedGradingSystem(use_lsa=True)
    
    # Grade all essays
    print("Grading essays...")
    results = grader.grade_multiple_answers(
        df,
        student_col='essay',
        model_col='model_answer',
        score_col='normalized_human_score'
    )
    
    # Denormalize predicted scores
    results['predicted_score_original'] = results['predicted_score'].apply(
        lambda x: denormalize_score(x, min_score, max_score)
    )
    
    # Add original human score
    results['human_score_original'] = results['human_score'].apply(
        lambda x: denormalize_score(x, min_score, max_score)
    )
    
    # Evaluate performance
    print("\nEvaluating performance...")
    metrics = grader.evaluate_performance(results, 'human_score', 'predicted_score')
    
    # Calculate additional metrics on original scale
    mae_original = np.mean(np.abs(
        results['human_score_original'] - results['predicted_score_original']
    ))
    rmse_original = np.sqrt(np.mean(
        (results['human_score_original'] - results['predicted_score_original']) ** 2
    ))
    
    metrics['mae_original_scale'] = mae_original
    metrics['rmse_original_scale'] = rmse_original
    
    # Display results
    print("\n" + "=" * 80)
    print("AUTOMATED GRADING RESULTS")
    print("=" * 80)
    
    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
    print(f"Mean Absolute Error (normalized): {metrics['mean_absolute_error']:.4f}")
    print(f"RMSE (normalized): {metrics['root_mean_squared_error']:.4f}")
    print(f"Mean Absolute Error (original scale): {metrics['mae_original_scale']:.4f}")
    print(f"RMSE (original scale): {metrics['rmse_original_scale']:.4f}")
    
    print("\n" + "-" * 80)
    print("Sample Predictions (first 10 essays):")
    print("-" * 80)
    
    for idx in range(min(10, len(results))):
        row = results.iloc[idx]
        print(f"\nEssay {idx + 1}:")
        print(f"  Text: {row['student_answer'][:100]}...")
        print(f"  Human Score: {row['human_score_original']:.1f}")
        print(f"  Predicted Score: {row['predicted_score_original']:.1f}")
        print(f"  Difference: {abs(row['human_score_original'] - row['predicted_score_original']):.2f}")
    
    # Save results if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 80)
    
    return results, metrics


def main():
    """Main function to process the dataset."""
    # Define the model/reference answer for photosynthesis
    model_answer = """
    Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide 
    to create oxygen and energy in the form of sugar. This process occurs in the chloroplasts 
    of plant cells, specifically using chlorophyll to capture light energy. The process consists 
    of two main stages: the light-dependent reactions and the Calvin cycle (light-independent reactions). 
    During the light-dependent reactions, light energy is used to split water molecules, 
    releasing oxygen and generating ATP and NADPH. In the Calvin cycle, these energy carriers 
    are used to fix carbon dioxide into glucose. Photosynthesis is fundamental to life on Earth 
    as it provides oxygen for respiration and forms the base of food chains.
    """
    
    # Process the dataset
    csv_path = 'test.csv'
    output_path = 'grading_results.csv'
    
    try:
        results, metrics = process_dataset(csv_path, model_answer, output_path)
        
        print("\nProcessing completed successfully!")
        print(f"Correlation with human graders: {metrics['pearson_correlation']:.4f}")
        print("\nNote: Typical human-to-human agreement correlation is ~0.85-0.90")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_path}'")
        print("Please ensure the test.csv file is in the current directory.")
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

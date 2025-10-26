"""
Automated Grading System
Main module that combines all similarity measures for automated grading.
"""

import pandas as pd
import numpy as np
from lexical_similarity import compute_lexical_similarity
from semantic_similarity import compute_semantic_similarity
from vector_similarity import compute_vector_similarity, LSAModel


class AutomatedGradingSystem:
    """
    Unsupervised automated grading system using text similarity measures.
    """
    
    def __init__(self, use_lsa=True):
        """
        Initialize the grading system.
        
        Args:
            use_lsa (bool): Whether to use LSA for vector similarity
        """
        self.use_lsa = use_lsa
        self.lsa_model = None
        self.weights = {
            'lexical': 0.3,
            'semantic': 0.4,
            'vector': 0.3
        }
    
    def train_lsa(self, corpus):
        """
        Train LSA model on a corpus of documents.
        
        Args:
            corpus (list): List of text documents
        """
        if self.use_lsa and len(corpus) > 1:
            self.lsa_model = LSAModel(n_components=min(100, len(corpus) - 1))
            self.lsa_model.fit(corpus)
    
    def grade_answer(self, student_answer, model_answer):
        """
        Grade a student answer by comparing it to the model answer.
        
        Args:
            student_answer (str): Student's answer
            model_answer (str): Model (correct) answer
            
        Returns:
            dict: Dictionary containing all similarity scores and final grade
        """
        # Compute all similarities
        lexical_scores = compute_lexical_similarity(student_answer, model_answer)
        semantic_scores = compute_semantic_similarity(student_answer, model_answer)
        vector_scores = compute_vector_similarity(student_answer, model_answer, self.lsa_model)
        
        # Calculate average scores for each category
        lexical_avg = np.mean(list(lexical_scores.values()))
        semantic_avg = np.mean(list(semantic_scores.values()))
        vector_avg = np.mean(list(vector_scores.values()))
        
        # Calculate weighted final score
        final_score = (
            self.weights['lexical'] * lexical_avg +
            self.weights['semantic'] * semantic_avg +
            self.weights['vector'] * vector_avg
        )
        
        # Compile all results
        results = {
            'lexical_scores': lexical_scores,
            'semantic_scores': semantic_scores,
            'vector_scores': vector_scores,
            'category_averages': {
                'lexical': lexical_avg,
                'semantic': semantic_avg,
                'vector': vector_avg
            },
            'final_score': final_score
        }
        
        return results
    
    def grade_multiple_answers(self, df, student_col='student_answer', 
                              model_col='model_answer', score_col=None):
        """
        Grade multiple student answers from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing answers
            student_col (str): Column name for student answers
            model_col (str): Column name for model answers
            score_col (str, optional): Column name for human scores (for correlation)
            
        Returns:
            pd.DataFrame: DataFrame with similarity scores and grades
        """
        results_list = []
        
        # Train LSA on all texts if needed
        if self.use_lsa:
            all_texts = []
            if student_col in df.columns:
                all_texts.extend(df[student_col].dropna().tolist())
            if model_col in df.columns:
                all_texts.extend(df[model_col].dropna().tolist())
            
            if all_texts:
                self.train_lsa(all_texts)
        
        # Grade each answer
        for idx, row in df.iterrows():
            student_ans = row.get(student_col, '')
            model_ans = row.get(model_col, '')
            
            if pd.isna(student_ans) or pd.isna(model_ans):
                continue
            
            result = self.grade_answer(str(student_ans), str(model_ans))
            
            row_result = {
                'index': idx,
                'student_answer': student_ans,
                'model_answer': model_ans,
                'predicted_score': result['final_score']
            }
            
            # Add detailed scores
            for key, val in result['lexical_scores'].items():
                row_result[f'lexical_{key}'] = val
            for key, val in result['semantic_scores'].items():
                row_result[f'semantic_{key}'] = val
            for key, val in result['vector_scores'].items():
                row_result[f'vector_{key}'] = val
            
            # Add human score if available
            if score_col and score_col in row:
                row_result['human_score'] = row[score_col]
            
            results_list.append(row_result)
        
        return pd.DataFrame(results_list)
    
    def evaluate_performance(self, results_df, human_score_col='human_score', 
                            predicted_score_col='predicted_score'):
        """
        Evaluate system performance against human scores.
        
        Args:
            results_df (pd.DataFrame): DataFrame with results
            human_score_col (str): Column name for human scores
            predicted_score_col (str): Column name for predicted scores
            
        Returns:
            dict: Performance metrics
        """
        if human_score_col not in results_df.columns:
            return {'error': 'Human scores not available'}
        
        human_scores = results_df[human_score_col].values
        predicted_scores = results_df[predicted_score_col].values
        
        # Calculate correlation
        correlation = np.corrcoef(human_scores, predicted_scores)[0, 1]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(human_scores - predicted_scores))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((human_scores - predicted_scores) ** 2))
        
        return {
            'pearson_correlation': correlation,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse
        }


def main():
    """Main function demonstrating the grading system."""
    # Example usage
    print("=" * 60)
    print("Automated Grading System - Text Similarity Based")
    print("=" * 60)
    print()
    
    # Create sample data
    sample_data = {
        'student_answer': [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Plants use sunlight to make food through photosynthesis.",
            "Photosynthesis happens when plants absorb sunlight.",
            "It is how plants grow using the sun."
        ],
        'model_answer': [
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        ],
        'human_score': [0.9, 0.7, 0.6, 0.4]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize grading system
    grader = AutomatedGradingSystem(use_lsa=True)
    
    # Grade all answers
    print("Grading student answers...")
    results = grader.grade_multiple_answers(df, 'student_answer', 'model_answer', 'human_score')
    
    # Display results
    print("\nResults:")
    print("-" * 60)
    for idx, row in results.iterrows():
        print(f"\nStudent Answer {idx + 1}:")
        print(f"  Student: {row['student_answer'][:60]}...")
        print(f"  Predicted Score: {row['predicted_score']:.3f}")
        print(f"  Human Score: {row['human_score']:.3f}")
        print(f"  Difference: {abs(row['predicted_score'] - row['human_score']):.3f}")
    
    # Evaluate performance
    print("\n" + "=" * 60)
    print("Performance Evaluation:")
    print("=" * 60)
    metrics = grader.evaluate_performance(results)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Grading system demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

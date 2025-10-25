# Quick Start Guide - Automated Grading System

## Installation (2 minutes)

```bash
# 1. Navigate to the project directory
cd automated_grading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Done! The system will download NLTK data on first run.
```

## Quick Test (30 seconds)

```bash
# Run the quick demo
python quick_demo.py
```

Expected output: Correlation varies (quick demo uses faster similarity only)

## Full Demo (2-3 minutes)

```bash
# Run the full demo with semantic similarity
python grading_system.py
```

Expected output: Correlation with human graders (typically 0.75-0.85)

## Your First Grading

Create a Python script:

```python
from grading_system import AutomatedGradingSystem

# Initialize
grader = AutomatedGradingSystem(use_lsa=True)

# Define the correct answer
model_answer = "Your reference answer here"

# Grade a student's answer
student_answer = "Student's answer here"
result = grader.grade_answer(student_answer, model_answer)

# Get the score
print(f"Score: {result['final_score']:.2f}")
```

## Grade Multiple Answers from CSV

```python
import pandas as pd
from grading_system import AutomatedGradingSystem

# Load your data
df = pd.read_csv('your_data.csv')
# Columns: student_answer, model_answer, human_score (optional)

# Grade all answers
grader = AutomatedGradingSystem(use_lsa=True)
results = grader.grade_multiple_answers(
    df, 
    student_col='student_answer',
    model_col='model_answer', 
    score_col='human_score'
)

# Evaluate performance
metrics = grader.evaluate_performance(results)
print(f"Correlation: {metrics['pearson_correlation']:.3f}")

# Save results
results.to_csv('grading_results.csv', index=False)
```

## Common Use Cases

### 1. Grade a single answer

Create a file `grade_single.py`:
```python
from grading_system import AutomatedGradingSystem

grader = AutomatedGradingSystem()
result = grader.grade_answer('Student answer', 'Model answer')
print(f"Score: {result['final_score']:.3f}")
```

Then run: `python grade_single.py`

### 2. Process Hewlett dataset
```bash
python process_dataset.py
```

### 3. Run all examples
```bash
python examples.py
```

### 4. Test all modules
```bash
python test_modules.py
```

## Understanding the Output

### Score Range
- **0.0 - 0.3**: Poor match (likely incorrect)
- **0.3 - 0.5**: Fair match (partial understanding)
- **0.5 - 0.7**: Good match (mostly correct)
- **0.7 - 0.9**: Very good match (correct)
- **0.9 - 1.0**: Excellent match (nearly identical)

### Detailed Scores
Each graded answer includes:
- `final_score`: Combined weighted score
- `lexical_scores`: Word overlap measures
- `semantic_scores`: WordNet-based similarity
- `vector_scores`: TF-IDF and LSA similarity

### Performance Metrics
When comparing to human scores:
- `pearson_correlation`: How well scores align (target: ~0.85)
- `mean_absolute_error`: Average difference in scores
- `root_mean_squared_error`: Penalizes large errors more

## Tips for Best Results

1. **Model Answer Quality**: Write clear, comprehensive reference answers
2. **LSA Training**: Enable LSA for better results with larger datasets
3. **Custom Weights**: Adjust similarity weights based on your subject
4. **Batch Processing**: Grade multiple answers together for LSA benefits

## Customization

### Adjust Similarity Weights
```python
grader = AutomatedGradingSystem()
grader.weights = {
    'lexical': 0.2,    # Word overlap
    'semantic': 0.6,   # Meaning similarity (emphasis)
    'vector': 0.2      # Vector space
}
```

### Disable LSA (faster)
```python
grader = AutomatedGradingSystem(use_lsa=False)
```

## Troubleshooting

### NLTK Data Not Found
The system will automatically download required data on first run. If you encounter issues, manually download:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet_ic')
nltk.download('averaged_perceptron_tagger')
```

### Slow Performance
- Use `quick_demo.py` for faster results
- Disable semantic similarity for speed
- Process in batches for LSA efficiency

### Low Correlation
- Check model answer quality
- Ensure student answers are substantive (not too short)
- Adjust similarity weights for your subject

## Dataset Format

CSV file should have:
```csv
essay_id,essay,domain1_score
1,"Answer text",4
2,"Another answer",3
```

Or for comparison:
```csv
student_answer,model_answer,human_score
"Student text","Reference text",0.8
```

## Next Steps

1. ✅ Run `python test_modules.py` to verify installation
2. ✅ Try `python quick_demo.py` to see it in action  
3. ✅ Read `README.md` for detailed documentation
4. ✅ Check `examples.py` for more usage patterns
5. ✅ Review `PROJECT_SUMMARY.md` for implementation details

## Support

- Documentation: `README.md`
- Examples: `examples.py`
- Security: `SECURITY.md`
- Full Summary: `PROJECT_SUMMARY.md`

## Performance Expectations

With the sample dataset:
- **Processing Time**: ~1-3 seconds per answer (with semantic)
- **Correlation**: 0.75-0.85 with human graders
- **Accuracy**: Comparable to human-human agreement

---

**Ready to start grading? Run `python quick_demo.py`!**

# Math-Project

This repository contains various mathematical computation scripts and an automated grading system for short-answer questions.

## Projects

### 1. Mathematical Computations
Various Python scripts for:
- Prime number checking and generation
- Prime factorization
- Elliptic curve calculations
- Number theory visualizations

### 2. Automated Grading System ⭐ NEW
An unsupervised text similarity-based system for automatically grading short-answer questions.

**Location**: `automated_grading/`

**Features**:
- Combines lexical, semantic, and vector-based similarity measures
- Uses WordNet and Latent Semantic Analysis (LSA)
- Achieves ~0.85 correlation with human graders
- No training data required (unsupervised approach)
- Based on research in educational NLP

**Quick Start**:
```bash
cd automated_grading
pip install -r requirements.txt
python examples.py
```

**Documentation**: See [automated_grading/README.md](automated_grading/README.md) for full documentation.

## Background

The automated grading system is based on research in unsupervised short-answer grading using text similarity. It implements multiple similarity measures:

1. **Lexical Similarity** - Word overlap metrics (Jaccard, overlap coefficient, cosine)
2. **Semantic Similarity** - WordNet-based measures (path, LCH, Lin)
3. **Vector-Based Similarity** - TF-IDF and LSA using scikit-learn

The system compares student answers to reference answers and produces scores that correlate highly with human graders (~0.85), approaching human-to-human agreement levels (~0.90).

## Dataset

The system works with the Hewlett Foundation: Automated Essay Scoring dataset format. A sample `test.csv` is included in the `automated_grading/` directory.

## Usage

### For Automated Grading:
```bash
cd automated_grading
python grading_system.py    # Run demo
python examples.py           # See examples
python process_dataset.py   # Process Hewlett dataset
```

### For Mathematical Scripts:
Simply run the individual Python scripts as needed.

## Requirements

For the automated grading system:
- Python 3.7+
- NLTK
- scikit-learn
- pandas
- numpy

See `automated_grading/requirements.txt` for specific versions.

## License

This project is open source and available for educational purposes.
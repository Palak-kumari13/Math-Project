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

## Contributing

We welcome contributions! Whether you want to add notes, improve documentation, or contribute code:

- **Adding Notes**: See [NOTES.md](NOTES.md) to add your observations and notes about the project
- **Contributing Code/Docs**: See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute

### Quick Guide to Commit Notes

1. Fork this repository
2. Create a new branch: `git checkout -b add-my-notes`
3. Add your notes to `NOTES.md` or create a new documentation file
4. Commit: `git commit -m "Add notes about [topic]"`
5. Push: `git push origin add-my-notes`
6. Open a Pull Request

For detailed instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is open source and available for educational purposes.
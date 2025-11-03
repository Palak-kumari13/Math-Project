# Contributing to Math-Project

Thank you for your interest in contributing to the Math-Project! This guide will help you understand how to contribute notes, code, and other improvements to this repository.

## Table of Contents

1. [How to Commit Notes](#how-to-commit-notes)
2. [How to Contribute Code](#how-to-contribute-code)
3. [Pull Request Process](#pull-request-process)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Testing Guidelines](#testing-guidelines)

## How to Commit Notes

If you want to add notes, observations, or documentation to this project, follow these steps:

### Step 1: Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Math-Project.git
cd Math-Project
```

### Step 2: Create a New Branch

```bash
# Create a new branch for your notes
git checkout -b add-my-notes
```

### Step 3: Add Your Notes

You can add notes in several ways:

#### Option A: Add to NOTES.md (Recommended)
Edit the `NOTES.md` file to add your observations:

```bash
# Open NOTES.md in your favorite editor
nano NOTES.md  # or vim, code, etc.
```

Add your notes under the appropriate section, or create a new section if needed.

#### Option B: Create a New Documentation File
If you have extensive notes on a specific topic:

```bash
# Create a new markdown file in a docs/ directory
mkdir -p docs
nano docs/my-topic-notes.md
```

#### Option C: Add Comments to Code
For code-specific notes, add clear comments directly in the relevant Python files.

### Step 4: Commit Your Changes

```bash
# Stage your changes
git add NOTES.md  # or the specific files you modified

# Commit with a clear message
git commit -m "Add notes about [topic]"

# Examples of good commit messages:
# git commit -m "Add notes about prime factorization algorithms"
# git commit -m "Add documentation for automated grading system setup"
# git commit -m "Add observations about elliptic curve calculations"
```

### Step 5: Push Your Changes

```bash
# Push your branch to your fork
git push origin add-my-notes
```

### Step 6: Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Add a clear title and description of your notes
4. Submit the pull request

## How to Contribute Code

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Math-Project.git
cd Math-Project

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r automated_grading/requirements.txt
```

### Making Code Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes thoroughly
4. Commit with clear messages
5. Push and create a pull request

## Pull Request Process

1. **Update Documentation**: If your changes affect usage, update README.md or relevant documentation
2. **Test Your Changes**: Ensure all existing functionality still works
3. **Clear Description**: Explain what your PR does and why
4. **Reference Issues**: If applicable, reference any related issues

### PR Title Format

- For notes: `docs: Add notes about [topic]`
- For bug fixes: `fix: [brief description]`
- For new features: `feat: [brief description]`
- For improvements: `improve: [brief description]`

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add comments for complex logic

Example:
```python
def calculate_prime_factors(n):
    """
    Calculate all prime factors of a given number.
    
    Args:
        n (int): The number to factorize
        
    Returns:
        list: A list of prime factors
    """
    factors = []
    # Implementation here
    return factors
```

### Documentation

- Use Markdown for documentation files
- Keep lines under 100 characters when possible
- Use clear section headers
- Include code examples where helpful

## Testing Guidelines

### Running Existing Tests

```bash
cd automated_grading
python test_modules.py
python examples.py
```

### Adding New Tests

If you add new functionality, include tests:

1. Create test functions in `test_modules.py` or a new test file
2. Follow existing test patterns
3. Test both success and failure cases
4. Include edge cases

## Questions or Issues?

If you have questions about contributing:

1. Check existing issues on GitHub
2. Review the README.md for project documentation
3. Open a new issue with your question

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Thank You!

Your contributions make this project better. Whether you're adding notes, fixing bugs, or implementing new features, we appreciate your effort!

---

**Need Help?** Feel free to open an issue or reach out to the maintainers.

# Security Notice

## Dependencies

This project uses several Python packages. We have taken steps to ensure all dependencies are up-to-date and free from known vulnerabilities.

## Security Fixes Applied

### NLTK Version
- **Issue**: NLTK < 3.9 has an unsafe deserialization vulnerability
- **Fix**: Updated to NLTK >= 3.9 (patched version)
- **Impact**: Low (our usage doesn't involve untrusted serialized data)

## Dependency Versions

Current requirements:
```
nltk>=3.9           # Security fix applied
scikit-learn==1.3.2 # No known vulnerabilities
numpy==1.26.2       # No known vulnerabilities
pandas==2.1.3       # No known vulnerabilities
scipy==1.11.4       # No known vulnerabilities
```

## Best Practices

1. **Regular Updates**: Keep dependencies updated to latest stable versions
2. **Minimal Dependencies**: Only essential packages are included
3. **No Remote Code Execution**: System doesn't execute arbitrary code
4. **Input Validation**: All text inputs are sanitized before processing
5. **No External API Calls**: System works entirely offline after NLTK data download

## Usage Security

This automated grading system:
- ✅ Does NOT execute student code
- ✅ Does NOT make external network requests (except for initial NLTK data)
- ✅ Does NOT store or transmit sensitive data
- ✅ Uses only text similarity computations
- ✅ Sanitizes all text inputs

## Reporting Security Issues

If you discover a security vulnerability, please:
1. Do NOT open a public issue
2. Contact the repository maintainers directly
3. Provide details about the vulnerability
4. Allow time for a fix before public disclosure

## Security Audit Summary

Last audit date: October 25, 2025

Findings:
- NLTK vulnerability identified and fixed
- All other dependencies verified against GitHub Advisory Database
- No critical or high severity vulnerabilities found
- System architecture reviewed for security best practices

Status: ✅ **SECURE**

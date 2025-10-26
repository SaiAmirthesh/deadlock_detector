# ML-Based Deadlock Detection System

A machine learning-based deadlock detection system that compares performance across different algorithms with comprehensive timing analysis, specifically designed for banking transaction data.

Live demo: https://deaddetector.streamlit.app/

## 🚀 Features

### ML Algorithms Comparison
- **Hybrid Model** - Intelligent method selection with optimized performance
- **Random Forest** - Ensemble tree-based classifier
- **XGBoost** - Gradient boosting optimized for performance  

### How to Use:

- Upload your transaction data (Excel/CSV)
- Select detection methods to compare    
- Run the analysis
- View deadlock detection results

### Expected Data Format:

-Your file should contain transaction details like:
    Account information
    Transaction amounts
    Transaction types
    Timestamps

### Confidence Scoring
ML Confidence Factors:
    
    - Data size: More data = higher confidence
    
    - Feature richness: More features available = higher confidence
    
    - Data quality: Less missing data = higher confidence

Traditional Confidence Factors:
    
    -Small datasets: Traditional methods work better
    
    -High complexity: Traditional methods more reliable for complex cases

### Key Advantages

    Adaptive: Chooses best method based on data characteristics

    Robust: Multiple fallback layers ensure reliability

    Transparent: Provides clear reasoning for method selection

    Ensemble: Combines multiple ML models for better accuracy

    Safe: Conservative fallbacks prevent catastrophic failures

### Live demo:


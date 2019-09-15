gle Competition - Digit Recognizer
-----------------------------------------------------

This is a learning and experimentation project for doing data science analysis for the Kaggle competition:

Digit Recognizer - Learn computer vision fundamentals with the famous MNIST data
- [https://www.kaggle.com/c/house-prices-advanced-regression-techniques]


## Installation
```
./requirements.sh           # Install/Update VirtualEnv
source venv/bin/activate    # Source VirtualEnv
jupyter lab                 # Explore Jupyter Notebooks                  
./main.py                   # Execute Data Pipeline

kaggle competitions submit -c house-prices-advanced-regression-techniques -f data/submissions/LeastSquaresCorr.csv -m "sklearn.linear_model.LinearRegression() on fields: .corr() > 0.5"
```
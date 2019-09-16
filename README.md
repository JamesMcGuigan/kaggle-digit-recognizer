Kaggle Competition - Digit Recognizer
-----------------------------------------------------

Learn computer vision fundamentals with the famous MNIST data
- https://www.kaggle.com/c/digit-recognizer


## Installation
```
# ./requirements.sh           # Install/Update VirtualEnv
# source venv/bin/activate    # Source VirtualEnv
# jupyter lab                 # Explore Jupyter Notebooks                  
# ./main.py                   # Execute Data Pipeline

kaggle competitions download -c digit-recognizer -p ./data/
node --experimental-modules src/gcloud-ocr/csv2png.js

kaggle competitions submit -c digit-recognizer -f submissions/submission.csv -m "message"
```
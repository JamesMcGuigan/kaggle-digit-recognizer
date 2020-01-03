# Kaggle Competition - Digit Recognizer

Learn computer vision fundamentals with the famous MNIST data
- https://www.kaggle.com/c/digit-recognizer

NOTE: training accuracy inside notebooks is claiming to be 98%, 
but submission to Kaggle is returning scores that are actually worse than random
this would seem to suggest some form of image/id mismatch

# Submissions
- Score: 0.99657 | Rank: ??? /2500 | ./submissions/fastai-resnet18-u100.csv - fastai: resnet18 + fit_one_cycle(50, 5e-2)
- Score: 0.71128 | Rank: 2194/2269 | ./submissions/keras.csv - first attempt
- Score: 0.09671 | Rank: 2487/2500 | ./submissions/random.csv



# Preprocessing
# csv2png: Image Generation
```
yarn
yarn download
==
kaggle competitions download -c digit-recognizer -p ./data/
unzip data/digit-recognizer.zip -d data
node --experimental-modules preprocessing/csv2png.js
# kaggle competitions submit -c digit-recognizer -f submissions/submission.csv -m "message"
```

# Methods

## FastAI Jupyter Notebooks
```
pip3 install -r requirements.in
PYTHONPATH='.' nice time python3 method_keras/main.py 
``` 
This method currently produces the best state-of-the-art results, with a top score of 0.99657 


## Keras
```
pip3 install -r requirements.in
 
```

Keras is a lower level library than fastai. 

Initial implementation works as a proof of concept, but the first attempt only produces a score of 0.71128



## Broken: Google Cloud OCR

This was intended as a cheat method, map the csv data back into pngs, then use the Google Vision API to conduct OCR

Doesn't seem to work!

Problems:
- Cost: $1.50 per 1000 requests * 28,000 test images = $42 cost
- Google OCR doesn't seem to like white-on-black single char text images
- Inverting the images (black on white) doesn't improve Google OCR  
- API Explorer: 
  https://cloud.google.com/vision/docs/quickstart?apix_params=%7B%22resource%22%3A%7B%22requests%22%3A%5B%7B%22features%22%3A%5B%7B%22type%22%3A%22DOCUMENT_TEXT_DETECTION%22%7D%5D%2C%22image%22%3A%7B%22source%22%3A%7B%22imageUri%22%3A%22gs%3A%2F%2Fkaggle-digit-recognizer%2Fdata-images%2Ftest%2F1.png%22%7D%7D%7D%5D%7D%7D
  - features.type = DOCUMENT_TEXT_DETECTION
  - image.source.imageUri = gs://kaggle-digit-recognizer/data-images/test/1.png


```
gsutil -m cp -r data/images/ gs://kaggle-digit-recognizer/
```


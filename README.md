# Kaggle Competition - MNIST Digit Recognizer

Learn computer vision fundamentals with the famous MNIST data
- https://www.kaggle.com/c/digit-recognizer

Installation Instructions: [README_CUDA.md](README_CUDA.md)

---

# Submissions
- Score: 0.99657 | Rank: ??? /2500 | ./submissions/fastai-resnet18-u100.csv - fastai: resnet18 + fit_one_cycle(50, 5e-2)
- Score: 0.71128 | Rank: 2194/2269 | ./submissions/keras.csv - first attempt
- Score: 0.09671 | Rank: 2487/2500 | ./submissions/random.csv

---

# Preprocessing
## csv2png: Image Generation
```
kaggle competitions download -c digit-recognizer -p ./data/
unzip data/digit-recognizer.zip -d data
node --experimental-modules src/utils/csv2png.js
# kaggle competitions submit -c digit-recognizer -f submissions/submission.csv -m "message"
```

Converts the CSV data into a filesystem directory tree of png images for better 
visibility and debugging, as well as for compatibility purposes with fastai ImageDataBunch 

NOTE: 
- This is a slower (IO bound) method compared to accessing raw numeric CSV data. 
- Dropbox crashes when trying to sync 2,016,000 individual files

---


# Methods

## Random Guess Method
```
node --experimental-modules  src/random/random.js 
wrote: ./submissions/random.csv
Accuracy = 2846/28000 = 10.16%
```

The random guess method provides a statistical noise baseline, which as expected averages around 10% accuracy


## FastAI Jupyter Notebooks
```
node ./preprocessing/csv2png.js 
jupyter lab  # 1_fastai-transfer-learning.ipynb
``` 
This method utilizes CNN resnet18 with transfer learning and currently produces the best state-of-the-art results, with a top score of 0.99657 
 

## Keras
Keras is a lower level library than fastai. 

### Keras Example Code - MNIST CNN 
```
CUDA_VISIBLE_DEVICES=""   # run with CPU instead of GPU
PYTHONPATH='.'            # needed for running local code 
time -p python3 src/keras/examples/keras_example_mnist_cnn.py 
Test loss: 0.6942943648338318
Test accuracy: 0.8384
```

Initial benchmark implementation works as a proof of concept. 
[Documentation code](https://keras.io/examples/mnist_cnn/) claims 99.25% test accuracy after 12 epochs, 
but running the code locally only produces a score of 83.84%

Timings:
- 2011 Macbook Pro CPU = 89s/epoc = 1.5ms/sample = 1070s
- 2019 Razer Blade CPU = 36s/epoc = 605us/sample =  443s ( 2.4x improvement over OSX)
- GeForce GTX 1060 GPU =  5s/epoc =  85us/sample =   66s ( 6.7x improvement over CPU)

### Tensorflow Keras Example Syntax
```
python3 src/examples/tensorflow/main.py
```
Working examples of Keras syntax: SequentialCNN, FunctionalCNN, ClassCNN, ClassNN 


### Convergence Search
- see extended comments in: [src/keras/experiments/convergence_search.py](src/keras/experiments/convergence_search.py)

Best Discovered Hyperparameter Combinations (with simple SequentialCNN) 
```python3
"optimizer": hp.Discrete([
    ### learning_rate vs optimizer + scheduler=constant | quickly converges with low learning_rate=0.001
    "Adam",      # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + constant/plateau2/linear_decay
    "Adamax",    # LR<=0.1
    "Nadam",     # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + plateau2 / CyclicLR / linear_decay || LR=0.001 + constant
    "RMSprop",   # LR=0.001 + constant || LR=0.01 + CyclicLR/plateau2/constant/linear_decay || LR=0.1 + CyclicLR (else breaks)
    
    ### learning_rate vs optimizer + scheduler=constant | needs high starting learning_rate=0.1 to quickly converge - may benefit from scheduler
    "Adadelta",  # Best with LR=1   + plateau2 (quick)
    "Adagrad",   # Best with LR=0.1 + triangular (slow/best) or plateau2 (quick)
    "SGD",       # Best with LR=1   + triangular2
    
    ### learning_rate vs optimizer + scheduler=constant | needs learning_rate=0.1 | random until 16 epocs, then quickly converges
    "Ftrl",      # Only works with: LR=0.1 + plateau2/constant OR LR=1 + CyclicLR_triangular
]),
"learning_rate": hp.Discrete([
    1.0,           # Works with: Adadelta + SGD/triangular2 + Adagrad/CyclicLR + Ftrl/triangular (breaks everything else)
    0.1,           # Adamax + Adam/Nadam/RMSprop with CyclicLR || Adagrad + triangular/plateau2
    0.01,          # Adamax + Adam/Nadam/RMSprop with CyclicLR/plateau2/constant/linear_decay
    # 0.001,       # ALL + constant
]),
"min_lr": hp.Discrete([
    0.001,    # 1e-03 (0.001)   - fastest, least overfitting and most accidental high-scores with enough random attempts
    0.0001,
    0.00001,  # 1e-05 (0.00001) - preferred by SGD
    0.000001,
]),
```

Shortlist of Optimised Schedulers (with simple SequentialCNN) 
```
"optimized_scheduler": {
    "Adagrad_triangular": { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "CyclicLR_triangular"  },
    "Adagrad_plateau":    { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "plateau2"      },
    "Adam_triangular2":   { "learning_rate": 0.01,   "optimizer": "Adam",     "scheduler": "CyclicLR_triangular2" },
    "Nadam_plateau":      { "learning_rate": 0.01,   "optimizer": "Nadam",    "scheduler": "plateau_sqrt"  },
    "Adadelta_plateau":   { "learning_rate": 1.0,    "optimizer": "Adadelta", "scheduler": "plateau10"     },
    "SGD_triangular2":    { "learning_rate": 1.0,    "optimizer": "SGD",      "scheduler": "CyclicLR_triangular2" },
    "RMSprop_constant":   { "learning_rate": 0.001,  "optimizer": "RMSprop",  "scheduler": "constant"      },
}
```
# Failed Attempts

## Google Cloud OCR - Broken

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
node ./preprocessing/csv2png.js 
gsutil -m cp -r data/images/ gs://kaggle-digit-recognizer/
```


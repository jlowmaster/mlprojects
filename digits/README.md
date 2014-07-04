This is my first go at savings some projects I've been working on
for re-use and to document what I've been working on. The data set
I'm working with is the classic MNIST handwritten Before putting
this together I did some experimenting with a number of different 
learning algorithms - logistic regression, decision tree, 
random forest, Gaussian Naive Bayes, Support Vector Machine with
rbf, and a linear discriminant analysis.  Among them, 
Random Forest was second best, but KNN routinely outperformed
in all non-tuned scenarios.  

The training set I'm working with consists of 42,000 observations, 
each a handwritten digit, 0-9.  They've been encoded as vectors of pixels,
28x28, or a feature space of 784.  As many of the algorithm choices I've
been working on are n^2, training a classifier on the entire 42k training
set was prohibitive on my macbook air and so I've downsampled.  

The script, digit_main.py, loads the train.csv file into memory as
a numpy ndarray objects, randomly splits the array into a training
and validation set, splits each into the features and the target labels.
Then we do use grid search to walk through the the parameter space for
2 different KNN parameters.  It finds the best model configurations for 
precision and recall and prints out the confusion matrix and 
score reports.

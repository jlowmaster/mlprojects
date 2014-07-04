This project is a letter recognition classification task, taken from
https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/.
The full training set has 20,000 observations but unlike the MNIST digit set
there are only 16 features.  I tried KNN and Random Forest with a simple grid 
search for hyper-parameter optimization and while I got good results for both,
unlike with the digit case, Random Forest was consistently better than KNN
for multiple test runs.  

While it might have something to do with the number of different classes, I 
suspect that it has to do with the smaller feature space.  I think it's probably
something to be mindful of in future.  A possible future experiment might be to
try a dimensionality reduction on the digits set and re-run.

# Emotion Clasification by using raw EEG Signals
This repository contains the source code for emotion classification from `SEED-IV` dataset and `raw EEG signals.`

## Introduction: ##
The project aims to classify emotions into four main groups including `neutral`, `sad`, `fear`, and `happy` based on raw electroencephalogram (EEG) signals. 

- A note: an electroencephalogram (EEG) is a machine  that detects electrical activity in a human brain using small metal discs (electrodes) attached to the scalp. The brain cells communicate via electrical impulses and are active all the time, even when we are asleep. This activity shows up as wavy lines on an EEG recording.

## Dataset Summary ##
The provided data set (downloaded from Google Drive) contains 3 folder, in which folder 1 contain 10 .mat (Matlab) files. The overall description of the experiment is described in the image below:

![Experiment Overview](images/eeg_trial.png "Experiment Overview")
### Raw EEG Signals:
- One `.mat` file represents one experiment for a person
- One individual experiment contains 24 trials correspondings to from `ha_eeg1` to `ha_eeg24 `respectively.
- For one trial, EEG signals from `62 channels` (e.g, FP1, FP2...) are collected.

### Ground truth
The labels of the three sessions for the same subjects are as follows: 
- session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]; 
- session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]; 
- session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]; 

The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.

Since only session1 is available so this will be in used to create the labels.

## Part 1: EEG Understanding ##
### Data Exploration
- Data is downsampled to 200Hz.
- The EEG signals of the same channel (e.g., FP1) for different trials are different as shown in the image below.
- The length of EEG signals depend on the duration of each video
![EEG Signals for a channel](images/eda_fp1_trials.png "EEG Signals for a channel")

### Preprocessing and Feature Extraction:

- First, the eeg signals is divided into 5 frequency sub bands by using `Discrete Wavelet Transform (DWT)`.
- The DWT uses a specific type of wavelet filter bank, called a quadrature mirror filter bank (QMF). The QMF consists of two filters: a low-pass filter and a high-pass filter. The low-pass filter is used to extract the low-frequency components of the signal, while the high-pass filter is used to extract the high-frequency components of the signal. The output of the QMF is a set of coefficients, which represent the different frequency components of the signal.
### FIXME
- We further take the approximation coefficient and pass it through the filter. We do this until the desired frequency ranges are not achieved. Since the filters are successively applied, they are known as filter banks.
### END OF FIXME

For each channel, the following steps are repeated:

- Extract the band power for each sub-band.
- There are 5 sub-bands, so 5 features are extracted for each channel.
- The feature extraction is complete, and each EEG pre-processed signal has 310 features (62 channels x 5 features/channel).

#### Bandpower Calculation
Band power for each sub-band are calculated based on following formular.
```
def calculate_band_power(coeff_d, band_limits):
    # Calculate the power spectrum of the coefficients.
    psd = np.abs(coeff_d)**2

    # Calculate the band power by integrating the power spectrum within the band.
    band_power = np.trapz(psd, dx=(band_limits[1] - band_limits[0]))

    return band_power
```

### Feature Reduction:
In the feature reduction phase, we use Principal Component Analysis (PCA) from scikit-learn. PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

Steps to perform Principal Components Analysis:
1. Mean normalisation of features.
2. Calculating Covariance Matrix.
3. Calculate EigenVectors.
4. Get the reduced features or principal components.

After this process, we have a new set of features called principal components (PCs). The PCs are uncorrelated and ordered by decreasing variance.

## Part 2: Multi Classes Classification: 
The problem would be famed to a multiclasses classification. Since most of data in forms of tabular data, so machine learning methods are selected to make it simple and more efficient.

The PCs from the previous step will be fed into 
multiple `sklearn classifiers` to see the results.

The 3 classifiers are selected to experiments including SVM, RandomForestClassifier and GradientBoostedClassifier.

### FIXME
image here
### END OF FIXME

### Findings
#### Experiment Results

- The SVM model has the highest F1 score for class 0 (0.60), followed by the Random Forest model (0.55) and the Gradient Boosting model (0.52).
- The SVM model has the highest precision for class 2 (0.62), followed by the Random Forest model (0.60) and the Gradient Boosting model (0.48).
- The SVM model has the highest recall for class 1 (0.75), followed by the Random Forest model (0.60) and the Gradient Boosting model (0.47).
- The Gradient Boosting model has the slowest training time, followed by the SVM model and the Random Forest model.

Overall, the SVM model seems to be the best performing model, followed by the Random Forest model and the Gradient Boosting model. Dataset is balanced so models are not biased towards any particular class. However, the F1 score is not really high. 

#### Improvement Points

Based on the experimental results, here are some points that could help improve the model:

- Even after reducing the number of features to 100 principal components (PCs), the number of features is still high for machine learning models, as the dataset is not large. Collecting more data would help improve this.
- Only band power is currently calculated to convert into features to feed into machine learning models. We can improve the model by finding more relevant features, such as band power.
- I have already tried using grid search to find the best parameters for machine learning models, but I did not have enough time to experiment with many parameters. Fine-tuning these hyperparameters would help improve the model.
- Finally, trying some basic deep learning models would also be helpful.


### References
1. [Seed Dataset](http://bcmi.sjtu.edu.cn/~seed/).
2. [Wavelet transform](http://users.rowan.edu/~polikar/WTtutorial.html).
3. [Principal Components Analysis](https://www.coursera.org/learn/machine-learning).
3. [Support Vector Machines](https://www.coursera.org/learn/machine-learning).


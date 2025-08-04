## Hybrid-VAD-Model
Voice Activity Detection (VAD) in stationary and non-stationary noise environments ,implemented in MATLAB using GFCC features and a hybrid machine learning model.
This MATLAB project implements a novel hybrid VAD model that combines three machine learning classifiers and uses gammatone frequency cepstral coefficients (GFCC) features to distinguish between voiced and unvoiced frames in challenging noisy environments.
The hybrid model leverages linear classification(linear-SVM), nonlinear separation(pseudo-QDA), and temporal modelling (bi-LSTM) to improve detection accuracy and robustness in both stationary and non-stationary noise conditions.

## Contents
[Key features] (## Key features)
[Databases ] (## Databases Used)
[Project structure ] (## Project structure)
[Code ] (## Code)

## Key features
Process '.wav' audio files with a sampling frequency of 16 Khz.
Apply audio pre-processing to enhance the signal-to-noise ratio and  the quality of the audio signal,including:
Pre-emphasis filters to boost high frequencies;
Framing with a Hamming window to reduce spectral leakage;
Extract GFCC acoustic features.
Use a hybrid "leaner-SVM+ pseudo-QDA +modified Bi-LSTM) model for VAD.
Perform frame-level binary classification (voiced/unvoiced)
Visualise of speech detection over time.

## Databases Used
This project uses both audio databases (TIMIT and  LibriSpeech) and (Aurora and NOISEX92) noise datasets ,which are combined to generate noisy audio data for training and evaluation.
# Speech databases
Tha main audio databases include:
TIMIT and  LibriSpeech  speech databases which are used in speech processing systems,including
speech recongnition,speech enhancement,and voce activity detection.
- TIMIT Acoustic-Phonetic Continuous Corpus 
The link is: 
(https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) 
- LibriSpeech 
LibriSpeech is a corpus of approximately 1000 hours of read English speech derived from audiobooks that are part of the public domain (LibriVox). 
The link is: 
(https://www.openslr.org/12). 
# Noise databases
- non-stationary noisy: Freesound technical demo.
The link is: 
https://doi.org/10.1145/2502081.2502245.
- NOISEX92
Available at:
- https://github.com/speechdnn/Noises/tree/master/NoiseX-92
## Project structure
# data:
Mdl_linear1.mat          %%  Pre-trained linear-SVM model
Mdl_discriminantModel.mat %  Pre-trained pseudo-QDA model
# model 
net_rnn_gtcc2.mat                       %modified bi-lstm
# results
final_VAD_Decision_with_knocking_door_transient.fig
final_VAD_Decision_with_keyboaed_taps_transient.fig
response_gtcc.jpg
# scripts
gtcc_features.m  %function for Audio Pe-Processing and GFCCfeature extraction steps
training_VAD_hybrid_model.m 
test_VAD_hybrid_model.m    
smoothing_transient.m

function [coeffs] = gtcc_features(x,fs)
%%Apply preemphasis to the audio signal to amplify high-frequency components.
x = filter([1, -0.97], 1, x);
% Parameters
frame_length = 512;
overlap = 256;
num_coeffs = 13; % Number of cepstral coefficients
%%Compute GTCC features
[coeffs] = gtcc(x,fs, ...
                       'NumCoeffs',num_coeffs, ...
                        'Window',hamming(frame_length,"periodic"), ...
                       'OverlapLength',overlap, ...
                       'LogEnergy',"replace");

%Normalize the GFCC features
 coeffs = (coeffs - mean(coeffs(:))) / std(coeffs(:)); % Zero mean and unit variance normalization
end
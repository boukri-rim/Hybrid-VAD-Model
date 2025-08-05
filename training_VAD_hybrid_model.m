clear all
clc 
%%%%%%%%%%%%%%%%%%%first_stage training data
load('first_stage_lebels.mat','first_stage_lebels');
% % bilstm training data 
load('labelsTrain link.mat','labelsTrain');
labelsTrain=;
%Load the Audio:
filename_low_snr ='.wav';
[audio_low_snr, fs] = audioread(filename_low_snr);
filename_heigh_snr ='.wav';
[audio_heigh_snr, fs] = audioread(filename_heigh_snr);
audio=[audio_low_snr;audio_heigh_snr];
%%%%%%%%%
[featuresTrain_low] = gtcc_features(audio_low_snr,fs);           %feature training for SVM
YTrain_low =[first_stage_lebels(1,1:length(featuresTrain_low))];  % Training labels for SVM
[featuresTrain_heigh] = gtcc_features(audio_heigh_snr,fs);   %feature training for pseudo_QDA
YTrain_heigh =[first_stage_lebels(1,1:length(featuresTrain_heigh))];  % Training labels for pseudo_QDA
labelsTrain=[YTrain_low;YTrain_heigh;YTrain_low;YTrain_heigh]; %Training labels for modified Bi_lSTM
%%%%%%%%%%%%%%%%%%%%
t = templateSVM('Standardize',true,'KernelFunction','linear');
 Mdl_linear1= fitcecoc(featuresTrain_low,YTrain_low,'Learners',t);
    saveLearnerForCoder(Mdl_linear1,'Mdl_linear_low_heigh.mat')
%%%%%%%%%%%%%%%%%%%%%%%
  t = templateDiscriminant('DiscrimType','pseudoquadratic') ;
 Mdl_discriminantModel= fitcecoc(featuresTrain_heigh,YTrain_heigh,'Learners',t);
  save('Mdl_discriminantModel.mat')
%%%%%%%%%%%%%%%% training bi-lstm data

filename_transient ='.wav';
[transient, fs_transient] = audioread(filename_transient);
 audio_transient=audio+transient(1:length(audio),1);
audio_train=[audio;audio_transient];
[featuresTrain] = gtcc_features(audio_train,fs);  %feature training for modified Bi_lSTM
 % % Generate training data for modified bi_lstm
  YTestPred_SVM = predict(Mdl_linear1, featuresTrain)';
 YTestPred_dis_P_QDA = predict(Mdl_discriminantModel, featuresTrain)';
  YTestPred_SVM=logical(YTestPred_SVM);
   YTestPred_dis_P_QDA=logical(YTestPred_dis_P_QDA);
   T = size(featuresTrain, 1);
 for j=1:T
D_stage1(j)=YTestPred_SVM(j)&&YTestPred_dis_P_QDA(j);
 end
 D_stage1=double(D_stage1);
% Resize labels if needed
groundTruth = labelsTrain;
% Define network manually
inputSize = 15;
numHiddenUnits = 128;
numClasses = 2;
% define Layers 
layers = [
    sequenceInputLayer(inputSize, 'Name', 'input')
    bilstmLayer(numHiddenUnits, 'OutputMode','sequence', 'Name', 'bilstm1')
    bilstmLayer(numHiddenUnits, 'OutputMode','sequence', 'Name', 'bilstm2')
     bilstmLayer(numHiddenUnits, 'OutputMode','sequence', 'Name', 'bilstm3')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
];

lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

% Training setup with trainingOptions-like config
numEpochs = 100;
learningRate = 0.001;
gradientThreshold = 1;

lossHistory = zeros(T * numEpochs, 1);
accHistory = zeros(T * numEpochs, 1);
step = 1;
D_prev = 0;

for epoch = 1:numEpochs
    fprintf('Epoch %d\n', epoch);
    for t = 1:T
         inputFrame = [coeffs(t,:), D_prev,D_stage1(t)];
        assert(isequal(size(inputFrame), [1 15]), 'Input frame is not size [1 x 14]');
        dlX = dlarray(inputFrame', 'CBT');
        trueLabel = onehotencode(categorical(groundTruth(t), [0 1]), 1);
        dlYtrue = dlarray(trueLabel, 'CBT');
        [loss, gradients] = dlfeval(@modelLoss, dlnet, dlX, dlYtrue);
        % Apply gradient clipping
        gradients = dlupdate(@(g) max(min(g, gradientThreshold), -gradientThreshold), gradients);
        % Forward to get prediction for current D_curr (before tracking accuracy)
        dlY = forward(dlnet, dlX);
        probs = extractdata(dlY);
        [~, D_curr] = max(probs);
        D_curr = D_curr - 1; % Convert to 0/1
        % Track loss and accuracy
        lossHistory(step) = double(gather(extractdata(loss)));
        accHistory(step) = double(D_curr == groundTruth(t));
        step = step + 1;
        % Update network
        dlnet = dlupdate(@(w,g) w - learningRate * g, dlnet, gradients);
        % Set D_prev for next input
        D_prev = D_curr;
    end
end
% Save model
save('dlnet_vad_model.mat', 'dlnet');
disp('Training completed.')
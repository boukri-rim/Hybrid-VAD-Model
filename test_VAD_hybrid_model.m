clear all
clc
       load('Mdl_discriminantModel.mat');   %Pre-trained pseudo_QDA model
   Mdl=loadLearnerForCoder('Mdl_linear1.mat')      %Pre-trained linear-SVM model
  load('dlnet_vad_model.mat', 'dlnet');                   %Pre-trained modified Bi_LSTM model
filename_transient =strcat('..\446252__lukiacostello__knocking-then-pounding-on-wood-screen-door.wav');
[transient, fs_transient] = audioread(filename_transient);

  filename_transient1 =strcat('C:\Users\Admin\Desktop\article2\160869__pacmangamer__typewriter-keyboard-tapping.wav');
 [transient1, fs_transient] = audioread(filename_transient1);
 
% % % testing phase
  accuracy_bilstm_final=0;
for i=1:200
     i=num2str(i);
  load(strcat('link_data_correct\data',i,'_256.mat'),'YTest')
  filename = strcat('link_audio_clean\sa',i,'.wav');
   [x_clean, fs_clean] = audioread(filename);
    %x_clean= audioNormalization_YW(x_clean, 0.4);
filename = strcat('link_audio_noisy\sa',i,'restaurant_sn15.wav');
  [x_test, fs_test] = audioread(filename);
   x_test_tr=x_test+transient1(1:length(x_test),:);
   %%Apply preemphasis ,framing,windoing,and GFCC feature extraction
   [coeffs_test] = gtcc_features(x_test_tr,fs_test);
%%%%%%%%%%% SVM classifier
  YTestPred_SVM = predict(Mdl, coeffs_test);
YTestPred_SVM =YTestPred_SVM';
 %P_QDA classification
YTestPred_dis_P_QDA = predict(Mdl_discriminantModel, coeffs_test);
YTestPred_dis_P_QDA =YTestPred_dis_P_QDA';
YTestPred_SVM=double(string(YTestPred_SVM));
for j=1:length(coeffs_test)
D_stage1(j)=YTestPred_SVM(j)&&YTestPred_dis_P_QDA(j);
end
D_stage1=double(D_stage1);
T = size(coeffs_test, 1);
groundTruth=YTest';
groundTruth = groundTruth(1:T); % Make sure that this vector exists and corresponds to T

% Initialize the previous decision D_{i-1} of bilstm
D_prev = 0;
predictions = zeros(T,1);

for t = 1:T
    % Construction du vecteur d'entrée : 13 GFCC + D_first_stage{i} + D(i-1)
    inputFeature = [coeffs_test(t,:),D_stage1(t), D_prev];
    dlX = dlarray(reshape(inputFeature, [15 1 1]), 'CTB');
    % bilstm classification 
    dlY = predict(dlnet, dlX);
    %% Binary decision
   probs = extractdata(dlY);
     [~, D_curr] = max(probs);
     D_curr = D_curr - 1; % convert to 0 or 1
    predictions(t) = D_curr;
    D_prev = D_curr; % update for next frame
end
  
 [predictions] = smoothing(coeffs_test,predictions')';
% --- Calcul des métriques de performance (si groundTruth disponible) ---
if exist('groundTruth', 'var') && length(groundTruth) == T
    TP = sum(predictions == 1 & groundTruth == 1);
    TN = sum(predictions == 0 & groundTruth == 0);
    FP = sum(predictions == 1 & groundTruth == 0);
    FN = sum(predictions == 0 & groundTruth == 1);
 accuracy = (TP + TN) / T;

    fprintf("\nPerformance du modèle :\n");
    fprintf("Accuracy  : %.2f %%\n", accuracy * 100);
end
 timeVector =((0:size(x_test,1)-1)/256);%temp d"echantillon *nbr d"elm tableau
       figure
      %axis([0 1240 -1 1])
      %%%%%%%%%%%speaker1 with Knocking
       subplot(4,1,1)
     plot(timeVector,x_clean);
     ylim([-1, 1]);
     xlabel('Time(s)');
    ylabel('Amplitude');
    title('speaker1');
    % plot(YTestPred_gtcc,'g');
     %plot(YTest,'r');
 subplot(4,1,2)
     plot(timeVector,transient(1:length(x_test),1),'m');
     ylim([-1, 1]);
    xlabel('Time(s)');
    ylabel('Amplitude');
    title('transient: Knocking');
 subplot(4,1,3)
     hold on
      plot(timeVector,x_clean);
     plot(timeVector,transient(1:length(x_test),1),'m');
     ylim([-1, 1]);
     xlabel('Time(s)');
    ylabel('Amplitude');
    title('speaker1 with Knocking');
     hold off
     subplot(4,1,4)
          hold on
                plot(timeVector,x_clean);
                ylim([-1, 1]);
     plot(timeVector,transient(1:length(x_test1),1),'m');
    plot(y_final1,'g');
%%    plot(timeVector, noise(1:length(x_clean),1),'y');
    xlabel('Time(s)');
    ylabel('Amplitude');
    title('Hybrid final decision');
     hold off
% %      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%speaker1 with Keyboard taps
% figure
%       subplot(4,1,1)
%     plot(timeVector,x_clean);
%     ylim([-1, 1]);
%      xlabel('Time(s)');
%     ylabel('Amplitude');
%     title('speaker1');
%      hold off
%  % plot(timeVector,x_test);
%        subplot(4,1,2)
%             plot(timeVector,transient1(1:length(x_test),1),'m');
%             ylim([-1, 1]);
%     xlabel('Time(s)');
%     ylabel('Amplitude');
%     title('transient: Keyboard taps');
% 
%  subplot(4,1,3)
% hold on
%       plot(timeVector,x_clean);
%      plot(timeVector,transient1(1:length(x_test),1),'m');
%      ylim([-1, 1]);
%      xlabel('Time(s)');
%     ylabel('Amplitude');
%     title('speaker1 with Keyboard taps');
%     hold off
%    subplot(4,1,4)
%    hold on
%     plot(timeVector,x_clean);
%     ylim([-1, 1]);
%      plot(timeVector,transient1(1:length(x_test),1),'m');
%     plot(y_final,'g');
%     xlabel('Time(s)');
%     ylabel('Amplitude');
%     title('Hybrid final decision');
%      hold off
D_stage1=0;
end

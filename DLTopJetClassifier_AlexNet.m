%% Transfer Learning with AlexNet for Top Quark Jet Tagging
% Reads pre-saved PNGs from JetLabels/ (train) and ValLabels/ (validation).
% Requires: Deep Learning Toolbox, Deep Learning Toolbox Model for AlexNet
% (install via Add-On Explorer if not present: search "AlexNet").

clear; clc; close all;

%% ------------------- Parameters -------------------
trainFolder = fullfile(pwd,'Train');
valFolder   = fullfile(pwd,'Val');

maxEpochs        = 3;      % AlexNet converges faster than the basic CNN since it's pretrained
miniBatchSize     = 64;    % AlexNet is heavier than the basic CNN, smaller batch is safer for 16GB RAM/RTX4050
initialLearnRate = 1e-4;   % lower LR for fine-tuning a pretrained network
freezeEarlyLayers = true;  % speeds up training, usually fine for transfer learning

%% ------------------- Load AlexNet -------------------
try
    net0 = alexnet;
    fprintf('✓ AlexNet loaded\n');
catch
    error('AlexNet not installed. Go to Add-On Explorer and install "Deep Learning Toolbox Model for AlexNet Network".');
end

inputSize = net0.Layers(1).InputSize;   % [227 227 3]
fprintf('AlexNet expects input size: %s\n', mat2str(inputSize));

%% ------------------- Load datastores -------------------
imdsTrain = imageDatastore(trainFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
imdsVal   = imageDatastore(valFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

fprintf('Train images: %d | Val images: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files));

% Resize + convert grayscale (37x37x1) to RGB (227x227x3) on-the-fly.
% augmentedImageDatastore does this without duplicating data on disk.
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing','gray2rgb');
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal,   'ColorPreprocessing','gray2rgb');

%% ------------------- Replace final layers for 2-class output -------------------
layersTransfer(1) = imageInputLayer(inputSize, "Name", layersTransfer(1).Name, "Normalization","none");

if freezeEarlyLayers
    for i = 1:numel(layersTransfer)
        if isprop(layersTransfer(i),'WeightLearnRateFactor')
            layersTransfer(i).WeightLearnRateFactor = 0;
            layersTransfer(i).BiasLearnRateFactor = 0;
        end
    end
    fprintf('✓ Early layers frozen\n');
end

numClasses = 2;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,"Name","fc_jet","WeightLearnRateFactor",10,"BiasLearnRateFactor",10)
    softmaxLayer("Name","softmax_jet")
    classificationLayer("Name","classoutput_jet")];

clear net0 layersTransfer   % only 'layers' is needed from here on; net0 holds the full pretrained AlexNet (~230MB)

%% ------------------- Train -------------------
options = trainingOptions('adam', ...
    'InitialLearnRate',initialLearnRate, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',50, ...
    'ExecutionEnvironment','auto', ...        % uses GPU automatically if available
    'DispatchInBackground',true, ...          % parallel prefetching, avoids holding full dataset in RAM
    'CheckpointPath',pwd, ...
    'CheckpointFrequency',1, ...
    'CheckpointFrequencyUnit','epoch');

fprintf('\n========== STARTING ALEXNET TRANSFER LEARNING ==========\n');
netTransfer = trainNetwork(augTrain, layers, options);
save('trainedNetwork_AlexNetJetTagging.mat','netTransfer');
fprintf('✓ Training complete, network saved\n');
clear augTrain imdsTrain layers   % no longer needed; only netTransfer + validation data used from here

%% ------------------- Evaluate -------------------
[labelsPredicted, scores] = classify(netTransfer, augVal);
labelsTrue = imdsVal.Labels;

C = confusionmat(labelsTrue, labelsPredicted);
classificationAccuracy = trace(C)/sum(C(:));
fprintf('Validation accuracy: %.2f%%\n', 100*classificationAccuracy);

figure; confusionchart(labelsTrue, labelsPredicted, "Title","AlexNet Confusion matrix", 'FontSize',14);

topJetsIdx = find(netTransfer.Layers(end).Classes == 'TopJets');
[X,Y,~,AUC] = perfcurve(labelsTrue, scores(:,topJetsIdx), 'TopJets');
figure; plot(X,Y);
xlabel('False positive rate','FontSize',14);
ylabel('True positive rate','FontSize',14);
title(sprintf('ROC - AlexNet transfer learning (AUC = %.3f)',AUC),'FontSize',16);

fprintf('✓ Script completed successfully\n');

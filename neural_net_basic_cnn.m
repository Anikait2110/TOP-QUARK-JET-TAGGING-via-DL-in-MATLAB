%% Basic CNN Training - Top Quark Jet Tagging
% Reads pre-saved PNGs from Train/ and Val/ folders (created by train.m and val.m).
% Uses augmentedImageDatastore with parallel background dispatch.

clear; clc; close all;

%% ------------------- Parameters -------------------
trainFolder = fullfile(pwd,'Train');
valFolder   = fullfile(pwd,'Val');

Nb = 37;   % image size, must match what train.m/val.m produced

maxEpochs         = 15;
miniBatchSize      = 128;
initialLearnRate  = 1e-3;

%% ------------------- Ensure parallel pool is running -------------------
poolobj = gcp('nocreate');
if isempty(poolobj)
    poolobj = parpool;   % starts default Parallel Computing Toolbox pool
end
fprintf('✓ Parallel pool active: %d workers\n', poolobj.NumWorkers);

%% ------------------- Load datastores -------------------
imdsTrain = imageDatastore(trainFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
imdsVal   = imageDatastore(valFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

fprintf('Train images: %d | Val images: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files));

% augmentedImageDatastore resizes/reformats on the fly and, with
% DispatchInBackground true, prefetches batches using the parallel pool
% instead of blocking the main training loop on disk reads.
augTrain = augmentedImageDatastore([Nb Nb], imdsTrain, 'ColorPreprocessing','none');
augVal   = augmentedImageDatastore([Nb Nb], imdsVal,   'ColorPreprocessing','none');

%% ------------------- CNN architecture -------------------
layers = [
    imageInputLayer([Nb Nb 1],"Name","imageinput","Normalization","none")
    convolution2dLayer([4 4],64,"Name","conv_1","Padding","same","Stride",[2 2],"WeightLearnRateFactor",0.1)
    reluLayer("Name","relu1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],64,"Name","conv_2","Padding","same","Stride",[2 2],"WeightLearnRateFactor",0.1)
    reluLayer("Name","relu2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],64,"Name","conv_3","Padding","same","Stride",[2 2],"WeightLearnRateFactor",0.1)
    reluLayer("Name","relu3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    fullyConnectedLayer(2,"Name","fc","WeightLearnRateFactor",0.1)
    reluLayer("Name","relu4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

figure; plot(layerGraph(layers)); title('Jet-tagging CNN architecture');

%% ------------------- Train -------------------
options = trainingOptions('adam', ...
    'InitialLearnRate',initialLearnRate, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto', ...        % uses GPU automatically if available
    'DispatchInBackground',true, ...          % parallel prefetching via Parallel Computing Toolbox
    'CheckpointPath',pwd, ...
    'CheckpointFrequency',1, ...
    'CheckpointFrequencyUnit','epoch');

fprintf('\n========== STARTING TRAINING ==========\n');
net = trainNetwork(augTrain, layers, options);
save('trainedNetwork_BasicCNN.mat','net');
fprintf('✓ Training complete, network saved\n');
clear augTrain imdsTrain layers   % no longer needed; only net + validation data used from here

%% ------------------- Evaluate -------------------
[labelsPredicted, scores] = classify(net, augVal);
labelsTrue = imdsVal.Labels;

C = confusionmat(labelsTrue, labelsPredicted);
classificationAccuracy = trace(C)/sum(C(:));
fprintf('Validation accuracy: %.2f%%\n', 100*classificationAccuracy);

figure; confusionchart(labelsTrue, labelsPredicted, "Title","Confusion matrix", 'FontSize',14);

topJetsIdx = find(net.Layers(end).Classes == 'TopJets');
[X,Y,~,AUC] = perfcurve(labelsTrue, scores(:,topJetsIdx), 'TopJets');
figure; plot(X,Y);
xlabel('False positive rate','FontSize',14);
ylabel('True positive rate','FontSize',14);
title(sprintf('ROC for top-jet tagging (AUC = %.3f)',AUC),'FontSize',16);

fprintf('✓ Script completed successfully\n');

%% Deep Learning for Top Quark Jet Tagging - Validation set preprocessing
% Mirrors train.m exactly, but processes val.parquet.gzip -> Val/ folder.

clear; clc; close all;

%% ------------------- User-configurable parameters -------------------
parquetFile = 'val.parquet.gzip';        % source dataset
outputRoot  = fullfile(pwd,'Val');
bkgFolder   = fullfile(outputRoot,'Background');
topFolder   = fullfile(outputRoot,'TopJets');

Nb            = 37;    % number of eta/phi buckets -> Nb x Nb jet image
gridHalfWidth = 1.6;   % eta/phi grid spans [-gridHalfWidth, gridHalfWidth]

if ~exist(bkgFolder,'dir'); mkdir(bkgFolder); end
if ~exist(topFolder,'dir'); mkdir(topFolder); end

%% ------------------- Load and organize raw data -------------------
T_jets = parquetread(parquetFile);
T_jets = sortrows(T_jets,'is_signal_new');

G = groupsummary(T_jets,'is_signal_new');
fprintf('✓ Data loaded and sorted\n');
nBkg = G.GroupCount(1);
nTop = G.GroupCount(2);
fprintf('Loaded %d jets (%d background, %d top).\n', nBkg+nTop, nBkg, nTop);

%% ------------------- Jet kinematics -------------------
Jet = Table_To_3DTensor(T_jets);   % 4 x 35 x N: four-momenta of leading 35 constituents
nJets = size(Jet,3);
clear T_jets                       % no longer needed once Jet is extracted

pT  = sqrt(Jet(:,2,:).^2 + Jet(:,3,:).^2);        % transverse momentum
eta = squeeze(asinh(Jet(:,4,:)./pT));             % pseudorapidity, 35 x N
phi = squeeze(atan2(Jet(:,3,:),Jet(:,2,:)));      % azimuth, 35 x N
clear Jet                                          % raw four-momenta no longer needed

eta = eta - eta(1,:);   % center the leading (hardest) constituent at eta=phi=0
phi = phi - phi(1,:);

Xedges = linspace(-gridHalfWidth,gridHalfWidth,Nb+1);
Yedges = linspace(-gridHalfWidth,gridHalfWidth,Nb+1);

%% ------------------- Build jet images (pT-weighted pixel intensity) -------------------
xGrp = discretize(eta,Xedges);
yGrp = discretize(phi,Yedges);
zGrp = repelem(1:nJets,size(eta,1));

V = rmmissing([pT(:) xGrp(:) yGrp(:) zGrp(:)]);   % drop constituents outside the grid
clear eta phi xGrp yGrp zGrp pT
fprintf('✓ Jet kinematics computed (eta, phi, pT)\n');

JetImage = accumarray(V(:,2:4), V(:,1), [Nb, Nb, nJets]);
fprintf('✓ Jet images created via accumarray\n');
clear V

% Center each image on its hottest pixel and normalize in batches
JetImageFlat = zeros(Nb^2, nJets);
batchSize = 2000;
nBatches = ceil(nJets/batchSize);

for b = 1:nBatches
    idx = (b-1)*batchSize+1 : min(b*batchSize, nJets);
    batchImages = JetImage(:,:,idx);        % only this small chunk is sent to workers
    nIdx = numel(idx);
    batchFlat = zeros(Nb^2, nIdx);
    parfor k = 1:nIdx
        col = reshape(batchImages(:,:,k),Nb^2,1);
        col = circshift(col, ceil(Nb^2/2) - find(col==max(col),1));
        batchFlat(:,k) = col;
    end
    JetImageFlat(:,idx) = batchFlat;
    clear batchImages batchFlat
    fprintf('Batch %d/%d done\n', b, nBatches);
end

for b = 1:nBatches
    idx = (b-1)*batchSize+1 : min(b*batchSize, nJets);
    JetImageFlat(:,idx) = normalize(JetImageFlat(:,idx),'norm');
    fprintf('Normalize batch %d/%d done\n', b, nBatches);
end
fprintf('✓ All images normalized\n');

JetImage = reshape(JetImageFlat,Nb,Nb,[]);
clear JetImageFlat

for b = 1:nBatches
    idx = (b-1)*batchSize+1 : min(b*batchSize, nJets);
    JetImage(:,:,idx) = zscore(JetImage(:,:,idx),0,3);
    JetImage(:,:,idx) = rescale(JetImage(:,:,idx));
end
fprintf('✓ Images standardized and rescaled\n');

figure;
subplot(1,2,1); imshow(mean(JetImage(:,:,1:nBkg),3),[]);     title('Mean background jet');
subplot(1,2,2); imshow(mean(JetImage(:,:,nBkg+1:end),3),[]); title('Mean top jet');

%% ------------------- Save labeled images to disk -------------------
g = min(nBkg,nTop);
saveBatchSize = 2000;
nSaveBatches = ceil(g/saveBatchSize);
for b = 1:nSaveBatches
    idx = (b-1)*saveBatchSize+1 : min(b*saveBatchSize, g);
    bkgChunk = JetImage(:,:,idx);
    topChunk = JetImage(:,:,idx+g);
    parfor k = 1:numel(idx)
        aa = idx(k);
        imwrite(bkgChunk(:,:,k), fullfile(bkgFolder, sprintf('jet%d.png',aa)));
        imwrite(topChunk(:,:,k), fullfile(topFolder, sprintf('jet%d.png',aa)));
    end
    clear bkgChunk topChunk
    fprintf('Saved batch %d/%d\n', b, nSaveBatches);
end
fprintf('Saved %d image pairs to Val/\n', g);
clear JetImage nJets g nBkg nTop

%% ------------------- Helper function -------------------
function y = Table_To_3DTensor(x)
    y = x{:,2:4*35+1}';
    y = reshape(y,4,35,[]);
    y = pagetranspose(y);
end
function info = fcnTest(varargin)
dbstop if error;
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/NYU' ;
opts.modelFamily = 'matconvnet' ;
opts.mode = 'multi' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.modelPath = fullfile(opts.expDir, 'net-epoch-8.mat') ;
opts.imdbPath = fullfile(opts.expDir, 'nyu_new_imdb.mat') ;
opts.splitPath = fullfile(opts.expDir, 'splits.mat') ;
opts.metaPath = fullfile(opts.expDir, 'new_meta.mat') ;
opts.nyuAdditionalSegmentations = true ;
opts.gpus = [2] ;
opts = vl_argparse(opts, varargin) ;

resPath = fullfile(opts.expDir, 'results.mat') ;
% if exist(resPath)
%   info = load(resPath) ;
%   return ;
% end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% Setup data
if exist(opts.imdbPath)
  	imdb = load(opts.imdbPath) ;
else
	keyboard
end

% setup training and test/validation subsets
if exist(opts.splitPath)
	split = load(opts.splitPath) ;
else
	keyboard
end
if exist(opts.metaPath)
	meta = load(opts.metaPath) ;
else
	keyboard
end
% Get training and test/validation subsets
train = split.trainNdxs ;
val = split.testNdxs ;
rgbMean = meta.rgbMean ;

net.meta.normalization.rgbMean = rgbMean ;
net.meta.classes = imdb.className ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
  case 'matconvnet'
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.mode = 'test' ;
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    switch opts.mode
    case 'image'
      net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean(1,:),1,1,3) ;
    case 'depth'
      net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean(2,:),1,1,3) ;
    case 'multi'
      net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean',1,1,6) ;
    end
    
    predVar = net.getVarIndex('fusion_2') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1)) ;
  net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

numGpus = 0 ;
confusion = zeros(40) ;
pixel_a = 0;
tp_a = zeros(1,40);
pos_a = zeros(1,40);
count = 0;

for i = 1:numel(val)
  imId = val(i) ;
  
  depth = imdb.depths(:,:,:,imId) ;
%   depth = depth2rgb(depth);
  rgb = imdb.images(:,:,:,imId) ;

  img_six(:,:,1:3) = rgb;
  img_six(:,:,4:6) = depth;

  anno = imdb.labels(:,:,imId) ;
  % Load an image and gt segmentation
  lb = single(anno) ;

  switch opts.mode
    case 'image'
      im = rgb;
    case 'depth'
      im = depth;
    case 'multi'
      im = img_six;
  end

  % Subtract the mean (color)
  im = bsxfun(@minus, single(im), net.meta.normalization.averageImage) ;

  % Soome networks requires the image to be a multiple of 32 pixels
  if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end

  if ~isempty(opts.gpus)
    im_ = gpuArray(im_) ;
  end

  net.eval({inputVar, im_}) ;
  scores_ = gather(net.vars(predVar).value) ;
  [~,pred_] = max(scores_,[],3) ;

  if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
  else
    pred = pred_ ;
  end

  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[40 40]) ;

  % Plots
  if mod(i - 1,30) == 0 || i == numel(val)
    clear info ;
%     [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
    [info.iu, info.miu, info.pacc, info.macc,tp,pos] = getAccuracies(confusion); 
%% ==
    pixel_a = pixel_a+info.pacc;
    tp_a = tp_a+tp';
    pos_a = pos_a+pos';
    count = count+1;
 %% ==
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * info.iu) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
            100*info.miu, 100*info.pacc, 100*info.macc) ;

%     figure(1) ; clf;
%     imagesc(normalizeConfusion(confusion)) ;
%     axis image ; set(gca,'ydir','normal') ;
%     colormap(jet) ;
%     drawnow ;
% 
%     % Print segmentation
%     figure(100) ;clf ;
%     displayImage(rgb, lb, pred) ;
%     drawnow ;
% 
%     % Save segmentation
%     imPath = fullfile(opts.expDir, [num2str(imId) '.png']) ;
%     imwrite(pred,labelColors(),imPath,'png');
  end
end

pixel_a = pixel_a/count;
mean_a = tp_a./max(1,pos_a); 

% Save results
save(resPath, '-struct', 'info') ;

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy,tp,pos] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

% -------------------------------------------------------------------------
function displayImage(im, lb, pred)
% -------------------------------------------------------------------------
subplot(2,2,1) ;
image(im) ;
axis image ;
title('source image') ;

subplot(2,2,2) ;
image(uint8(lb)) ;
axis image ;
title('ground truth')

cmap = labelColors() ;
subplot(2,2,3) ;
image(uint8(pred)) ;
axis image ;
title('predicted') ;

colormap(cmap) ;

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N = 40;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
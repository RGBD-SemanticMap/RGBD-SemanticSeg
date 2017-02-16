function info = getPrediction(varargin)
dbstop if error;
% run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/NYU' ;
opts.modelFamily = 'matconvnet' ;
opts.mode = 'multi' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.modelPath = fullfile(opts.expDir, 'final_net.mat') ;
opts.imdbPath = fullfile(opts.expDir, 'nyu_imdb.mat') ;
opts.splitPath = fullfile(opts.expDir, 'splits.mat') ;
opts.metaPath = fullfile(opts.expDir, 'meta.mat') ;
opts.nyuAdditionalSegmentations = true ;
opts.gpus = [] ;
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
      net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,6) ;
    end
    
    predVar = net.getVarIndex('prediction') ;
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

chosen = [117 118 40 137 220 310 447 516 551 561 579 586 802 803];
for i = 1 : numel(chosen)
  imId = chosen(i) ;
  
  depth = imdb.depths(:,:,imId) ;
  depth = depth2rgb(depth);
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
  temp = exp(scores_);
  unary = temp./repmat(sum(temp,3),[1 1 40]);
  fname = fullfile(opts.expDir, ['unary_final_' num2str(chosen(i))]);
  save(fname, 'unary');
end

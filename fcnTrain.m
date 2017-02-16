function fcnTrain(varargin)
    dbstop if error;
    
% 	run matconvnet/matlab/vl_setupnn ;
	addpath matconvnet/examples ;

	% experiment and data paths
	%opts.expDir = fullfile('data', 'fcn8s-msrc') ;
	opts.expDir = fullfile('data', 'NYU') ;
% 	opts.modelType = 'fcn8s' ;
    opts.mode = 'multi' ;
	opts.sourceModelPath_image = 'data/models/net-448-epoch-9.mat' ; 
    opts.sourceModelPath_depth = 'data/models/net-448-epoch-10.mat' ; 
	[opts, varargin] = vl_argparse(opts, varargin) ;

	% experiment setup
	opts.imdbPath = fullfile(opts.expDir, 'nyu_aug5_imdb.mat') ;
	opts.splitPath = fullfile(opts.expDir, 'aug5_splits.mat') ;
    opts.metaPath = fullfile(opts.expDir, 'new_meta.mat') ;
	%opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
	opts.nyuAdditionalSegmentations = true ;
    opts.numFetchThreads = 1 ; % not used yet


	% training options (SGD)
	opts.train = struct() ;
	[opts, varargin] = vl_argparse(opts, varargin) ;
	if ~isfield(opts.train, 'gpus'), opts.train.gpus = [2]; end;

	trainOpts.batchSize = 20 ;
	trainOpts.numSubBatches = 10;
	trainOpts.continue = true ;
	trainOpts.gpus = [2] ;
    
	trainOpts.prefetch = false ;
	trainOpts.expDir = opts.expDir ;
	trainOpts.learningRate = 0.000001 * ones(1,50) ;
	trainOpts.numEpochs = numel(trainOpts.learningRate) ;

	%% Setup data
	if exist(opts.imdbPath)
  		imdb = load(opts.imdbPath) ;
	else
		% not finish yet
  		% imdb = nyuSetup('dataDir', opts.dataDir, ...
    % 		'includeSegmentation', true) ;
  		% mkdir(opts.expDir) ;
  		% save(opts.imdbPath, '-struct', 'imdb') ;
  		keyboard
	end

	%% setup training and test/validation subsets
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
	% % Get dataset statistics
	% if exist(opts.imdbStatsPath)
 %  		stats = load(opts.imdbStatsPath) ;
 %    else
 %  		stats = getDatasetStatistics(imdb) ;
 %  		save(opts.imdbStatsPath, '-struct', 'stats') ;
	% end

	% Get initial model from VGG-VD-16
%     nopts.sourceModelPath_image = opts.sourceModelPath_image;
%     nopts.sourceModelPath_depth = opts.sourceModelPath_depth;
%     [nopts, varargin] = vl_argparse(nopts, varargin) ;
% 	net = MultiModel_y(nopts);
    nopts.sourceModelPath_image = opts.sourceModelPath_image;
    nopts.sourceModelPath_depth = opts.sourceModelPath_depth;
    [nopts, varargin] = vl_argparse(nopts, varargin) ;
	net = multiModel_fus2(nopts);
% 	if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
%   		% upgrade model to FCN16s
%   		net = fcnInitializeModel16s(net) ;
% 	end
% 	if strcmp(opts.modelType, 'fcn8s')
%   		% upgrade model fto FCN8s
%   		net = fcnInitializeModel8s(net) ;
% 	end
	net.meta.normalization.rgbMean = rgbMean ;
	net.meta.classes = imdb.className ;
    % net.meta.classes = imdb.names ;
    
%     keyboard
	%% Train
	% Setup data fetching options
	%bopts.numThreads = opts.numFetchThreads ;
	bopts.labelStride = 1 ;
	bopts.labelOffset = 1 ;
	bopts.classWeights = ones(1,40,'single') ;
	bopts.rgbMean = rgbMean ;
	bopts.useGpu = numel(opts.train.gpus) > 0 ;
    bopts.mode = opts.mode ;
	% Launch SGD
%     keyboard
	info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), ...
                     	trainOpts, ....
                     	'train', train, ...
                     	'val', val, ...
                     	opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
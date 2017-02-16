function y = getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [384 384]+64;
%opts.imageSize = [210, 210] ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 0 ;
opts.classWeights = ones(1,40,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useGpu = false ;
opts.mode = 'image';
opts = vl_argparse(opts, varargin);


if opts.prefetch
  % to be implemented
  ims = [] ;
  labels = [] ;
  return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
  opts.rgbMean = single([128;128;128; 128;128;128]) ;
end
if ~isempty(opts.rgbMean)               
      opts.rgbMean = [opts.rgbMean(1,:) opts.rgbMean(2,:)];
      opts.rgbMean = single(reshape(opts.rgbMean, [1 1 6])) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 6, ...
  numel(images)*opts.numAugments, 'single') ;

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize(2) ;
ly = opts.labelOffset : opts.labelStride : opts.imageSize(1) ;
labels = zeros(numel(ly), numel(lx), 1, numel(images)*opts.numAugments, 'single') ;
classWeights = [0 opts.classWeights(:)'] ;

im = cell(1,numel(images)) ;

si = 1 ;

for i=1:numel(images)

  % acquire image
  if isempty(im{i})
    % rgbPath = sprintf(imdb.paths.image, imdb.images.name{images(i)}) ;
    % labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name_GT{images(i)}) ;
    % rgb = vl_imreadjpeg({rgbPath}) ;

      depth_path = imdb.depths(images(i)) ;
%       depth = vl_imreadjpeg(depth_path);
      depth = load(depth_path{1});
      depth = depth.temp_depth;
%         depth = imdb.depths(:,:,:,images(i));
      
%       opts.rgbMean = opts.rgbMean(2,:);
     % depth = depth2rgb(depth);

      rgb_path = imdb.images(images(i)) ;
      rgb = load(rgb_path{1});
      rgb = rgb.temp_image;
%         rgb = imdb.images(:,:,:,images(i));
%       opts.rgbMean = opts.rgbMean(1,:);
      img_six(:,:,1:3) = rgb;
      img_six(:,:,4:6) = depth;

    % rgb = rgb{1} ;
    %rgb = imread(rgbPath) ;
    % anno = imread(labelsPath) ;
    anno_path = imdb.labels(images(i)) ;
    anno = load(anno_path{1});
    anno = anno.temp_label;
%     anno = imdb.labels(:,:,images(i));
  else
    img_six = im{i} ;
  end
  
%     %%%% incomplete
% %     rgb = %%% depth to rgb ;
%     

  img_six = single(img_six);

%   % crop & flip
%   h = size(img_six,1) ;
%   w = size(img_six,2) ;
%   for ai = 1:opts.numAugments
%     sz = opts.imageSize(1:2) ;
%     scale = max(h/sz(1), w/sz(2)) ;
%     scale = scale .* (1 + (rand(1)-.5)/5) ;
%     sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
%     sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
%     if rand > 0.5, sx = fliplr(sx) ; end
% 
%     okx = find(1 <= sx & sx <= w) ;
%     oky = find(1 <= sy & sy <= h) ;
%     if ~isempty(opts.rgbMean)
%       ims(oky,okx,:,si) = bsxfun(@minus, img_six(sy(oky),sx(okx),:), opts.rgbMean) ;
%     else
%       ims(oky,okx,:,si) = img_six(sy(oky),sx(okx),:) ;
%     end
% 
% %     llabel = zeros(sz(1)*sz(2), 1);
% % 
% %     tlabels = zeros(sz(1), sz(2)) ;
% %     tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
% %     tlabels = single(tlabels(ly,lx)) ;
% %     tlabels = reshape(tlabels, sz(1)*sz(2), 1) ;
% %     for l = 1:40
% %       idx_l = find(ismember(tlabels, l, 'rows')) ;
% %       llabel(idx_l) = l ;
% %     end
% %     labels(:,:,si) = reshape(llabel, sz(1), sz(2)) ;
% 
%     tlabels = zeros(sz(1), sz(2)) ;
%     tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
%     tlabels = single(tlabels(ly,lx)) ;
%     labels(:,:,si) = tlabels;
%     
%     si = si + 1 ;
%   end
ims(:,:,:,si) = img_six;
labels(:,:,si) = anno;
si = si+1;
  
end

if opts.useGpu
  ims = gpuArray(ims) ;
end
y = {'input', ims, 'label', labels} ;

function imdb = msrcSetup(varargin)

opts.dataDir = fullfile('data', 'NYU') ;
opts.includeTest = false ;
opts.includeSegmentation = false ;
% opts.archiveDir = fullfile('data', 'archives') ;
opts.mode = 'depth'
opts = vl_argparse(opts, varargin) ;

% Source images and classes
imdb.imdbPath = esc(fullfile(opts.dataDir, 'nyu_depth_v2_labeled.mat')) ;

imdb.imdbMode = opts.mode


% imdb.paths.classSegmentation = esc(fullfile(opts.dataDir, 'GroundTruth', '%s.bmp'));
imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.classes.id = uint8(1:21) ;
imdb.classes.name = {...
  'building', 'grass', 'tree', 'cow', 'sheep', 'sky', 'aeroplane', ...
  'water', 'face', 'car', 'bicycle', 'flower', 'sign', 'bird', ...
  'book', 'chair', 'road', 'cat', 'dog', 'body', 'boat'} ;
%imdb.classes.images = cell(1,21) ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.name_GT = {} ;
imdb.images.set = [] ;
index = containers.Map() ;
[imdb, index] = addImageSet(opts, imdb, index, 'train', 1) ;
[imdb, index] = addImageSet(opts, imdb, index, 'val', 2) ;
if opts.includeTest, [imdb, index] = addImageSet(opts, imdb, index, 'test', 3) ; end

% Compress data types
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;
% for i=1:21
%   imdb.classes.images{i} = uint32(imdb.classes.images{i}) ;
% end

% Check images on disk and get their size
imdb = getImageSizes(imdb) ;

% -------------------------------------------------------------------------
function [imdb, index] = addImageSet(opts, imdb, index, setName, setCode)
% -------------------------------------------------------------------------
j = length(imdb.images.id) ;

annoPath = fullfile('doc', [setName '.txt']) ;
fprintf('%s: reading %s\n', mfilename, annoPath) ;
names = textread(annoPath, '%s') ;
suffix = '.bmp' ;

for i=1:length(names)
  if ~index.isKey(names{i})
    j = j + 1 ;
    index(names{i}) = j ;
    imdb.images.id(j) = j ;
    imdb.images.set(j) = setCode ;
    imdb.images.name{j} = names{i}(1:end-4) ;
    imdb.images.name_GT{j} = [names{i}(1:end-4) '_GT'] ;
    imdb.images.segmentation(j) = true ;
  end
end

% -------------------------------------------------------------------------
function imdb = getImageSizes(imdb)
% -------------------------------------------------------------------------
for j=1:numel(imdb.images.id)
  info = imfinfo(sprintf(imdb.paths.image, imdb.images.name{j})) ;
  imdb.images.size(:,j) = uint16([info.Width ; info.Height]) ;
  fprintf('%s: checked image %s [%d x %d]\n', mfilename, imdb.images.name{j}, info.Height, info.Width) ;
end

% -------------------------------------------------------------------------
function str=esc(str)
% -------------------------------------------------------------------------
str = strrep(str, '\', '\\') ;



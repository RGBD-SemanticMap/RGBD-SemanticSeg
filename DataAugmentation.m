%Input database: NYUV2 output: 448*448 augmentated images
dbstop if error;
RootDir = '/media/pub/yq/xy/data/NYU';
imdbDir = fullfile(RootDir,'nyu_new_imdb.mat');
splitsDir = fullfile(RootDir,'splits.mat');
metaDir = fullfile(RootDir,'new_meta.mat');

image_size = [448,448];
num_augmentation = 5;

imdb = load(imdbDir);
splits = load(splitsDir);
meta = load(metaDir);
rgbMean = meta.rgbMean;
trainid = splits.trainNdxs;
testid = splits.testNdxs;
className = imdb.className;

depths = imdb.depths;
images = imdb.images;
images = single(images);
depths = single(depths);
annos = imdb.labels;



h = size(depths,1);
w = size(depths,2);

depth_path = fullfile(RootDir,'dataset/depth');
image_path = fullfile(RootDir,'dataset/image');
label_path = fullfile(RootDir,'dataset/label');
if ~exist(depth_path)
	mkdir(depth_path);
end
if ~exist(image_path)
	mkdir(image_path);
end
if ~exist(label_path)
	mkdir(label_path);
end

result_depth = cell(1,num_augmentation*numel(trainid)+numel(testid));
result_image = cell(1,num_augmentation*numel(trainid)+numel(testid));
labels = cell(1,num_augmentation*numel(trainid)+numel(testid));

newtrain = [];
newtest = [];

si = 1;
lx = 1:image_size(2);
ly = 1:image_size(1);

for num = 1:size(depths,4)
	depth = depths(:,:,:,num);
	image = images(:,:,:,num);
	anno = annos(:,:,num);

	if find(trainid == num)
		augment = num_augmentation;
	else
		augment = 1;
	end

	for k = 1:augment
		temp_depth = zeros(image_size(1),image_size(2),3,'single');
		temp_image = zeros(image_size(1),image_size(2),3,'single');
		temp_label = zeros(image_size(1),image_size(2),3,'single');

		sz = image_size(1:2);
		scale = max(h/sz(1),w/sz(2));
		scale = scale.*(1 + 2*(rand(1)-.5)/5);
		sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
	    sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
	    if rand > 0.5, sx = fliplr(sx) ; end

	    okx = find(1 <= sx & sx <= w) ;
	    oky = find(1 <= sy & sy <= h) ;

	    temp_depth(oky,okx,:) = bsxfun(@minus,depth(sy(oky),sx(okx),:),single(reshape(rgbMean(2,:),[1,1,3]))) ;
	    temp_image(oky,okx,:) = bsxfun(@minus,image(sy(oky),sx(okx),:),single(reshape(rgbMean(1,:),[1,1,3]))) ;
	    temp_depth(oky,okx,:) = bsxfun(@plus,temp_depth(oky,okx,:),single(randn(1,1,3)));
	    temp_image(oky,okx,:) = bsxfun(@plus,temp_image(oky,okx,:),single(randn(1,1,3)));

        
	    tlabels = zeros(sz(1), sz(2)) ;
	    tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
	    temp_label = single(tlabels(ly,lx)) ;
	    
	    save(fullfile(depth_path,[num2str(si),'.mat']),'temp_depth');
	    save(fullfile(image_path,[num2str(si),'.mat']),'temp_image');
	    save(fullfile(label_path,[num2str(si),'.mat']),'temp_label');
       

	    result_depth{si} = fullfile(depth_path,[num2str(si),'.mat']);
	    result_image{si} = fullfile(image_path,[num2str(si),'.mat']);
	    labels{si} = fullfile(label_path,[num2str(si),'.mat']);
	   
	    if augment == 1
	    	newtest = [newtest,si];
	    else
	    	newtrain = [newtrain,si];
        end
        
        si = si + 1 ;
    end
    fprintf('%d is finished\n',num);
end

trainNdxs = newtrain;
testNdxs = newtest;
depths = result_depth;
images = result_image;
save('./data/NYU/nyu_aug5_imdb.mat','className','depths','images','labels','-v7.3');
save('./data/NYU/aug5_splits.mat','trainNdxs','testNdxs');

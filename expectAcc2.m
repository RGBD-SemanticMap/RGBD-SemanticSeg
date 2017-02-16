dbstop if error;
expoPath = 'data/resNetPred/';

rgbCorrect = [];
depthCorrect = [];
fuseCorrect = [];

for i = 1 : 1449
	% heat map of rgb
	if ~exist([expoPath, 'RGB/', num2str(i), '.png'])
		continue;
	end
	
	rgbPred = imread([expoPath, 'RGB/', num2str(i), '.png']);
	depthPred = imread([expoPath, 'Depth/', num2str(i), '.png']);
	groundTruth = imread(['data/groundTruth/lb_', num2str(i), '.png']);
    N = sum(sum(groundTruth ~= 0));
	% res = ((rgbPred == groundTruth) | (depthPred == groundTruth));
	% rgbCorrect = [rgbCorrect,sum(sum((rgbPred == groundTruth)))/N];
	% depthCorrect = [depthCorrect,sum(sum((depthPred == groundTruth)))/N];
	% fuseCorrect = [fuseCorrect, sum(sum(res))/N];
	% imwrite(255*res, ['data/resNetPred/expAcc/fuse_', num2str(i), '.png']);
	% imwrite(255*(rgbPred == groundTruth), ['data/resNetPred/expAcc/rgb_', num2str(i), '.png']);
	% imwrite(255*(depthPred == groundTruth), ['data/resNetPred/expAcc/depth_', num2str(i), '.png']);
	res2 = ((rgbPred == groundTruth) + (depthPred == groundTruth));
	res2 = uint8(res2*127);
	imwrite(res2, ['data/resNetPred/expAcc/addFuse_', num2str(i), '.png']);

end
% rc = mean(rgbCorrect)
% dc = mean(depthCorrect)
% fc = mean(fuseCorrect)
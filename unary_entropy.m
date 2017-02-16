dbstop if error;
clc;clear;
expoPath = 'data/unary/';

res_rgb = [];
res_d = [];
for i = 1:202
	if ~exist([expoPath, 'RGB/rgb_', num2str(i), '.mat'])
		continue;
	end
	% heat map of rgb
	unaryPath = [expoPath, 'rgb/rgb_', num2str(i), '.mat'];
	prob = load(unaryPath);
	y = abs((prob.scores_).*log2(prob.scores_));
	entropy = sum(y, 3);
	res_rgb = {res_rgb, entropy};
	img = depth2rgb(entropy);
	imwrite(img, [expoPath, 'rgb/rgb_', num2str(i), '.png']);

	% heat map of depth
	unaryPath = [expoPath, 'depth/depth_', num2str(i), '.mat'];
	prob = load(unaryPath);
	y = abs((prob.scores_).*log2(prob.scores_));
	entropy = sum(y, 3);
	res_d = {res_d, entropy};
	img = depth2rgb(entropy);
	imwrite(img, [expoPath, 'depth/depth_', num2str(i), '.png']);
end




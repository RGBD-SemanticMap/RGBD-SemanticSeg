dbstop if error;

for i = 1 : 202
	% heat map of rgb
	if ~exist(['data/resNetPred/RGB/', num2str(i), '.png'])
		continue;
	end
	
	figure;
	[rgbPred, map] = imread(['data/resNetPred/RGB/', num2str(i), '.png']);
	subplot(5,3,1);
	subimage(rgbPred, map);

	[depthPred, map] = imread(['data/resNetPred/Depth/depth_', num2str(i), '.png']);
	subplot(5,3,2);
	subimage(depthPred, map);

	groundTruth = imread(['data/groundTruth/lb_', num2str(i), '.png']);
	subplot(5,3,3);
	subimage(groundTruth, map);

	[rgbHeat, mp] = imread(['data/unary/RGB/rgb_', num2str(i), '.png']);
	subplot(5,3,4);
	subimage(rgbHeat);

	depthHeat = imread(['data/unary/depth/depth_', num2str(i), '.png']);
	subplot(5,3,5);
	subimage(depthHeat);

	realimg = imread(['data/real/', num2str(i), '.png']);
	subplot(5,3,6);
	subimage(realimg);

	rgbBw = imread(['data/resNetPred/expAcc/rgb_', num2str(i), '.png']);
	subplot(5,3,7);
	subimage(rgbBw);

	depthBw = imread(['data/resNetPred/expAcc/depth_', num2str(i), '.png']);
	subplot(5,3,8);
	subimage(depthBw);

	fuseBw = imread(['data/resNetPred/expAcc/addFuse_', num2str(i), '.png']);
	subplot(5,3,9);
	subimage(fuseBw);

	rgbCold = rgbHeat .* repmat(uint8(rgbBw == 0),[1 1 3]);
	subplot(5,3,10);
	subimage(rgbCold);

	depthCold = depthHeat .* repmat(uint8(depthBw == 0),[1 1 3]);
	subplot(5,3,11);
	subimage(depthCold);
    
    set(gcf,'visible','off');
    print(gcf,'-dpng',sprintf('E:/shengchuang/matconvnet/RGB-D/data/resNetPred/show/%s.png',num2str(i)));

	pause(1);

end

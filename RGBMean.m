dep = {}
for i=1:size(depth_rgb,4)
    dep{i} = depth_rgb(:,:,:,i);
end
b = cell2mat(dep);
m = mean(b,1);
rgbMean_DEP = mean(m,2)
rgbMean_dep = [rgbMean_DEP(1,1,1) rgbMean_DEP(1,1,2) rgbMean_DEP(1,1,3)]



depths_rgb = [];
for i = 1: size(depths,3)
    tupian = depths(:,:,i);
    depth_rgb(:,:,:,i) = depth2rgb(tupian);
end



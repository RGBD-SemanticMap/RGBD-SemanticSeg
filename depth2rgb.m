function rgb_img = depth2rgb( depth )
%this function is used to change the depth image(we may not know the input
%data type)to RGB image(uint8 type)

%% check whether there are points with zero depth value
% depth=imread('apple.png');
depth=double(depth);
depth_mask=zeros(size(depth));
depth_mask(find(depth==0))=1;

%% scale the depth value to [0,255]
im_max=max(max(depth));
temp=depth(depth>0);
im_min=min(temp);
depth=(depth-im_min)/(im_max-im_min);
depth(find(depth_mask==1))=0;
depth=255*depth;
depth=uint8(depth);
% imshow(depth);
%% apply the colormap
rgb_img=ind2rgb(depth,jet(255));
rgb_img=uint8(rgb_img*255);
% figure(2);
% imshow(rgb_img);
end


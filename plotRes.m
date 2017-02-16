function plotRes()
clear all; close all;
addpath('..');

dbstop if error;
%%

% chosen = [117 118 40 137 220 310 447 516 551 561 579 802 803];
chosen = [551 802 803 220 447];
for i = 1 : numel(chosen)
    imId = chosen(i);
    base_path = 'E:\shengchuang\matconvnet\RGB-D\data\';
    imPath = [base_path 'res\x_' num2str(imId) '.png'];

    paths{1} = [base_path 'real\' num2str(imId) '.png'];    
    paths{2} = [base_path 'save_image\' num2str(imId) '.png'];
     paths{3} = [base_path 'save_depth\' num2str(imId) '.png'];
     paths{4} = [base_path 'save_decision_8s\' num2str(imId) '.png'];
      paths{5} = [base_path 'fea_8s\' num2str(imId) '.png'];
       paths{6} = [base_path 'final\' num2str(imId) '.png'];
    paths{7} = [base_path 'GroundTruth\lb_' num2str(imId) '.png'];
%     if ~exist(opts.unaryPath)
%         ['missing: ' opts.unaryPath]
%         continue;
%     end 
%     if ~exist(opts.imPath)
%         ['missing: ' opts.imPath]
%         continue;
%     end 
%     if ~exist(opts.gt_filename)
%         ['missing: ' opts.gt_filename]
%         continue;
%     end 


    %%

    %M.display()
    pic = {};result = [];
    for p = 1 : numel(paths)
        pic{p} = imread(paths{p});
        if(p ~= 1)
            cmap = labelColors();
            pic{p} = reshape(255*cmap(pic{p}(:)+1,:), [480 640 3]);
        end
        result(1:480, (p-1)*640+1 +5*(p-1):p*640 +5*(p-1), 1:3) = pic{p};
    end  
    imwrite(uint8(result),imPath,'png');
end

function displayImage(pred)
% -------------------------------------------------------------------------
cmap = labelColors() ;
image(uint8(pred)) ;
axis image ;
title('predicted') ;

colormap(cmap) ;

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N = 41;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
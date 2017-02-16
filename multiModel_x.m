function net = multiModel_x(varargin)
dbstop if error;
% run matconvnet/matlab/vl_setupnn ;
opts.sourceModelPath_image = 'data/NYU/image_net.mat' ;
opts.sourceModelPath_depth = 'data/NYU/depth_net.mat' ;
opts = vl_argparse(opts, varargin) ;

netStruct = load(opts.sourceModelPath_image) ;
net_rgb = dagnn.DagNN.loadobj(netStruct.net) ;
clear netStruct ;

netStruct = load(opts.sourceModelPath_depth) ;
net_d = dagnn.DagNN.loadobj(netStruct.net) ;
clear netStruct ;

removeNames = {net_rgb.layers([39:40]).name};
for i = 1 : numel(removeNames)
    net_rgb.removeLayer(removeNames{i});
    net_d.removeLayer(removeNames{i});
end

net_rgb.layers(end).outputs = {'fuse_rgb'};
net_d.layers(end).outputs = {'fuse_d'};

net = dagnn.DagNN();
sliceBlock1 = dagnn.Slice('sta',1,'terminus',3);
sliceBlock2 = dagnn.Slice('sta',4,'terminus',6);
net.addLayer('slice1', sliceBlock1, {'input'}, {'rgb_input'}, {});
net.addLayer('slice2', sliceBlock2, {'input'}, {'d_input'}, {});

for i = 1 : numel(net_rgb.layers)
    % to be implenment
    if ~isempty(net_rgb.layers(i).params)
        p = strcat(repmat({'rgb_'}, [1 numel(net_rgb.layers(i).params)]), net_rgb.layers(i).params);
    else
        p = {};
    end
    net.addLayer(['rgb_',net_rgb.layers(i).name], net_rgb.layers(i).block, ...
        strcat(repmat({'rgb_'}, [1 numel(net_rgb.layers(i).inputs)]), net_rgb.layers(i).inputs), ...
        strcat(repmat({'rgb_'}, [1 numel(net_rgb.layers(i).outputs)]), net_rgb.layers(i).outputs), ...
        p);
    if ~isempty(net.layers(end).paramIndexes)
        net.params(net.layers(end).paramIndexes) = net_rgb.getParam(net_rgb.layers(i).params);
        [net.params([net.layers(end).paramIndexes]).name] = deal(net.layers(end).params{:});
    end
    
    if ~isempty(net_d.layers(i).params)
        p = strcat(repmat({'d_'}, [1 numel(net_d.layers(i).params)]), net_d.layers(i).params);
    else
        p = {};
    end
    net.addLayer(['d_',net_d.layers(i).name], net_d.layers(i).block, ...
        strcat(repmat({'d_'}, [1 numel(net_d.layers(i).inputs)]), net_d.layers(i).inputs), ...
        strcat(repmat({'d_'}, [1 numel(net_d.layers(i).outputs)]), net_d.layers(i).outputs), ...
        p);
    
    if ~isempty(net.layers(end).paramIndexes)
        net.params(net.layers(end).paramIndexes) = net_d.getParam(net_d.layers(i).params);
        [net.params([net.layers(end).paramIndexes]).name] = deal(net.layers(end).params{:});
    end
    
end

for i = 1 : numel(net.params)
    net.params(i).learningRate = 0;
end

%% adding fusing layers
fuseblock = dagnn.Concat('dim', 3);
net.addLayer('fuse', fuseblock, {'rgb_fuse_rgb','d_fuse_d'}, {'fusion_1'}, {});

fuse_conv1_block = dagnn.Conv('size', [1 1 80 40], 'hasBias', true);
net.addLayer('fc1_fus', fuse_conv1_block, {'fusion_1'}, {'fusion_2'}, {'filters_f1', 'biases_f1'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 80 40];
  else
    sz = [1 40];
  end
  net.params(p).value = single(normrnd(0, 0.05, sz));
end
net.params(p).value = ones(sz, 'single');

fuse_relu1_block = dagnn.ReLU();
net.addLayer('relu1_fus', fuse_relu1_block, {'fusion_2'}, {'fusion_3'}, {});

% drop out layer for fusion
fuse_drop1_block = dagnn.DropOut('rate', 0.5);
net.addLayer('drop1_fus', fuse_drop1_block, {'fusion_3'}, {'prediction'}, {});

% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;


end

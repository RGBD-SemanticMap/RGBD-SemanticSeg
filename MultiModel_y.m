function net = MultiModel_y(varargin)

dbstop if error;
%run matconvnet/matlab/vl_setupnn ;
opts.sourceModelPath_image = 'data/NYU/image_net.mat' ;
opts.sourceModelPath_depth = 'data/NYU/depth_net.mat' ;
opts = vl_argparse(opts, varargin) ;

net1p = load(opts.sourceModelPath_image) ;
net2p = load(opts.sourceModelPath_depth) ;
net1p = net1p.net;
net2p = net2p.net;
% net1.removeLayer('accuracy');
% net1.removeLayer({'objective','accuracy','deconv32','fc8'});

% net1p = net1;
%% 把fcn8s转成dagnn的标准格式
layer = {'conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2',...
                 'conv3_1','conv3_2','conv3_3','pool3','conv4_1','conv4_2',...
                 'conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5',...
                 'fc6','fc7','dropout1'};
layersname = {net1p.layers.name};
layersparam = {net1p.params.name};
net1.layers = {};             
for k = 1: numel(layer)
    f_s = layer{k}(1);
    switch f_s
        case 'c'
            idx = find(strcmp(layer{k},layersname));
            f_idx = find(strcmp(net1p.layers(idx).params{1},layersparam));
            b_idx = find(strcmp(net1p.layers(idx).params{2},layersparam));
            net1.layers{end+1} = struct('type','conv',...
                           'filters',net1p.params(f_idx).value,...
                           'biases',net1p.params(b_idx).value',...
                           'stride',net1p.layers(idx).block.stride,...
                           'pad',net1p.layers(idx).block.pad);
        case 'p'
            idx = find(strcmp(layer{k},layersname));
            net1.layers{end+1} = struct('type','pool',...
                           'method',net1p.layers(idx).block.method,...
                           'pool',net1p.layers(idx).block.poolSize,...
                           'stride',net1p.layers(idx).block.stride,...
                           'pad',net1p.layers(idx).block.pad);
        case 'f'
            idx = find(strcmp(layer{k},layersname));
            f_idx = find(strcmp(net1p.layers(idx).params{1},layersparam));
            b_idx = find(strcmp(net1p.layers(idx).params{2},layersparam));
            net1.layers{end+1} = struct('type','conv',...
                           'filters',net1p.params(f_idx).value,...
                           'biases',net1p.params(b_idx).value',...
                           'stride',net1p.layers(idx).block.stride,...
                           'pad',net1p.layers(idx).block.pad);
        case 'd'
            idx = find(strcmp(layer{k},layersname));
            net1.layers{end+1} = struct('type','dropout',...
                           'rate',net1p.layers(idx).block.rate,...
                           'frozen',net1p.layers(idx).block.frozen);
    end
end
net1 = vl_simplenn_tidy(net1);
net1 = dagnn.DagNN.fromSimpleNN(net1) ;

for k = 1:numel(net1.layers)
    net1.renameLayer(net1.layers(k).name,layer{k});
end



layersname = {net2p.layers.name};
layersparam = {net2p.params.name};
net2.layers = {};             
for k = 1: numel(layer)
    f_s = layer{k}(1);
    switch f_s
        case 'c'
            idx = find(strcmp(layer{k},layersname));
            f_idx = find(strcmp(net2p.layers(idx).params{1},layersparam));
            b_idx = find(strcmp(net2p.layers(idx).params{2},layersparam));
            net2.layers{end+1} = struct('type','conv',...
                           'filters',net2p.params(f_idx).value,...
                           'biases',net2p.params(b_idx).value',...
                           'stride',net2p.layers(idx).block.stride,...
                           'pad',net2p.layers(idx).block.pad);
        case 'p'
            idx = find(strcmp(layer{k},layersname));
            net2.layers{end+1} = struct('type','pool',...
                           'method',net2p.layers(idx).block.method,...
                           'pool',net2p.layers(idx).block.poolSize,...
                           'stride',net2p.layers(idx).block.stride,...
                           'pad',net2p.layers(idx).block.pad);
        case 'f'
            idx = find(strcmp(layer{k},layersname));
            f_idx = find(strcmp(net2p.layers(idx).params{1},layersparam));
            b_idx = find(strcmp(net2p.layers(idx).params{2},layersparam));
            net2.layers{end+1} = struct('type','conv',...
                           'filters',net2p.params(f_idx).value,...
                           'biases',net2p.params(b_idx).value',...
                           'stride',net2p.layers(idx).block.stride,...
                           'pad',net2p.layers(idx).block.pad);
        case 'd'
            idx = find(strcmp(layer{k},layersname));
            net2.layers{end+1} = struct('type','dropout',...
                           'rate',net2p.layers(idx).block.rate,...
                           'frozen',net2p.layers(idx).block.frozen);
    end
end
net2 = vl_simplenn_tidy(net2);
net2 = dagnn.DagNN.fromSimpleNN(net2) ;

for k = 1:numel(net2.layers)
    net2.renameLayer(net2.layers(k).name,layer{k});
end

%% =================================================================
net = dagnn.DagNN();

% Slice Layer
sliceBlock1 = dagnn.Slice('sta',1,'terminus',3);
sliceBlock2 = dagnn.Slice('sta',4,'terminus',6);

net.addLayer('slice1', sliceBlock1, {'input'}, {'rgb1'}, {});
net.addLayer('slice2', sliceBlock2, {'input'}, {'d1'}, {});

% Conv1_1 Layer
idx = net1.getLayerIndex('conv1_1');
net.addLayer('conv11_rgb',net1.layers(idx).block,{'rgb1'},{'rgb2'},{'conv11f_rgb','conv11b_rgb'});
net.addLayer('conv11_d',net2.layers(idx).block,{'d1'},{'d2'},{'conv11f_d','conv11b_d'});

f1 = net.getParamIndex('conv11f_rgb');
f2 = net.getParamIndex('conv11b_rgb');
f3 = net.getParamIndex('conv11f_d');
f4 = net.getParamIndex('conv11b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv11f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv11b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv11f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv11b_d';

% Relu1_1 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu11_rgb', reluBlock_rgb, {'rgb2'}, {'rgb3'}, {}) ;
net.addLayer('relu11_d', reluBlock_d, {'d2'}, {'d3'}, {}) ;

% Conv1_2 Layer
idx = net1.getLayerIndex('conv1_2');
net.addLayer('conv12_rgb',net1.layers(idx).block,{'rgb3'},{'rgb4'},{'conv12f_rgb','conv12b_rgb'});
net.addLayer('conv12_d',net2.layers(idx).block,{'d3'},{'d4'},{'conv12f_d','conv12b_d'});

f1 = net.getParamIndex('conv12f_rgb');
f2 = net.getParamIndex('conv12b_rgb');
f3 = net.getParamIndex('conv12f_d');
f4 = net.getParamIndex('conv12b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv12f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv12b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv12f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv12b_d';

% Relu1_2 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu12_rgb', reluBlock_rgb, {'rgb4'}, {'rgb5'}, {}) ;
net.addLayer('relu12_d', reluBlock_d, {'d4'}, {'d5'}, {}) ;

% Pooling_1 Layer
idx = net1.getLayerIndex('pool1');
net.addLayer('pool1_rgb',net1.layers(idx).block,{'rgb5'},{'rgb6'},{});
net.addLayer('pool1_d',net2.layers(idx).block,{'d5'},{'d6'},{});

% Conv2_1 Layer
idx = net1.getLayerIndex('conv2_1');
net.addLayer('conv21_rgb',net1.layers(idx).block,{'rgb6'},{'rgb7'},{'conv21f_rgb','conv21b_rgb'});
net.addLayer('conv21_d',net2.layers(idx).block,{'d6'},{'d7'},{'conv21f_d','conv21b_d'});

f1 = net.getParamIndex('conv21f_rgb');
f2 = net.getParamIndex('conv21b_rgb');
f3 = net.getParamIndex('conv21f_d');
f4 = net.getParamIndex('conv21b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv21f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv21b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv21f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv21b_d';

% Relu2_1 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu21_rgb', reluBlock_rgb, {'rgb7'}, {'rgb8'}, {}) ;
net.addLayer('relu21_d', reluBlock_d, {'d7'}, {'d8'}, {}) ;

% Conv2_2 Layer
idx = net1.getLayerIndex('conv2_2');
net.addLayer('conv22_rgb',net1.layers(idx).block,{'rgb8'},{'rgb10'},{'conv22f_rgb','conv22b_rgb'});
net.addLayer('conv22_d',net1.layers(idx).block,{'d8'},{'d10'},{'conv22f_d','conv22b_d'});

f1 = net.getParamIndex('conv22f_rgb');
f2 = net.getParamIndex('conv22b_rgb');
f3 = net.getParamIndex('conv22f_d');
f4 = net.getParamIndex('conv22b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv22f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv22b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv22f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv22b_d';

% Relu2_2 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu22_rgb', reluBlock_rgb, {'rgb10'}, {'rgb11'}, {}) ;
net.addLayer('relu22_d', reluBlock_d, {'d10'}, {'d11'}, {}) ;

% Pooling_2 Layer
idx = net1.getLayerIndex('pool2');
net.addLayer('pool2_rgb',net1.layers(idx).block,{'rgb11'},{'rgb12'},{});
net.addLayer('pool2_d',net2.layers(idx).block,{'d11'},{'d12'},{});

% Conv3_1 Layer
idx = net1.getLayerIndex('conv3_1');
net.addLayer('conv31_rgb',net1.layers(idx).block,{'rgb12'},{'rgb13'},{'conv31f_rgb','conv31b_rgb'});
net.addLayer('conv31_d',net2.layers(idx).block,{'d12'},{'d13'},{'conv31f_d','conv31b_d'});

f1 = net.getParamIndex('conv31f_rgb');
f2 = net.getParamIndex('conv31b_rgb');
f3 = net.getParamIndex('conv31f_d');
f4 = net.getParamIndex('conv31b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv31f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv31b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv31f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv31b_d';

% Relu3_1 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu31_rgb', reluBlock_rgb, {'rgb13'}, {'rgb14'}, {}) ;
net.addLayer('relu31_d', reluBlock_d, {'d13'}, {'d14'}, {}) ;

% Conv3_2 Layer
idx = net1.getLayerIndex('conv3_2');
net.addLayer('conv32_rgb',net1.layers(idx).block,{'rgb14'},{'rgb15'},{'conv32f_rgb','conv32b_rgb'});
net.addLayer('conv32_d',net2.layers(idx).block,{'d14'},{'d15'},{'conv32f_d','conv32b_d'});

f1 = net.getParamIndex('conv32f_rgb');
f2 = net.getParamIndex('conv32b_rgb');
f3 = net.getParamIndex('conv32f_d');
f4 = net.getParamIndex('conv32b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv32f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv32b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv32f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv32b_d';

% Relu3_2 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu32_rgb', reluBlock_rgb, {'rgb15'}, {'rgb16'}, {}) ;
net.addLayer('relu32_d', reluBlock_d, {'d15'}, {'d16'}, {}) ;

% Conv3_3 Layer
idx = net1.getLayerIndex('conv3_3');
net.addLayer('conv33_rgb',net1.layers(idx).block,{'rgb16'},{'rgb17'},{'conv33f_rgb','conv33b_rgb'});
net.addLayer('conv33_d',net2.layers(idx).block,{'d16'},{'d17'},{'conv33f_d','conv33b_d'});

f1 = net.getParamIndex('conv33f_rgb');
f2 = net.getParamIndex('conv33b_rgb');
f3 = net.getParamIndex('conv33f_d');
f4 = net.getParamIndex('conv33b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv33f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv33b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv33f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv33b_d';

% Relu3_3 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu33_rgb', reluBlock_rgb, {'rgb17'}, {'rgb18'}, {});
net.addLayer('relu33_d', reluBlock_d, {'d17'}, {'d18'}, {});

% Pooling_3 Layer
idx = net1.getLayerIndex('pool3');
net.addLayer('pool3_rgb',net1.layers(idx).block,{'rgb18'},{'rgb19'},{});
net.addLayer('pool3_d',net2.layers(idx).block,{'d18'},{'d19'},{});

% Conv4_1 Layer
idx = net1.getLayerIndex('conv4_1');
net.addLayer('conv41_rgb',net1.layers(idx).block,{'rgb19'},{'rgb20'},{'conv41f_rgb','conv41b_rgb'});
net.addLayer('conv41_d',net2.layers(idx).block,{'d19'},{'d20'},{'conv41f_d','conv41b_d'});

f1 = net.getParamIndex('conv41f_rgb');
f2 = net.getParamIndex('conv41b_rgb');
f3 = net.getParamIndex('conv41f_d');
f4 = net.getParamIndex('conv41b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv41f_rgb';
net.params(f2) = f_d(2);
net.params(f2).name = 'conv41b_rgb';
net.params(f3) = f_rgb(1);
net.params(f3).name = 'conv41f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv41b_d';

% Relu4_1 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu41_rgb', reluBlock_rgb, {'rgb20'}, {'rgb21'}, {});
net.addLayer('relu41_d', reluBlock_d, {'d20'}, {'d21'}, {});

% Conv4_2 Layer
idx = net1.getLayerIndex('conv4_2');
net.addLayer('conv42_rgb',net1.layers(idx).block,{'rgb21'},{'rgb22'},{'conv42f_rgb','conv42b_rgb'});
net.addLayer('conv42_d',net2.layers(idx).block,{'d21'},{'d22'},{'conv42f_d','conv42b_d'});

f1 = net.getParamIndex('conv42f_rgb');
f2 = net.getParamIndex('conv42b_rgb');
f3 = net.getParamIndex('conv42f_d');
f4 = net.getParamIndex('conv42b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv42f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv42b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv42f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv42b_d';

% Relu4_2 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu42_rgb', reluBlock_rgb, {'rgb22'}, {'rgb23'}, {});
net.addLayer('relu42_d', reluBlock_d, {'d22'}, {'d23'}, {});

% Conv4_3 Layer
idx = net1.getLayerIndex('conv4_3');
net.addLayer('conv43_rgb',net1.layers(idx).block,{'rgb23'},{'rgb24'},{'conv43f_rgb','conv43b_rgb'});
net.addLayer('conv43_d',net2.layers(idx).block,{'d23'},{'d24'},{'conv43f_d','conv43b_d'});

f1 = net.getParamIndex('conv43f_rgb');
f2 = net.getParamIndex('conv43b_rgb');
f3 = net.getParamIndex('conv43f_d');
f4 = net.getParamIndex('conv43b_d');


f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv43f_rgb';
net.params(f2) = f_d(2);
net.params(f2).name = 'conv43b_rgb';
net.params(f3) = f_rgb(1);
net.params(f3).name = 'conv43f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv43b_d';

% Relu4_3 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu43_rgb', reluBlock_rgb, {'rgb24'}, {'rgb25'}, {});
net.addLayer('relu43_d', reluBlock_d, {'d24'}, {'d25'}, {});

% Pooling_4 Layer
idx = net1.getLayerIndex('pool4');
net.addLayer('pool4_rgb',net1.layers(idx).block,{'rgb25'},{'rgb26'},{});
net.addLayer('pool4_d',net2.layers(idx).block,{'d25'},{'d26'},{});

% Conv5_1 Layer
idx = net1.getLayerIndex('conv5_1');
net.addLayer('conv51_rgb',net1.layers(idx).block,{'rgb26'},{'rgb27'},{'conv51f_rgb','conv51b_rgb'});
net.addLayer('conv51_d',net2.layers(idx).block,{'d26'},{'d27'},{'conv51f_d','conv51b_d'});

f1 = net.getParamIndex('conv51f_rgb');
f2 = net.getParamIndex('conv51b_rgb');
f3 = net.getParamIndex('conv51f_d');
f4 = net.getParamIndex('conv51b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv51f_rgb';
net.params(f2) = f_d(2);
net.params(f2).name = 'conv51b_rgb';
net.params(f3) = f_rgb(1);
net.params(f3).name = 'conv51f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv51b_d';

% Relu5_1 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu51_rgb', reluBlock_rgb, {'rgb27'}, {'rgb28'}, {});
net.addLayer('relu51_d', reluBlock_d, {'d27'}, {'d28'}, {});

% Conv5_2 Layer
idx = net1.getLayerIndex('conv5_2');
net.addLayer('conv52_rgb',net1.layers(idx).block,{'rgb28'},{'rgb29'},{'conv52f_rgb','conv52b_rgb'});
net.addLayer('conv52_d',net2.layers(idx).block,{'d28'},{'d29'},{'conv52f_d','conv52b_d'});

f1 = net.getParamIndex('conv52f_rgb');
f2 = net.getParamIndex('conv52b_rgb');
f3 = net.getParamIndex('conv52f_d');
f4 = net.getParamIndex('conv52b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv52f_rgb';
net.params(f2) = f_d(2);
net.params(f2).name = 'conv52b_rgb';
net.params(f3) = f_rgb(1);
net.params(f3).name = 'conv52f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv52b_d';

% Relu5_2 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu52_rgb', reluBlock_rgb, {'rgb29'}, {'rgb30'}, {});
net.addLayer('relu52_d', reluBlock_d, {'d29'}, {'d30'}, {});

% Conv5_3 Layer
idx = net1.getLayerIndex('conv5_3');
net.addLayer('conv53_rgb',net1.layers(idx).block,{'rgb30'},{'rgb31'},{'conv53f_rgb','conv53b_rgb'});
net.addLayer('conv53_d',net2.layers(idx).block,{'d30'},{'d31'},{'conv53f_d','conv53b_d'});

f1 = net.getParamIndex('conv53f_rgb');
f2 = net.getParamIndex('conv53b_rgb');
f3 = net.getParamIndex('conv53f_d');
f4 = net.getParamIndex('conv53b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv53f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv53b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv53f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv53b_d';

% Relu5_3 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu53_rgb', reluBlock_rgb, {'rgb31'}, {'rgb32'}, {});
net.addLayer('relu53_d', reluBlock_d, {'d31'}, {'d32'}, {});

% Pooling_5 Layer
idx = net1.getLayerIndex('pool5');
net.addLayer('pool5_rgb',net1.layers(idx).block,{'rgb32'},{'rgb33'},{});
net.addLayer('pool5_d',net2.layers(idx).block,{'d32'},{'d33'},{});

% fc6
idx = net1.getLayerIndex('fc6');
net.addLayer('fc6_rgb',net1.layers(idx).block,{'rgb33'},{'rgb34'},{'fc6f_rgb','fc6b_rgb'});
net.addLayer('fc6_d',net2.layers(idx).block,{'d33'},{'d34'},{'fc6f_d','fc6b_d'});

f1 = net.getParamIndex('fc6f_rgb');
f2 = net.getParamIndex('fc6b_rgb');
f3 = net.getParamIndex('fc6f_d');
f4 = net.getParamIndex('fc6b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'fc6f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'fc6b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'fc6f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'fc6b_d';

% Relu6 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu6_rgb', reluBlock_rgb, {'rgb34'}, {'rgb35'}, {});
net.addLayer('relu6_d', reluBlock_d, {'d34'}, {'d35'}, {});

idx = net1.getLayerIndex('dropout1');
net.addLayer('dropout_rgb', net1.layers(idx).block,{'rgb35'},{'rgb35a'}, {});
net.addLayer('dropout_d', net2.layers(idx).block,{'d35'},{'d35a'}, {});

% fc7
idx = net1.getLayerIndex('fc7');
net.addLayer('fc7_rgb',net1.layers(idx).block,{'rgb35a'},{'rgb36'},{'fc7f_rgb','fc7b_rgb'});
net.addLayer('fc7_d',net1.layers(idx).block,{'d35a'},{'d36'},{'fc7f_d','fc7b_d'});

f1 = net.getParamIndex('fc7f_rgb');
f2 = net.getParamIndex('fc7b_rgb');
f3 = net.getParamIndex('fc7f_d');
f4 = net.getParamIndex('fc7b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'fc7f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'fc7b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'fc7f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'fc7b_d';

% Relu7 Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('relu7_rgb', reluBlock_rgb, {'rgb36'}, {'fc7_1'}, {});
net.addLayer('relu7_d', reluBlock_d, {'d36'}, {'fc7_2'}, {});

% fuse the RGB and D layer, combine them by channels.
fuseblock = dagnn.Concat('dim', 3);
net.addLayer('fuse', fuseblock, {'fc7_1','fc7_2'}, {'fusion_1'}, {});

% first conv layer after fusion
fuse_conv1_block = dagnn.Conv('size', [1 1 8192 4096], 'hasBias', true);
net.addLayer('fc1_fus', fuse_conv1_block, {'fusion_1'}, {'fusion_2'}, {'filters_f1', 'biases_f1'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 8192 4096];
  else
    sz = [1 4096];
  end
  net.params(p).value = zeros(sz, 'single');
end
net.params(p).value = ones(sz, 'single');

fuse_relu1_block = dagnn.ReLU();
net.addLayer('relu1_fus', fuse_relu1_block, {'fusion_2'}, {'fusion_3'}, {});

% drop out layer for fusion
fuse_drop1_block = dagnn.DropOut('rate', 0.5);
net.addLayer('drop1_fus', fuse_drop1_block, {'fusion_3'}, {'fusion_4'}, {});

% second conv layer, according to fcn model.
fuse_conv2_block = dagnn.Conv('size', [1 1 4096 40], 'hasBias', true);
net.addLayer('fc2_fus', fuse_conv2_block, {'fusion_4'}, {'fusion_5'}, {'filters_f2', 'biases_f2'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 4096 40];
  else
    sz = [1 40];
  end
  net.params(p).value = zeros(sz, 'single');
end
net.params(p).value = ones(sz, 'single');


fuse_filters0 = single(bilinear_u(6, 1, 40)) ;
net.addLayer('deconv1', ...
  dagnn.ConvTranspose('size', size(fuse_filters0), ...
                      'upsample', 4, ...
                      'crop', 1, ...
                      'hasBias', false), ...
             'fusion_5', 'fusion_6', 'deconv1f') ;
f = net.getParamIndex('deconv1f') ;
net.params(f).value = fuse_filters0 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;


fuse_filters = single(bilinear_u(16, 40, 40)) ;
fuse_deconv_block = dagnn.ConvTranspose(...
  'size', size(fuse_filters), ...
  'upsample', 8, ...
  'crop', 4, ...
  'numGroups', 40, ...
  'hasBias', false, ...
    'opts', {});                   %这里原来是net.meta.cudnnOpts
net.addLayer('deconv2', fuse_deconv_block, 'fusion_6', 'prediction', 'deconvf');

f = net.getParamIndex('deconvf') ;
net.params(f).value = fuse_filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

for i = 1:52
    net.params(i).learningRate = 0;
end

% net.initParams() ;

function net = multimodel_resnet(varargin)

dbstop if error;
%run matconvnet/matlab/vl_setupnn ;
opts.sourceModelPath_image = 'data/NYU/result/RGB/net-epoch-12.mat' ;
opts.sourceModelPath_depth = 'data/NYU/result/Depth/net-epoch-24.mat' ;
opts = vl_argparse(opts, varargin) ;

net1p = load(opts.sourceModelPath_image) ;
net2p = load(opts.sourceModelPath_depth) ;
net1 = dagnn.DagNN.loadobj(net1p.net);
net2 = dagnn.DagNN.loadobj(net2p.net);

net = dagnn.DagNN();

% Slice Layer
sliceBlock1 = dagnn.Slice('sta',1,'terminus',3);
sliceBlock2 = dagnn.Slice('sta',4,'terminus',6);

net.addLayer('slice1', sliceBlock1, {'input'}, {'rgb1'}, {});
net.addLayer('slice2', sliceBlock2, {'input'}, {'d1'}, {});

% Conv1_1 Layer
idx = net1.getLayerIndex('conv1');
net.addLayer('conv1_rgb',net1.layers(idx).block,{'rgb1'},{'conv1_rgb'},{'conv1f_rgb','conv1b_rgb'});
net.addLayer('conv1_d',net2.layers(idx).block,{'d1'},{'conv1_d'},{'conv1f_d','conv1b_d'});

f1 = net.getParamIndex('conv1f_rgb');
f2 = net.getParamIndex('conv1b_rgb');
f3 = net.getParamIndex('conv1f_d');
f4 = net.getParamIndex('conv1b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'conv1f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'conv1b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'conv1f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'conv1b_d';

% bn_conv1 layer
idx = net1.getLayerIndex('bn_conv1');
net.addLayer('bn_conv1_rgb',net1.layers(idx).block,{'conv1_rgb'},{'conv1x_rgb'},{'bn_conv1_mult_rgb','bn_conv1_bias_rgb','bn_conv1_moments_rgb'});
net.addLayer('bn_conv1_d',net2.layers(idx).block,{'conv1_d'},{'conv1x_d'},{'bn_conv1_mult_d','bn_conv1_bias_d','bn_conv1_moments_d'});

f1 = net.getParamIndex('bn_conv1_mult_rgb');
f2 = net.getParamIndex('bn_conv1_bias_rgb');
f3 = net.getParamIndex('bn_conv1_moments_rgb');
f4 = net.getParamIndex('bn_conv1_mult_d');
f5 = net.getParamIndex('bn_conv1_bias_d');
f6 = net.getParamIndex('bn_conv1_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn_conv1_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn_conv1_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn_conv1_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn_conv1_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn_conv1_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn_conv1_moments_d';


% conv1_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('conv1_relu_rgb', reluBlock_rgb, {'conv1x_rgb'}, {'conv1xxx_rgb'}, {}) ;
net.addLayer('conv1_relu_d', reluBlock_d, {'conv1x_d'}, {'conv1xxx_d'}, {}) ;

% Pool1 layer
idx = net1.getLayerIndex('pool1');
net.addLayer('pool1_rgb',net1.layers(idx).block,{'conv1xxx_rgb'},{'pool1_rgb'},{});
net.addLayer('pool1_d',net2.layers(idx).block,{'conv1xxx_d'},{'pool1_d'},{});


% res2a_branch1 layer
idx = net1.getLayerIndex('res2a_branch1');
net.addLayer('res2a_branch1_rgb',net1.layers(idx).block,{'pool1_rgb'},{'res2a_branch1_rgb'},{'res2a_branch1_filter_rgb'});
net.addLayer('res2a_branch1_d',net2.layers(idx).block,{'pool1_d'},{'res2a_branch1_d'},{'res2a_branch1_filter_d'});

f1 = net.getParamIndex('res2a_branch1_filter_rgb');
f2 = net.getParamIndex('res2a_branch1_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2a_branch1_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2a_branch1_filter_d';

% bn2a_branch1 Layer
idx = net1.getLayerIndex('bn2a_branch1');
net.addLayer('bn2a_branch1_rgb',net1.layers(idx).block,{'res2a_branch1_rgb'},{'res2a_branch1x_rgb'},{'bn2a_branch1_mult_rgb','bn2a_branch1_bias_rgb','bn2a_branch1_moments_rgb'});
net.addLayer('bn2a_branch1_d',net2.layers(idx).block,{'res2a_branch1_d'},{'res2a_branch1x_d'},{'bn2a_branch1_mult_d','bn2a_branch1_bias_d','bn2a_branch1_moments_d'});


f1 = net.getParamIndex('bn2a_branch1_mult_rgb');
f2 = net.getParamIndex('bn2a_branch1_bias_rgb');
f3 = net.getParamIndex('bn2a_branch1_moments_rgb');
f4 = net.getParamIndex('bn2a_branch1_mult_d');
f5 = net.getParamIndex('bn2a_branch1_bias_d');
f6 = net.getParamIndex('bn2a_branch1_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2a_branch1_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2a_branch1_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2a_branch1_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2a_branch1_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2a_branch1_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2a_branch1_moments_d';

% res2a_branch2a layer
idx = net1.getLayerIndex('res2a_branch2a');
net.addLayer('res2a_branch2a_rgb',net1.layers(idx).block,{'pool1_rgb'},{'res2a_branch2a_rgb'},{'res2a_branch2a_filter_rgb'});
net.addLayer('res2a_branch2a_d',net2.layers(idx).block,{'pool1_d'},{'res2a_branch2a_d'},{'res2a_branch2a_filter_d'});

f1 = net.getParamIndex('res2a_branch2a_filter_rgb');
f2 = net.getParamIndex('res2a_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2a_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2a_branch2a_filter_d';

% bn2a_branch2a layer
idx = net1.getLayerIndex('bn2a_branch2a');
net.addLayer('bn2a_branch2a_rgb',net1.layers(idx).block,{'res2a_branch2a_rgb'},{'res2a_branch2ax_rgb'},{'bn2a_branch2a_mult_rgb','bn2a_branch2a_bias_rgb','bn2a_branch2a_moments_rgb'});
net.addLayer('bn2a_branch2a_d',net2.layers(idx).block,{'res2a_branch2a_d'},{'res2a_branch2ax_d'},{'bn2a_branch2a_mult_d','bn2a_branch2a_bias_d','bn2a_branch2a_moments_d'});


f1 = net.getParamIndex('bn2a_branch2a_mult_rgb');
f2 = net.getParamIndex('bn2a_branch2a_bias_rgb');
f3 = net.getParamIndex('bn2a_branch2a_moments_rgb');
f4 = net.getParamIndex('bn2a_branch2a_mult_d');
f5 = net.getParamIndex('bn2a_branch2a_bias_d');
f6 = net.getParamIndex('bn2a_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2a_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2a_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2a_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2a_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2a_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2a_branch2a_moments_d';

% res2a_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2a_branch2a_relu_rgb', reluBlock_rgb, {'res2a_branch2ax_rgb'}, {'res2a_branch2axxx_rgb'}, {}) ;
net.addLayer('res2a_branch2a_relu_d', reluBlock_d, {'res2a_branch2ax_d'}, {'res2a_branch2axxx_d'}, {}) ;

% res2a_branch2b Layer
idx = net1.getLayerIndex('res2a_branch2b');
net.addLayer('res2a_branch2b_rgb',net1.layers(idx).block,{'res2a_branch2axxx_rgb'},{'res2a_branch2b_rgb'},{'res2a_branch2b_filter_rgb'});
net.addLayer('res2a_branch2b_d',net2.layers(idx).block,{'res2a_branch2axxx_d'},{'res2a_branch2b_d'},{'res2a_branch2b_filter_d'});

f1 = net.getParamIndex('res2a_branch2b_filter_rgb');
f2 = net.getParamIndex('res2a_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2a_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2a_branch2b_filter_d';

% bn2a_branch2b layer
idx = net1.getLayerIndex('bn2a_branch2b');
net.addLayer('bn2a_branch2b_rgb',net1.layers(idx).block,{'res2a_branch2b_rgb'},{'res2a_branch2bx_rgb'},{'bn2a_branch2b_mult_rgb','bn2a_branch2b_bias_rgb','bn2a_branch2b_moments_rgb'});
net.addLayer('bn2a_branch2b_d',net2.layers(idx).block,{'res2a_branch2b_d'},{'res2a_branch2bx_d'},{'bn2a_branch2b_mult_d','bn2a_branch2b_bias_d','bn2a_branch2b_moments_d'});


f1 = net.getParamIndex('bn2a_branch2b_mult_rgb');
f2 = net.getParamIndex('bn2a_branch2b_bias_rgb');
f3 = net.getParamIndex('bn2a_branch2b_moments_rgb');
f4 = net.getParamIndex('bn2a_branch2b_mult_d');
f5 = net.getParamIndex('bn2a_branch2b_bias_d');
f6 = net.getParamIndex('bn2a_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2a_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2a_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2a_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2a_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2a_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2a_branch2b_moments_d';

% res2a_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2a_branch2b_relu_rgb', reluBlock_rgb, {'res2a_branch2bx_rgb'}, {'res2a_branch2bxxx_rgb'}, {}) ;
net.addLayer('res2a_branch2b_relu_d', reluBlock_d, {'res2a_branch2bx_d'}, {'res2a_branch2bxxx_d'}, {}) ;

% res2a_branch2c Layer
idx = net1.getLayerIndex('res2a_branch2c');
net.addLayer('res2a_branch2c_rgb',net1.layers(idx).block,{'res2a_branch2bxxx_rgb'},{'res2a_branch2c_rgb'},{'res2a_branch2c_filter_rgb'});
net.addLayer('res2a_branch2c_d',net2.layers(idx).block,{'res2a_branch2bxxx_d'},{'res2a_branch2c_d'},{'res2a_branch2c_filter_d'});

f1 = net.getParamIndex('res2a_branch2c_filter_rgb');
f2 = net.getParamIndex('res2a_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2a_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2a_branch2c_filter_d';

% bn2a_branch2c layer
idx = net1.getLayerIndex('bn2a_branch2c');
net.addLayer('bn2a_branch2c_rgb',net1.layers(idx).block,{'res2a_branch2c_rgb'},{'res2a_branch2cx_rgb'},{'bn2a_branch2c_mult_rgb','bn2a_branch2c_bias_rgb','bn2a_branch2c_moments_rgb'});
net.addLayer('bn2a_branch2c_d',net2.layers(idx).block,{'res2a_branch2c_d'},{'res2a_branch2cx_d'},{'bn2a_branch2c_mult_d','bn2a_branch2c_bias_d','bn2a_branch2c_moments_d'});


f1 = net.getParamIndex('bn2a_branch2c_mult_rgb');
f2 = net.getParamIndex('bn2a_branch2c_bias_rgb');
f3 = net.getParamIndex('bn2a_branch2c_moments_rgb');
f4 = net.getParamIndex('bn2a_branch2c_mult_d');
f5 = net.getParamIndex('bn2a_branch2c_bias_d');
f6 = net.getParamIndex('bn2a_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2a_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2a_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2a_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2a_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2a_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2a_branch2c_moments_d';

% res2a layer(sum)
idx = net1.getLayerIndex('res2a');
net.addLayer('res2a_rgb',net1.layers(idx).block,{'res2a_branch1x_rgb','res2a_branch2cx_rgb'},{'res2a_rgb'},{});
net.addLayer('res2a_d',net2.layers(idx).block,{'res2a_branch1x_d','res2a_branch2cx_d'},{'res2a_d'},{});

% res2a_relu layer
idx = net1.getLayerIndex('res2a_relu');
net.addLayer('res2a_relu_rgb',net1.layers(idx).block,{'res2a_rgb'},{'res2ax_rgb'},{});
net.addLayer('res2a_relu_d',net2.layers(idx).block,{'res2a_d'},{'res2ax_d'},{});




% res2b_branch2a layer
idx = net1.getLayerIndex('res2b_branch2a');
net.addLayer('res2b_branch2a_rgb',net1.layers(idx).block,{'res2ax_rgb'},{'res2b_branch2a_rgb'},{'res2b_branch2a_filter_rgb'});
net.addLayer('res2b_branch2a_d',net2.layers(idx).block,{'res2ax_d'},{'res2b_branch2a_d'},{'res2b_branch2a_filter_d'});

f1 = net.getParamIndex('res2b_branch2a_filter_rgb');
f2 = net.getParamIndex('res2b_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2b_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2b_branch2a_filter_d';

% bn2b_branch2a layer
idx = net1.getLayerIndex('bn2b_branch2a');
net.addLayer('bn2b_branch2a_rgb',net1.layers(idx).block,{'res2b_branch2a_rgb'},{'res2b_branch2ax_rgb'},{'bn2b_branch2a_mult_rgb','bn2b_branch2a_bias_rgb','bn2b_branch2a_moments_rgb'});
net.addLayer('bn2b_branch2a_d',net2.layers(idx).block,{'res2b_branch2a_d'},{'res2b_branch2ax_d'},{'bn2b_branch2a_mult_d','bn2b_branch2a_bias_d','bn2b_branch2a_moments_d'});


f1 = net.getParamIndex('bn2b_branch2a_mult_rgb');
f2 = net.getParamIndex('bn2b_branch2a_bias_rgb');
f3 = net.getParamIndex('bn2b_branch2a_moments_rgb');
f4 = net.getParamIndex('bn2b_branch2a_mult_d');
f5 = net.getParamIndex('bn2b_branch2a_bias_d');
f6 = net.getParamIndex('bn2b_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2b_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2b_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2b_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2b_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2b_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2b_branch2a_moments_d';

% res2b_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2b_branch2a_relu_rgb', reluBlock_rgb, {'res2b_branch2ax_rgb'}, {'res2b_branch2axxx_rgb'}, {}) ;
net.addLayer('res2b_branch2a_relu_d', reluBlock_d, {'res2b_branch2ax_d'}, {'res2b_branch2axxx_d'}, {}) ;

% res2b_branch2b Layer
idx = net1.getLayerIndex('res2b_branch2b');
net.addLayer('res2b_branch2b_rgb',net1.layers(idx).block,{'res2b_branch2axxx_rgb'},{'res2b_branch2b_rgb'},{'res2b_branch2b_filter_rgb'});
net.addLayer('res2b_branch2b_d',net2.layers(idx).block,{'res2b_branch2axxx_d'},{'res2b_branch2b_d'},{'res2b_branch2b_filter_d'});

f1 = net.getParamIndex('res2b_branch2b_filter_rgb');
f2 = net.getParamIndex('res2b_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2b_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2b_branch2b_filter_d';

% bn2b_branch2b layer
idx = net1.getLayerIndex('bn2b_branch2b');
net.addLayer('bn2b_branch2b_rgb',net1.layers(idx).block,{'res2b_branch2b_rgb'},{'res2b_branch2bx_rgb'},{'bn2b_branch2b_mult_rgb','bn2b_branch2b_bias_rgb','bn2b_branch2b_moments_rgb'});
net.addLayer('bn2b_branch2b_d',net2.layers(idx).block,{'res2b_branch2b_d'},{'res2b_branch2bx_d'},{'bn2b_branch2b_mult_d','bn2b_branch2b_bias_d','bn2b_branch2b_moments_d'});


f1 = net.getParamIndex('bn2b_branch2b_mult_rgb');
f2 = net.getParamIndex('bn2b_branch2b_bias_rgb');
f3 = net.getParamIndex('bn2b_branch2b_moments_rgb');
f4 = net.getParamIndex('bn2b_branch2b_mult_d');
f5 = net.getParamIndex('bn2b_branch2b_bias_d');
f6 = net.getParamIndex('bn2b_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2b_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2b_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2b_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2b_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2b_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2b_branch2b_moments_d';

% res2b_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2b_branch2b_relu_rgb', reluBlock_rgb, {'res2b_branch2bx_rgb'}, {'res2b_branch2bxxx_rgb'}, {}) ;
net.addLayer('res2b_branch2b_relu_d', reluBlock_d, {'res2b_branch2bx_d'}, {'res2b_branch2bxxx_d'}, {}) ;

% res2b_branch2c Layer
idx = net1.getLayerIndex('res2b_branch2c');
net.addLayer('res2b_branch2c_rgb',net1.layers(idx).block,{'res2b_branch2bxxx_rgb'},{'res2b_branch2c_rgb'},{'res2b_branch2c_filter_rgb'});
net.addLayer('res2b_branch2c_d',net2.layers(idx).block,{'res2b_branch2bxxx_d'},{'res2b_branch2c_d'},{'res2b_branch2c_filter_d'});

f1 = net.getParamIndex('res2b_branch2c_filter_rgb');
f2 = net.getParamIndex('res2b_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2b_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2b_branch2c_filter_d';

% bn2b_branch2c layer
idx = net1.getLayerIndex('bn2b_branch2c');
net.addLayer('bn2b_branch2c_rgb',net1.layers(idx).block,{'res2b_branch2c_rgb'},{'res2b_branch2cx_rgb'},{'bn2b_branch2c_mult_rgb','bn2b_branch2c_bias_rgb','bn2b_branch2c_moments_rgb'});
net.addLayer('bn2b_branch2c_d',net2.layers(idx).block,{'res2b_branch2c_d'},{'res2b_branch2cx_d'},{'bn2b_branch2c_mult_d','bn2b_branch2c_bias_d','bn2b_branch2c_moments_d'});


f1 = net.getParamIndex('bn2b_branch2c_mult_rgb');
f2 = net.getParamIndex('bn2b_branch2c_bias_rgb');
f3 = net.getParamIndex('bn2b_branch2c_moments_rgb');
f4 = net.getParamIndex('bn2b_branch2c_mult_d');
f5 = net.getParamIndex('bn2b_branch2c_bias_d');
f6 = net.getParamIndex('bn2b_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2b_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2b_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2b_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2b_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2b_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2b_branch2c_moments_d';

% res2b layer(sum)
idx = net1.getLayerIndex('res2b');
net.addLayer('res2b_rgb',net1.layers(idx).block,{'res2ax_rgb','res2b_branch2cx_rgb'},{'res2b_rgb'},{});
net.addLayer('res2b_d',net2.layers(idx).block,{'res2ax_d','res2b_branch2cx_d'},{'res2b_d'},{});

% res2b_relu layer
idx = net1.getLayerIndex('res2b_relu');
net.addLayer('res2b_relu_rgb',net1.layers(idx).block,{'res2b_rgb'},{'res2bx_rgb'},{});
net.addLayer('res2b_relu_d',net2.layers(idx).block,{'res2b_d'},{'res2bx_d'},{});





% res2c_branch2a layer
idx = net1.getLayerIndex('res2c_branch2a');
net.addLayer('res2c_branch2a_rgb',net1.layers(idx).block,{'res2bx_rgb'},{'res2c_branch2a_rgb'},{'res2c_branch2a_filter_rgb'});
net.addLayer('res2c_branch2a_d',net2.layers(idx).block,{'res2bx_d'},{'res2c_branch2a_d'},{'res2c_branch2a_filter_d'});

f1 = net.getParamIndex('res2c_branch2a_filter_rgb');
f2 = net.getParamIndex('res2c_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2c_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2c_branch2a_filter_d';

% bn2c_branch2a layer
idx = net1.getLayerIndex('bn2c_branch2a');
net.addLayer('bn2c_branch2a_rgb',net1.layers(idx).block,{'res2c_branch2a_rgb'},{'res2c_branch2ax_rgb'},{'bn2c_branch2a_mult_rgb','bn2c_branch2a_bias_rgb','bn2c_branch2a_moments_rgb'});
net.addLayer('bn2c_branch2a_d',net2.layers(idx).block,{'res2c_branch2a_d'},{'res2c_branch2ax_d'},{'bn2c_branch2a_mult_d','bn2c_branch2a_bias_d','bn2c_branch2a_moments_d'});


f1 = net.getParamIndex('bn2c_branch2a_mult_rgb');
f2 = net.getParamIndex('bn2c_branch2a_bias_rgb');
f3 = net.getParamIndex('bn2c_branch2a_moments_rgb');
f4 = net.getParamIndex('bn2c_branch2a_mult_d');
f5 = net.getParamIndex('bn2c_branch2a_bias_d');
f6 = net.getParamIndex('bn2c_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2c_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2c_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2c_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2c_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2c_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2c_branch2a_moments_d';

% res2c_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2c_branch2a_relu_rgb', reluBlock_rgb, {'res2c_branch2ax_rgb'}, {'res2c_branch2axxx_rgb'}, {}) ;
net.addLayer('res2c_branch2a_relu_d', reluBlock_d, {'res2c_branch2ax_d'}, {'res2c_branch2axxx_d'}, {}) ;

% res2c_branch2b Layer
idx = net1.getLayerIndex('res2c_branch2b');
net.addLayer('res2c_branch2b_rgb',net1.layers(idx).block,{'res2c_branch2axxx_rgb'},{'res2c_branch2b_rgb'},{'res2c_branch2b_filter_rgb'});
net.addLayer('res2c_branch2b_d',net2.layers(idx).block,{'res2c_branch2axxx_d'},{'res2c_branch2b_d'},{'res2c_branch2b_filter_d'});

f1 = net.getParamIndex('res2c_branch2b_filter_rgb');
f2 = net.getParamIndex('res2c_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2c_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2c_branch2b_filter_d';

% bn2c_branch2b layer
idx = net1.getLayerIndex('bn2c_branch2b');
net.addLayer('bn2c_branch2b_rgb',net1.layers(idx).block,{'res2c_branch2b_rgb'},{'res2c_branch2bx_rgb'},{'bn2c_branch2b_mult_rgb','bn2c_branch2b_bias_rgb','bn2c_branch2b_moments_rgb'});
net.addLayer('bn2c_branch2b_d',net2.layers(idx).block,{'res2c_branch2b_d'},{'res2c_branch2bx_d'},{'bn2c_branch2b_mult_d','bn2c_branch2b_bias_d','bn2c_branch2b_moments_d'});


f1 = net.getParamIndex('bn2c_branch2b_mult_rgb');
f2 = net.getParamIndex('bn2c_branch2b_bias_rgb');
f3 = net.getParamIndex('bn2c_branch2b_moments_rgb');
f4 = net.getParamIndex('bn2c_branch2b_mult_d');
f5 = net.getParamIndex('bn2c_branch2b_bias_d');
f6 = net.getParamIndex('bn2c_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2c_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2c_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2c_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2c_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2c_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2c_branch2b_moments_d';

% res2c_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res2c_branch2b_relu_rgb', reluBlock_rgb, {'res2c_branch2bx_rgb'}, {'res2c_branch2bxxx_rgb'}, {}) ;
net.addLayer('res2c_branch2b_relu_d', reluBlock_d, {'res2c_branch2bx_d'}, {'res2c_branch2bxxx_d'}, {}) ;

% res2c_branch2c Layer
idx = net1.getLayerIndex('res2c_branch2c');
net.addLayer('res2c_branch2c_rgb',net1.layers(idx).block,{'res2c_branch2bxxx_rgb'},{'res2c_branch2c_rgb'},{'res2c_branch2c_filter_rgb'});
net.addLayer('res2c_branch2c_d',net2.layers(idx).block,{'res2c_branch2bxxx_d'},{'res2c_branch2c_d'},{'res2c_branch2c_filter_d'});

f1 = net.getParamIndex('res2c_branch2c_filter_rgb');
f2 = net.getParamIndex('res2c_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res2c_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res2c_branch2c_filter_d';

% bn2c_branch2c layer
idx = net1.getLayerIndex('bn2c_branch2c');
net.addLayer('bn2c_branch2c_rgb',net1.layers(idx).block,{'res2c_branch2c_rgb'},{'res2c_branch2cx_rgb'},{'bn2c_branch2c_mult_rgb','bn2c_branch2c_bias_rgb','bn2c_branch2c_moments_rgb'});
net.addLayer('bn2c_branch2c_d',net2.layers(idx).block,{'res2c_branch2c_d'},{'res2c_branch2cx_d'},{'bn2c_branch2c_mult_d','bn2c_branch2c_bias_d','bn2c_branch2c_moments_d'});


f1 = net.getParamIndex('bn2c_branch2c_mult_rgb');
f2 = net.getParamIndex('bn2c_branch2c_bias_rgb');
f3 = net.getParamIndex('bn2c_branch2c_moments_rgb');
f4 = net.getParamIndex('bn2c_branch2c_mult_d');
f5 = net.getParamIndex('bn2c_branch2c_bias_d');
f6 = net.getParamIndex('bn2c_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn2c_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn2c_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn2c_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn2c_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn2c_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn2c_branch2c_moments_d';

% res2c layer(sum)
idx = net1.getLayerIndex('res2c');
net.addLayer('res2c_rgb',net1.layers(idx).block,{'res2bx_rgb','res2c_branch2cx_rgb'},{'res2c_rgb'},{});
net.addLayer('res2c_d',net2.layers(idx).block,{'res2bx_d','res2c_branch2cx_d'},{'res2c_d'},{});

% res2c_relu layer
idx = net1.getLayerIndex('res2c_relu');
net.addLayer('res2c_relu_rgb',net1.layers(idx).block,{'res2c_rgb'},{'res2cx_rgb'},{});
net.addLayer('res2c_relu_d',net2.layers(idx).block,{'res2c_d'},{'res2cx_d'},{});








% res3a_branch1 layer
idx = net1.getLayerIndex('res3a_branch1');
net.addLayer('res3a_branch1_rgb',net1.layers(idx).block,{'res2cx_rgb'},{'res3a_branch1_rgb'},{'res3a_branch1_filter_rgb'});
net.addLayer('res3a_branch1_d',net2.layers(idx).block,{'res2cx_d'},{'res3a_branch1_d'},{'res3a_branch1_filter_d'});

f1 = net.getParamIndex('res3a_branch1_filter_rgb');
f2 = net.getParamIndex('res3a_branch1_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3a_branch1_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3a_branch1_filter_d';

% bn3a_branch1 Layer
idx = net1.getLayerIndex('bn3a_branch1');
net.addLayer('bn3a_branch1_rgb',net1.layers(idx).block,{'res3a_branch1_rgb'},{'res3a_branch1x_rgb'},{'bn3a_branch1_mult_rgb','bn3a_branch1_bias_rgb','bn3a_branch1_moments_rgb'});
net.addLayer('bn3a_branch1_d',net2.layers(idx).block,{'res3a_branch1_d'},{'res3a_branch1x_d'},{'bn3a_branch1_mult_d','bn3a_branch1_bias_d','bn3a_branch1_moments_d'});


f1 = net.getParamIndex('bn3a_branch1_mult_rgb');
f2 = net.getParamIndex('bn3a_branch1_bias_rgb');
f3 = net.getParamIndex('bn3a_branch1_moments_rgb');
f4 = net.getParamIndex('bn3a_branch1_mult_d');
f5 = net.getParamIndex('bn3a_branch1_bias_d');
f6 = net.getParamIndex('bn3a_branch1_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3a_branch1_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3a_branch1_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3a_branch1_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3a_branch1_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3a_branch1_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3a_branch1_moments_d';

% res3a_branch2a layer
idx = net1.getLayerIndex('res3a_branch2a');
net.addLayer('res3a_branch2a_rgb',net1.layers(idx).block,{'res2cx_rgb'},{'res3a_branch2a_rgb'},{'res3a_branch2a_filter_rgb'});
net.addLayer('res3a_branch2a_d',net2.layers(idx).block,{'res2cx_d'},{'res3a_branch2a_d'},{'res3a_branch2a_filter_d'});

f1 = net.getParamIndex('res3a_branch2a_filter_rgb');
f2 = net.getParamIndex('res3a_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3a_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3a_branch2a_filter_d';

% bn3a_branch2a layer
idx = net1.getLayerIndex('bn3a_branch2a');
net.addLayer('bn3a_branch2a_rgb',net1.layers(idx).block,{'res3a_branch2a_rgb'},{'res3a_branch2ax_rgb'},{'bn3a_branch2a_mult_rgb','bn3a_branch2a_bias_rgb','bn3a_branch2a_moments_rgb'});
net.addLayer('bn3a_branch2a_d',net2.layers(idx).block,{'res3a_branch2a_d'},{'res3a_branch2ax_d'},{'bn3a_branch2a_mult_d','bn3a_branch2a_bias_d','bn3a_branch2a_moments_d'});


f1 = net.getParamIndex('bn3a_branch2a_mult_rgb');
f2 = net.getParamIndex('bn3a_branch2a_bias_rgb');
f3 = net.getParamIndex('bn3a_branch2a_moments_rgb');
f4 = net.getParamIndex('bn3a_branch2a_mult_d');
f5 = net.getParamIndex('bn3a_branch2a_bias_d');
f6 = net.getParamIndex('bn3a_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3a_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3a_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3a_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3a_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3a_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3a_branch2a_moments_d';

% res3a_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3a_branch2a_relu_rgb', reluBlock_rgb, {'res3a_branch2ax_rgb'}, {'res3a_branch2axxx_rgb'}, {}) ;
net.addLayer('res3a_branch2a_relu_d', reluBlock_d, {'res3a_branch2ax_d'}, {'res3a_branch2axxx_d'}, {}) ;

% res3a_branch2b Layer
idx = net1.getLayerIndex('res3a_branch2b');
net.addLayer('res3a_branch2b_rgb',net1.layers(idx).block,{'res3a_branch2axxx_rgb'},{'res3a_branch2b_rgb'},{'res3a_branch2b_filter_rgb'});
net.addLayer('res3a_branch2b_d',net2.layers(idx).block,{'res3a_branch2axxx_d'},{'res3a_branch2b_d'},{'res3a_branch2b_filter_d'});

f1 = net.getParamIndex('res3a_branch2b_filter_rgb');
f2 = net.getParamIndex('res3a_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3a_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3a_branch2b_filter_d';

% bn3a_branch2b layer
idx = net1.getLayerIndex('bn3a_branch2b');
net.addLayer('bn3a_branch2b_rgb',net1.layers(idx).block,{'res3a_branch2b_rgb'},{'res3a_branch2bx_rgb'},{'bn3a_branch2b_mult_rgb','bn3a_branch2b_bias_rgb','bn3a_branch2b_moments_rgb'});
net.addLayer('bn3a_branch2b_d',net2.layers(idx).block,{'res3a_branch2b_d'},{'res3a_branch2bx_d'},{'bn3a_branch2b_mult_d','bn3a_branch2b_bias_d','bn3a_branch2b_moments_d'});


f1 = net.getParamIndex('bn3a_branch2b_mult_rgb');
f2 = net.getParamIndex('bn3a_branch2b_bias_rgb');
f3 = net.getParamIndex('bn3a_branch2b_moments_rgb');
f4 = net.getParamIndex('bn3a_branch2b_mult_d');
f5 = net.getParamIndex('bn3a_branch2b_bias_d');
f6 = net.getParamIndex('bn3a_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3a_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3a_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3a_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3a_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3a_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3a_branch2b_moments_d';

% res3a_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3a_branch2b_relu_rgb', reluBlock_rgb, {'res3a_branch2bx_rgb'}, {'res3a_branch2bxxx_rgb'}, {}) ;
net.addLayer('res3a_branch2b_relu_d', reluBlock_d, {'res3a_branch2bx_d'}, {'res3a_branch2bxxx_d'}, {}) ;

% res3a_branch2c Layer
idx = net1.getLayerIndex('res3a_branch2c');
net.addLayer('res3a_branch2c_rgb',net1.layers(idx).block,{'res3a_branch2bxxx_rgb'},{'res3a_branch2c_rgb'},{'res3a_branch2c_filter_rgb'});
net.addLayer('res3a_branch2c_d',net2.layers(idx).block,{'res3a_branch2bxxx_d'},{'res3a_branch2c_d'},{'res3a_branch2c_filter_d'});

f1 = net.getParamIndex('res3a_branch2c_filter_rgb');
f2 = net.getParamIndex('res3a_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3a_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3a_branch2c_filter_d';

% bn3a_branch2c layer
idx = net1.getLayerIndex('bn3a_branch2c');
net.addLayer('bn3a_branch2c_rgb',net1.layers(idx).block,{'res3a_branch2c_rgb'},{'res3a_branch2cx_rgb'},{'bn3a_branch2c_mult_rgb','bn3a_branch2c_bias_rgb','bn3a_branch2c_moments_rgb'});
net.addLayer('bn3a_branch2c_d',net2.layers(idx).block,{'res3a_branch2c_d'},{'res3a_branch2cx_d'},{'bn3a_branch2c_mult_d','bn3a_branch2c_bias_d','bn3a_branch2c_moments_d'});


f1 = net.getParamIndex('bn3a_branch2c_mult_rgb');
f2 = net.getParamIndex('bn3a_branch2c_bias_rgb');
f3 = net.getParamIndex('bn3a_branch2c_moments_rgb');
f4 = net.getParamIndex('bn3a_branch2c_mult_d');
f5 = net.getParamIndex('bn3a_branch2c_bias_d');
f6 = net.getParamIndex('bn3a_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3a_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3a_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3a_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3a_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3a_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3a_branch2c_moments_d';

% res3a layer(sum)
idx = net1.getLayerIndex('res3a');
net.addLayer('res3a_rgb',net1.layers(idx).block,{'res3a_branch1x_rgb','res3a_branch2cx_rgb'},{'res3a_rgb'},{});
net.addLayer('res3a_d',net2.layers(idx).block,{'res3a_branch1x_d','res3a_branch2cx_d'},{'res3a_d'},{});

% res3a_relu layer
idx = net1.getLayerIndex('res3a_relu');
net.addLayer('res3a_relu_rgb',net1.layers(idx).block,{'res3a_rgb'},{'res3ax_rgb'},{});
net.addLayer('res3a_relu_d',net2.layers(idx).block,{'res3a_d'},{'res3ax_d'},{});




% res3b_branch2a layer
idx = net1.getLayerIndex('res3b_branch2a');
net.addLayer('res3b_branch2a_rgb',net1.layers(idx).block,{'res3ax_rgb'},{'res3b_branch2a_rgb'},{'res3b_branch2a_filter_rgb'});
net.addLayer('res3b_branch2a_d',net2.layers(idx).block,{'res3ax_d'},{'res3b_branch2a_d'},{'res3b_branch2a_filter_d'});

f1 = net.getParamIndex('res3b_branch2a_filter_rgb');
f2 = net.getParamIndex('res3b_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3b_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3b_branch2a_filter_d';

% bn3b_branch2a layer
idx = net1.getLayerIndex('bn3b_branch2a');
net.addLayer('bn3b_branch2a_rgb',net1.layers(idx).block,{'res3b_branch2a_rgb'},{'res3b_branch2ax_rgb'},{'bn3b_branch2a_mult_rgb','bn3b_branch2a_bias_rgb','bn3b_branch2a_moments_rgb'});
net.addLayer('bn3b_branch2a_d',net2.layers(idx).block,{'res3b_branch2a_d'},{'res3b_branch2ax_d'},{'bn3b_branch2a_mult_d','bn3b_branch2a_bias_d','bn3b_branch2a_moments_d'});


f1 = net.getParamIndex('bn3b_branch2a_mult_rgb');
f2 = net.getParamIndex('bn3b_branch2a_bias_rgb');
f3 = net.getParamIndex('bn3b_branch2a_moments_rgb');
f4 = net.getParamIndex('bn3b_branch2a_mult_d');
f5 = net.getParamIndex('bn3b_branch2a_bias_d');
f6 = net.getParamIndex('bn3b_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3b_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3b_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3b_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3b_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3b_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3b_branch2a_moments_d';

% res3b_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3b_branch2a_relu_rgb', reluBlock_rgb, {'res3b_branch2ax_rgb'}, {'res3b_branch2axxx_rgb'}, {}) ;
net.addLayer('res3b_branch2a_relu_d', reluBlock_d, {'res3b_branch2ax_d'}, {'res3b_branch2axxx_d'}, {}) ;

% res3b_branch2b Layer
idx = net1.getLayerIndex('res3b_branch2b');
net.addLayer('res3b_branch2b_rgb',net1.layers(idx).block,{'res3b_branch2axxx_rgb'},{'res3b_branch2b_rgb'},{'res3b_branch2b_filter_rgb'});
net.addLayer('res3b_branch2b_d',net2.layers(idx).block,{'res3b_branch2axxx_d'},{'res3b_branch2b_d'},{'res3b_branch2b_filter_d'});

f1 = net.getParamIndex('res3b_branch2b_filter_rgb');
f2 = net.getParamIndex('res3b_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3b_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3b_branch2b_filter_d';

% bn3b_branch2b layer
idx = net1.getLayerIndex('bn3b_branch2b');
net.addLayer('bn3b_branch2b_rgb',net1.layers(idx).block,{'res3b_branch2b_rgb'},{'res3b_branch2bx_rgb'},{'bn3b_branch2b_mult_rgb','bn3b_branch2b_bias_rgb','bn3b_branch2b_moments_rgb'});
net.addLayer('bn3b_branch2b_d',net2.layers(idx).block,{'res3b_branch2b_d'},{'res3b_branch2bx_d'},{'bn3b_branch2b_mult_d','bn3b_branch2b_bias_d','bn3b_branch2b_moments_d'});


f1 = net.getParamIndex('bn3b_branch2b_mult_rgb');
f2 = net.getParamIndex('bn3b_branch2b_bias_rgb');
f3 = net.getParamIndex('bn3b_branch2b_moments_rgb');
f4 = net.getParamIndex('bn3b_branch2b_mult_d');
f5 = net.getParamIndex('bn3b_branch2b_bias_d');
f6 = net.getParamIndex('bn3b_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3b_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3b_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3b_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3b_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3b_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3b_branch2b_moments_d';

% res3b_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3b_branch2b_relu_rgb', reluBlock_rgb, {'res3b_branch2bx_rgb'}, {'res3b_branch2bxxx_rgb'}, {}) ;
net.addLayer('res3b_branch2b_relu_d', reluBlock_d, {'res3b_branch2bx_d'}, {'res3b_branch2bxxx_d'}, {}) ;

% res3b_branch2c Layer
idx = net1.getLayerIndex('res3b_branch2c');
net.addLayer('res3b_branch2c_rgb',net1.layers(idx).block,{'res3b_branch2bxxx_rgb'},{'res3b_branch2c_rgb'},{'res3b_branch2c_filter_rgb'});
net.addLayer('res3b_branch2c_d',net2.layers(idx).block,{'res3b_branch2bxxx_d'},{'res3b_branch2c_d'},{'res3b_branch2c_filter_d'});

f1 = net.getParamIndex('res3b_branch2c_filter_rgb');
f2 = net.getParamIndex('res3b_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3b_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3b_branch2c_filter_d';

% bn3b_branch2c layer
idx = net1.getLayerIndex('bn3b_branch2c');
net.addLayer('bn3b_branch2c_rgb',net1.layers(idx).block,{'res3b_branch2c_rgb'},{'res3b_branch2cx_rgb'},{'bn3b_branch2c_mult_rgb','bn3b_branch2c_bias_rgb','bn3b_branch2c_moments_rgb'});
net.addLayer('bn3b_branch2c_d',net2.layers(idx).block,{'res3b_branch2c_d'},{'res3b_branch2cx_d'},{'bn3b_branch2c_mult_d','bn3b_branch2c_bias_d','bn3b_branch2c_moments_d'});


f1 = net.getParamIndex('bn3b_branch2c_mult_rgb');
f2 = net.getParamIndex('bn3b_branch2c_bias_rgb');
f3 = net.getParamIndex('bn3b_branch2c_moments_rgb');
f4 = net.getParamIndex('bn3b_branch2c_mult_d');
f5 = net.getParamIndex('bn3b_branch2c_bias_d');
f6 = net.getParamIndex('bn3b_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3b_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3b_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3b_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3b_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3b_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3b_branch2c_moments_d';

% res3b layer(sum)
idx = net1.getLayerIndex('res3b');
net.addLayer('res3b_rgb',net1.layers(idx).block,{'res3ax_rgb','res3b_branch2cx_rgb'},{'res3b_rgb'},{});
net.addLayer('res3b_d',net2.layers(idx).block,{'res3ax_d','res3b_branch2cx_d'},{'res3b_d'},{});

% res3b_relu layer
idx = net1.getLayerIndex('res3b_relu');
net.addLayer('res3b_relu_rgb',net1.layers(idx).block,{'res3b_rgb'},{'res3bx_rgb'},{});
net.addLayer('res3b_relu_d',net2.layers(idx).block,{'res3b_d'},{'res3bx_d'},{});




% res3c_branch2a layer
idx = net1.getLayerIndex('res3c_branch2a');
net.addLayer('res3c_branch2a_rgb',net1.layers(idx).block,{'res3bx_rgb'},{'res3c_branch2a_rgb'},{'res3c_branch2a_filter_rgb'});
net.addLayer('res3c_branch2a_d',net2.layers(idx).block,{'res3bx_d'},{'res3c_branch2a_d'},{'res3c_branch2a_filter_d'});

f1 = net.getParamIndex('res3c_branch2a_filter_rgb');
f2 = net.getParamIndex('res3c_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3c_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3c_branch2a_filter_d';

% bn3c_branch2a layer
idx = net1.getLayerIndex('bn3c_branch2a');
net.addLayer('bn3c_branch2a_rgb',net1.layers(idx).block,{'res3c_branch2a_rgb'},{'res3c_branch2ax_rgb'},{'bn3c_branch2a_mult_rgb','bn3c_branch2a_bias_rgb','bn3c_branch2a_moments_rgb'});
net.addLayer('bn3c_branch2a_d',net2.layers(idx).block,{'res3c_branch2a_d'},{'res3c_branch2ax_d'},{'bn3c_branch2a_mult_d','bn3c_branch2a_bias_d','bn3c_branch2a_moments_d'});


f1 = net.getParamIndex('bn3c_branch2a_mult_rgb');
f2 = net.getParamIndex('bn3c_branch2a_bias_rgb');
f3 = net.getParamIndex('bn3c_branch2a_moments_rgb');
f4 = net.getParamIndex('bn3c_branch2a_mult_d');
f5 = net.getParamIndex('bn3c_branch2a_bias_d');
f6 = net.getParamIndex('bn3c_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3c_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3c_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3c_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3c_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3c_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3c_branch2a_moments_d';

% res3c_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3c_branch2a_relu_rgb', reluBlock_rgb, {'res3c_branch2ax_rgb'}, {'res3c_branch2axxx_rgb'}, {}) ;
net.addLayer('res3c_branch2a_relu_d', reluBlock_d, {'res3c_branch2ax_d'}, {'res3c_branch2axxx_d'}, {}) ;

% res3c_branch2b Layer
idx = net1.getLayerIndex('res3c_branch2b');
net.addLayer('res3c_branch2b_rgb',net1.layers(idx).block,{'res3c_branch2axxx_rgb'},{'res3c_branch2b_rgb'},{'res3c_branch2b_filter_rgb'});
net.addLayer('res3c_branch2b_d',net2.layers(idx).block,{'res3c_branch2axxx_d'},{'res3c_branch2b_d'},{'res3c_branch2b_filter_d'});

f1 = net.getParamIndex('res3c_branch2b_filter_rgb');
f2 = net.getParamIndex('res3c_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3c_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3c_branch2b_filter_d';

% bn3c_branch2b layer
idx = net1.getLayerIndex('bn3c_branch2b');
net.addLayer('bn3c_branch2b_rgb',net1.layers(idx).block,{'res3c_branch2b_rgb'},{'res3c_branch2bx_rgb'},{'bn3c_branch2b_mult_rgb','bn3c_branch2b_bias_rgb','bn3c_branch2b_moments_rgb'});
net.addLayer('bn3c_branch2b_d',net2.layers(idx).block,{'res3c_branch2b_d'},{'res3c_branch2bx_d'},{'bn3c_branch2b_mult_d','bn3c_branch2b_bias_d','bn3c_branch2b_moments_d'});


f1 = net.getParamIndex('bn3c_branch2b_mult_rgb');
f2 = net.getParamIndex('bn3c_branch2b_bias_rgb');
f3 = net.getParamIndex('bn3c_branch2b_moments_rgb');
f4 = net.getParamIndex('bn3c_branch2b_mult_d');
f5 = net.getParamIndex('bn3c_branch2b_bias_d');
f6 = net.getParamIndex('bn3c_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3c_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3c_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3c_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3c_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3c_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3c_branch2b_moments_d';

% res3c_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3c_branch2b_relu_rgb', reluBlock_rgb, {'res3c_branch2bx_rgb'}, {'res3c_branch2bxxx_rgb'}, {}) ;
net.addLayer('res3c_branch2b_relu_d', reluBlock_d, {'res3c_branch2bx_d'}, {'res3c_branch2bxxx_d'}, {}) ;

% res3c_branch2c Layer
idx = net1.getLayerIndex('res3c_branch2c');
net.addLayer('res3c_branch2c_rgb',net1.layers(idx).block,{'res3c_branch2bxxx_rgb'},{'res3c_branch2c_rgb'},{'res3c_branch2c_filter_rgb'});
net.addLayer('res3c_branch2c_d',net2.layers(idx).block,{'res3c_branch2bxxx_d'},{'res3c_branch2c_d'},{'res3c_branch2c_filter_d'});

f1 = net.getParamIndex('res3c_branch2c_filter_rgb');
f2 = net.getParamIndex('res3c_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3c_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3c_branch2c_filter_d';

% bn3c_branch2c layer
idx = net1.getLayerIndex('bn3c_branch2c');
net.addLayer('bn3c_branch2c_rgb',net1.layers(idx).block,{'res3c_branch2c_rgb'},{'res3c_branch2cx_rgb'},{'bn3c_branch2c_mult_rgb','bn3c_branch2c_bias_rgb','bn3c_branch2c_moments_rgb'});
net.addLayer('bn3c_branch2c_d',net2.layers(idx).block,{'res3c_branch2c_d'},{'res3c_branch2cx_d'},{'bn3c_branch2c_mult_d','bn3c_branch2c_bias_d','bn3c_branch2c_moments_d'});


f1 = net.getParamIndex('bn3c_branch2c_mult_rgb');
f2 = net.getParamIndex('bn3c_branch2c_bias_rgb');
f3 = net.getParamIndex('bn3c_branch2c_moments_rgb');
f4 = net.getParamIndex('bn3c_branch2c_mult_d');
f5 = net.getParamIndex('bn3c_branch2c_bias_d');
f6 = net.getParamIndex('bn3c_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3c_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3c_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3c_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3c_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3c_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3c_branch2c_moments_d';

% res3c layer(sum)
idx = net1.getLayerIndex('res3c');
net.addLayer('res3c_rgb',net1.layers(idx).block,{'res3bx_rgb','res3c_branch2cx_rgb'},{'res3c_rgb'},{});
net.addLayer('res3c_d',net2.layers(idx).block,{'res3bx_d','res3c_branch2cx_d'},{'res3c_d'},{});

% res3c_relu layer
idx = net1.getLayerIndex('res3c_relu');
net.addLayer('res3c_relu_rgb',net1.layers(idx).block,{'res3c_rgb'},{'res3cx_rgb'},{});
net.addLayer('res3c_relu_d',net2.layers(idx).block,{'res3c_d'},{'res3cx_d'},{});




% res3d_branch2a layer
idx = net1.getLayerIndex('res3d_branch2a');
net.addLayer('res3d_branch2a_rgb',net1.layers(idx).block,{'res3cx_rgb'},{'res3d_branch2a_rgb'},{'res3d_branch2a_filter_rgb'});
net.addLayer('res3d_branch2a_d',net2.layers(idx).block,{'res3cx_d'},{'res3d_branch2a_d'},{'res3d_branch2a_filter_d'});

f1 = net.getParamIndex('res3d_branch2a_filter_rgb');
f2 = net.getParamIndex('res3d_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3d_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3d_branch2a_filter_d';

% bn3d_branch2a layer
idx = net1.getLayerIndex('bn3d_branch2a');
net.addLayer('bn3d_branch2a_rgb',net1.layers(idx).block,{'res3d_branch2a_rgb'},{'res3d_branch2ax_rgb'},{'bn3d_branch2a_mult_rgb','bn3d_branch2a_bias_rgb','bn3d_branch2a_moments_rgb'});
net.addLayer('bn3d_branch2a_d',net2.layers(idx).block,{'res3d_branch2a_d'},{'res3d_branch2ax_d'},{'bn3d_branch2a_mult_d','bn3d_branch2a_bias_d','bn3d_branch2a_moments_d'});


f1 = net.getParamIndex('bn3d_branch2a_mult_rgb');
f2 = net.getParamIndex('bn3d_branch2a_bias_rgb');
f3 = net.getParamIndex('bn3d_branch2a_moments_rgb');
f4 = net.getParamIndex('bn3d_branch2a_mult_d');
f5 = net.getParamIndex('bn3d_branch2a_bias_d');
f6 = net.getParamIndex('bn3d_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3d_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3d_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3d_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3d_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3d_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3d_branch2a_moments_d';

% res3d_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3d_branch2a_relu_rgb', reluBlock_rgb, {'res3d_branch2ax_rgb'}, {'res3d_branch2axxx_rgb'}, {}) ;
net.addLayer('res3d_branch2a_relu_d', reluBlock_d, {'res3d_branch2ax_d'}, {'res3d_branch2axxx_d'}, {}) ;

% res3d_branch2b Layer
idx = net1.getLayerIndex('res3d_branch2b');
net.addLayer('res3d_branch2b_rgb',net1.layers(idx).block,{'res3d_branch2axxx_rgb'},{'res3d_branch2b_rgb'},{'res3d_branch2b_filter_rgb'});
net.addLayer('res3d_branch2b_d',net2.layers(idx).block,{'res3d_branch2axxx_d'},{'res3d_branch2b_d'},{'res3d_branch2b_filter_d'});

f1 = net.getParamIndex('res3d_branch2b_filter_rgb');
f2 = net.getParamIndex('res3d_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3d_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3d_branch2b_filter_d';

% bn3d_branch2b layer
idx = net1.getLayerIndex('bn3d_branch2b');
net.addLayer('bn3d_branch2b_rgb',net1.layers(idx).block,{'res3d_branch2b_rgb'},{'res3d_branch2bx_rgb'},{'bn3d_branch2b_mult_rgb','bn3d_branch2b_bias_rgb','bn3d_branch2b_moments_rgb'});
net.addLayer('bn3d_branch2b_d',net2.layers(idx).block,{'res3d_branch2b_d'},{'res3d_branch2bx_d'},{'bn3d_branch2b_mult_d','bn3d_branch2b_bias_d','bn3d_branch2b_moments_d'});


f1 = net.getParamIndex('bn3d_branch2b_mult_rgb');
f2 = net.getParamIndex('bn3d_branch2b_bias_rgb');
f3 = net.getParamIndex('bn3d_branch2b_moments_rgb');
f4 = net.getParamIndex('bn3d_branch2b_mult_d');
f5 = net.getParamIndex('bn3d_branch2b_bias_d');
f6 = net.getParamIndex('bn3d_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3d_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3d_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3d_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3d_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3d_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3d_branch2b_moments_d';

% res3d_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res3d_branch2b_relu_rgb', reluBlock_rgb, {'res3d_branch2bx_rgb'}, {'res3d_branch2bxxx_rgb'}, {}) ;
net.addLayer('res3d_branch2b_relu_d', reluBlock_d, {'res3d_branch2bx_d'}, {'res3d_branch2bxxx_d'}, {}) ;

% res3d_branch2c Layer
idx = net1.getLayerIndex('res3d_branch2c');
net.addLayer('res3d_branch2c_rgb',net1.layers(idx).block,{'res3d_branch2bxxx_rgb'},{'res3d_branch2c_rgb'},{'res3d_branch2c_filter_rgb'});
net.addLayer('res3d_branch2c_d',net2.layers(idx).block,{'res3d_branch2bxxx_d'},{'res3d_branch2c_d'},{'res3d_branch2c_filter_d'});

f1 = net.getParamIndex('res3d_branch2c_filter_rgb');
f2 = net.getParamIndex('res3d_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res3d_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res3d_branch2c_filter_d';

% bn3d_branch2c layer
idx = net1.getLayerIndex('bn3d_branch2c');
net.addLayer('bn3d_branch2c_rgb',net1.layers(idx).block,{'res3d_branch2c_rgb'},{'res3d_branch2cx_rgb'},{'bn3d_branch2c_mult_rgb','bn3d_branch2c_bias_rgb','bn3d_branch2c_moments_rgb'});
net.addLayer('bn3d_branch2c_d',net2.layers(idx).block,{'res3d_branch2c_d'},{'res3d_branch2cx_d'},{'bn3d_branch2c_mult_d','bn3d_branch2c_bias_d','bn3d_branch2c_moments_d'});


f1 = net.getParamIndex('bn3d_branch2c_mult_rgb');
f2 = net.getParamIndex('bn3d_branch2c_bias_rgb');
f3 = net.getParamIndex('bn3d_branch2c_moments_rgb');
f4 = net.getParamIndex('bn3d_branch2c_mult_d');
f5 = net.getParamIndex('bn3d_branch2c_bias_d');
f6 = net.getParamIndex('bn3d_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn3d_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn3d_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn3d_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn3d_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn3d_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn3d_branch2c_moments_d';

% res3d layer(sum)
idx = net1.getLayerIndex('res3d');
net.addLayer('res3d_rgb',net1.layers(idx).block,{'res3cx_rgb','res3d_branch2cx_rgb'},{'res3d_rgb'},{});
net.addLayer('res3d_d',net2.layers(idx).block,{'res3cx_d','res3d_branch2cx_d'},{'res3d_d'},{});

% res3d_relu layer
idx = net1.getLayerIndex('res3d_relu');
net.addLayer('res3d_relu_rgb',net1.layers(idx).block,{'res3d_rgb'},{'res3dx_rgb'},{});
net.addLayer('res3d_relu_d',net2.layers(idx).block,{'res3d_d'},{'res3dx_d'},{});







% res4a_branch1 layer
idx = net1.getLayerIndex('res4a_branch1');
net.addLayer('res4a_branch1_rgb',net1.layers(idx).block,{'res3dx_rgb'},{'res4a_branch1_rgb'},{'res4a_branch1_filter_rgb'});
net.addLayer('res4a_branch1_d',net2.layers(idx).block,{'res3dx_d'},{'res4a_branch1_d'},{'res4a_branch1_filter_d'});

f1 = net.getParamIndex('res4a_branch1_filter_rgb');
f2 = net.getParamIndex('res4a_branch1_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4a_branch1_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4a_branch1_filter_d';

% bn4a_branch1 Layer
idx = net1.getLayerIndex('bn4a_branch1');
net.addLayer('bn4a_branch1_rgb',net1.layers(idx).block,{'res4a_branch1_rgb'},{'res4a_branch1x_rgb'},{'bn4a_branch1_mult_rgb','bn4a_branch1_bias_rgb','bn4a_branch1_moments_rgb'});
net.addLayer('bn4a_branch1_d',net2.layers(idx).block,{'res4a_branch1_d'},{'res4a_branch1x_d'},{'bn4a_branch1_mult_d','bn4a_branch1_bias_d','bn4a_branch1_moments_d'});


f1 = net.getParamIndex('bn4a_branch1_mult_rgb');
f2 = net.getParamIndex('bn4a_branch1_bias_rgb');
f3 = net.getParamIndex('bn4a_branch1_moments_rgb');
f4 = net.getParamIndex('bn4a_branch1_mult_d');
f5 = net.getParamIndex('bn4a_branch1_bias_d');
f6 = net.getParamIndex('bn4a_branch1_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4a_branch1_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4a_branch1_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4a_branch1_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4a_branch1_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4a_branch1_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4a_branch1_moments_d';

% res4a_branch2a layer
idx = net1.getLayerIndex('res4a_branch2a');
net.addLayer('res4a_branch2a_rgb',net1.layers(idx).block,{'res3dx_rgb'},{'res4a_branch2a_rgb'},{'res4a_branch2a_filter_rgb'});
net.addLayer('res4a_branch2a_d',net2.layers(idx).block,{'res3dx_d'},{'res4a_branch2a_d'},{'res4a_branch2a_filter_d'});

f1 = net.getParamIndex('res4a_branch2a_filter_rgb');
f2 = net.getParamIndex('res4a_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4a_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4a_branch2a_filter_d';

% bn4a_branch2a layer
idx = net1.getLayerIndex('bn4a_branch2a');
net.addLayer('bn4a_branch2a_rgb',net1.layers(idx).block,{'res4a_branch2a_rgb'},{'res4a_branch2ax_rgb'},{'bn4a_branch2a_mult_rgb','bn4a_branch2a_bias_rgb','bn4a_branch2a_moments_rgb'});
net.addLayer('bn4a_branch2a_d',net2.layers(idx).block,{'res4a_branch2a_d'},{'res4a_branch2ax_d'},{'bn4a_branch2a_mult_d','bn4a_branch2a_bias_d','bn4a_branch2a_moments_d'});


f1 = net.getParamIndex('bn4a_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4a_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4a_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4a_branch2a_mult_d');
f5 = net.getParamIndex('bn4a_branch2a_bias_d');
f6 = net.getParamIndex('bn4a_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4a_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4a_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4a_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4a_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4a_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4a_branch2a_moments_d';

% res4a_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4a_branch2a_relu_rgb', reluBlock_rgb, {'res4a_branch2ax_rgb'}, {'res4a_branch2axxx_rgb'}, {}) ;
net.addLayer('res4a_branch2a_relu_d', reluBlock_d, {'res4a_branch2ax_d'}, {'res4a_branch2axxx_d'}, {}) ;

% res4a_branch2b Layer
idx = net1.getLayerIndex('res4a_branch2b');
net.addLayer('res4a_branch2b_rgb',net1.layers(idx).block,{'res4a_branch2axxx_rgb'},{'res4a_branch2b_rgb'},{'res4a_branch2b_filter_rgb'});
net.addLayer('res4a_branch2b_d',net2.layers(idx).block,{'res4a_branch2axxx_d'},{'res4a_branch2b_d'},{'res4a_branch2b_filter_d'});

f1 = net.getParamIndex('res4a_branch2b_filter_rgb');
f2 = net.getParamIndex('res4a_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4a_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4a_branch2b_filter_d';

% bn4a_branch2b layer
idx = net1.getLayerIndex('bn4a_branch2b');
net.addLayer('bn4a_branch2b_rgb',net1.layers(idx).block,{'res4a_branch2b_rgb'},{'res4a_branch2bx_rgb'},{'bn4a_branch2b_mult_rgb','bn4a_branch2b_bias_rgb','bn4a_branch2b_moments_rgb'});
net.addLayer('bn4a_branch2b_d',net2.layers(idx).block,{'res4a_branch2b_d'},{'res4a_branch2bx_d'},{'bn4a_branch2b_mult_d','bn4a_branch2b_bias_d','bn4a_branch2b_moments_d'});


f1 = net.getParamIndex('bn4a_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4a_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4a_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4a_branch2b_mult_d');
f5 = net.getParamIndex('bn4a_branch2b_bias_d');
f6 = net.getParamIndex('bn4a_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4a_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4a_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4a_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4a_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4a_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4a_branch2b_moments_d';

% res4a_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4a_branch2b_relu_rgb', reluBlock_rgb, {'res4a_branch2bx_rgb'}, {'res4a_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4a_branch2b_relu_d', reluBlock_d, {'res4a_branch2bx_d'}, {'res4a_branch2bxxx_d'}, {}) ;

% res4a_branch2c Layer
idx = net1.getLayerIndex('res4a_branch2c');
net.addLayer('res4a_branch2c_rgb',net1.layers(idx).block,{'res4a_branch2bxxx_rgb'},{'res4a_branch2c_rgb'},{'res4a_branch2c_filter_rgb'});
net.addLayer('res4a_branch2c_d',net2.layers(idx).block,{'res4a_branch2bxxx_d'},{'res4a_branch2c_d'},{'res4a_branch2c_filter_d'});

f1 = net.getParamIndex('res4a_branch2c_filter_rgb');
f2 = net.getParamIndex('res4a_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4a_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4a_branch2c_filter_d';

% bn4a_branch2c layer
idx = net1.getLayerIndex('bn4a_branch2c');
net.addLayer('bn4a_branch2c_rgb',net1.layers(idx).block,{'res4a_branch2c_rgb'},{'res4a_branch2cx_rgb'},{'bn4a_branch2c_mult_rgb','bn4a_branch2c_bias_rgb','bn4a_branch2c_moments_rgb'});
net.addLayer('bn4a_branch2c_d',net2.layers(idx).block,{'res4a_branch2c_d'},{'res4a_branch2cx_d'},{'bn4a_branch2c_mult_d','bn4a_branch2c_bias_d','bn4a_branch2c_moments_d'});


f1 = net.getParamIndex('bn4a_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4a_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4a_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4a_branch2c_mult_d');
f5 = net.getParamIndex('bn4a_branch2c_bias_d');
f6 = net.getParamIndex('bn4a_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4a_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4a_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4a_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4a_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4a_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4a_branch2c_moments_d';

% res4a layer(sum)
idx = net1.getLayerIndex('res4a');
net.addLayer('res4a_rgb',net1.layers(idx).block,{'res4a_branch1x_rgb','res4a_branch2cx_rgb'},{'res4a_rgb'},{});
net.addLayer('res4a_d',net2.layers(idx).block,{'res4a_branch1x_d','res4a_branch2cx_d'},{'res4a_d'},{});

% res4a_relu layer
idx = net1.getLayerIndex('res4a_relu');
net.addLayer('res4a_relu_rgb',net1.layers(idx).block,{'res4a_rgb'},{'res4ax_rgb'},{});
net.addLayer('res4a_relu_d',net2.layers(idx).block,{'res4a_d'},{'res4ax_d'},{});




% res4b_branch2a layer
idx = net1.getLayerIndex('res4b_branch2a');
net.addLayer('res4b_branch2a_rgb',net1.layers(idx).block,{'res4ax_rgb'},{'res4b_branch2a_rgb'},{'res4b_branch2a_filter_rgb'});
net.addLayer('res4b_branch2a_d',net2.layers(idx).block,{'res4ax_d'},{'res4b_branch2a_d'},{'res4b_branch2a_filter_d'});

f1 = net.getParamIndex('res4b_branch2a_filter_rgb');
f2 = net.getParamIndex('res4b_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4b_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4b_branch2a_filter_d';

% bn4b_branch2a layer
idx = net1.getLayerIndex('bn4b_branch2a');
net.addLayer('bn4b_branch2a_rgb',net1.layers(idx).block,{'res4b_branch2a_rgb'},{'res4b_branch2ax_rgb'},{'bn4b_branch2a_mult_rgb','bn4b_branch2a_bias_rgb','bn4b_branch2a_moments_rgb'});
net.addLayer('bn4b_branch2a_d',net2.layers(idx).block,{'res4b_branch2a_d'},{'res4b_branch2ax_d'},{'bn4b_branch2a_mult_d','bn4b_branch2a_bias_d','bn4b_branch2a_moments_d'});


f1 = net.getParamIndex('bn4b_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4b_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4b_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4b_branch2a_mult_d');
f5 = net.getParamIndex('bn4b_branch2a_bias_d');
f6 = net.getParamIndex('bn4b_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4b_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4b_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4b_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4b_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4b_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4b_branch2a_moments_d';

% res4b_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4b_branch2a_relu_rgb', reluBlock_rgb, {'res4b_branch2ax_rgb'}, {'res4b_branch2axxx_rgb'}, {}) ;
net.addLayer('res4b_branch2a_relu_d', reluBlock_d, {'res4b_branch2ax_d'}, {'res4b_branch2axxx_d'}, {}) ;

% res4b_branch2b Layer
idx = net1.getLayerIndex('res4b_branch2b');
net.addLayer('res4b_branch2b_rgb',net1.layers(idx).block,{'res4b_branch2axxx_rgb'},{'res4b_branch2b_rgb'},{'res4b_branch2b_filter_rgb'});
net.addLayer('res4b_branch2b_d',net2.layers(idx).block,{'res4b_branch2axxx_d'},{'res4b_branch2b_d'},{'res4b_branch2b_filter_d'});

f1 = net.getParamIndex('res4b_branch2b_filter_rgb');
f2 = net.getParamIndex('res4b_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4b_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4b_branch2b_filter_d';

% bn4b_branch2b layer
idx = net1.getLayerIndex('bn4b_branch2b');
net.addLayer('bn4b_branch2b_rgb',net1.layers(idx).block,{'res4b_branch2b_rgb'},{'res4b_branch2bx_rgb'},{'bn4b_branch2b_mult_rgb','bn4b_branch2b_bias_rgb','bn4b_branch2b_moments_rgb'});
net.addLayer('bn4b_branch2b_d',net2.layers(idx).block,{'res4b_branch2b_d'},{'res4b_branch2bx_d'},{'bn4b_branch2b_mult_d','bn4b_branch2b_bias_d','bn4b_branch2b_moments_d'});


f1 = net.getParamIndex('bn4b_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4b_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4b_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4b_branch2b_mult_d');
f5 = net.getParamIndex('bn4b_branch2b_bias_d');
f6 = net.getParamIndex('bn4b_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4b_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4b_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4b_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4b_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4b_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4b_branch2b_moments_d';

% res4b_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4b_branch2b_relu_rgb', reluBlock_rgb, {'res4b_branch2bx_rgb'}, {'res4b_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4b_branch2b_relu_d', reluBlock_d, {'res4b_branch2bx_d'}, {'res4b_branch2bxxx_d'}, {}) ;

% res4b_branch2c Layer
idx = net1.getLayerIndex('res4b_branch2c');
net.addLayer('res4b_branch2c_rgb',net1.layers(idx).block,{'res4b_branch2bxxx_rgb'},{'res4b_branch2c_rgb'},{'res4b_branch2c_filter_rgb'});
net.addLayer('res4b_branch2c_d',net2.layers(idx).block,{'res4b_branch2bxxx_d'},{'res4b_branch2c_d'},{'res4b_branch2c_filter_d'});

f1 = net.getParamIndex('res4b_branch2c_filter_rgb');
f2 = net.getParamIndex('res4b_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4b_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4b_branch2c_filter_d';

% bn4b_branch2c layer
idx = net1.getLayerIndex('bn4b_branch2c');
net.addLayer('bn4b_branch2c_rgb',net1.layers(idx).block,{'res4b_branch2c_rgb'},{'res4b_branch2cx_rgb'},{'bn4b_branch2c_mult_rgb','bn4b_branch2c_bias_rgb','bn4b_branch2c_moments_rgb'});
net.addLayer('bn4b_branch2c_d',net2.layers(idx).block,{'res4b_branch2c_d'},{'res4b_branch2cx_d'},{'bn4b_branch2c_mult_d','bn4b_branch2c_bias_d','bn4b_branch2c_moments_d'});


f1 = net.getParamIndex('bn4b_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4b_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4b_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4b_branch2c_mult_d');
f5 = net.getParamIndex('bn4b_branch2c_bias_d');
f6 = net.getParamIndex('bn4b_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4b_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4b_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4b_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4b_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4b_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4b_branch2c_moments_d';

% res4b layer(sum)
idx = net1.getLayerIndex('res4b');
net.addLayer('res4b_rgb',net1.layers(idx).block,{'res4ax_rgb','res4b_branch2cx_rgb'},{'res4b_rgb'},{});
net.addLayer('res4b_d',net2.layers(idx).block,{'res4ax_d','res4b_branch2cx_d'},{'res4b_d'},{});

% res4b_relu layer
idx = net1.getLayerIndex('res4b_relu');
net.addLayer('res4b_relu_rgb',net1.layers(idx).block,{'res4b_rgb'},{'res4bx_rgb'},{});
net.addLayer('res4b_relu_d',net2.layers(idx).block,{'res4b_d'},{'res4bx_d'},{});




% res4c_branch2a layer
idx = net1.getLayerIndex('res4c_branch2a');
net.addLayer('res4c_branch2a_rgb',net1.layers(idx).block,{'res4bx_rgb'},{'res4c_branch2a_rgb'},{'res4c_branch2a_filter_rgb'});
net.addLayer('res4c_branch2a_d',net2.layers(idx).block,{'res4bx_d'},{'res4c_branch2a_d'},{'res4c_branch2a_filter_d'});

f1 = net.getParamIndex('res4c_branch2a_filter_rgb');
f2 = net.getParamIndex('res4c_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4c_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4c_branch2a_filter_d';

% bn4c_branch2a layer
idx = net1.getLayerIndex('bn4c_branch2a');
net.addLayer('bn4c_branch2a_rgb',net1.layers(idx).block,{'res4c_branch2a_rgb'},{'res4c_branch2ax_rgb'},{'bn4c_branch2a_mult_rgb','bn4c_branch2a_bias_rgb','bn4c_branch2a_moments_rgb'});
net.addLayer('bn4c_branch2a_d',net2.layers(idx).block,{'res4c_branch2a_d'},{'res4c_branch2ax_d'},{'bn4c_branch2a_mult_d','bn4c_branch2a_bias_d','bn4c_branch2a_moments_d'});


f1 = net.getParamIndex('bn4c_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4c_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4c_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4c_branch2a_mult_d');
f5 = net.getParamIndex('bn4c_branch2a_bias_d');
f6 = net.getParamIndex('bn4c_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4c_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4c_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4c_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4c_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4c_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4c_branch2a_moments_d';

% res4c_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4c_branch2a_relu_rgb', reluBlock_rgb, {'res4c_branch2ax_rgb'}, {'res4c_branch2axxx_rgb'}, {}) ;
net.addLayer('res4c_branch2a_relu_d', reluBlock_d, {'res4c_branch2ax_d'}, {'res4c_branch2axxx_d'}, {}) ;

% res4c_branch2b Layer
idx = net1.getLayerIndex('res4c_branch2b');
net.addLayer('res4c_branch2b_rgb',net1.layers(idx).block,{'res4c_branch2axxx_rgb'},{'res4c_branch2b_rgb'},{'res4c_branch2b_filter_rgb'});
net.addLayer('res4c_branch2b_d',net2.layers(idx).block,{'res4c_branch2axxx_d'},{'res4c_branch2b_d'},{'res4c_branch2b_filter_d'});

f1 = net.getParamIndex('res4c_branch2b_filter_rgb');
f2 = net.getParamIndex('res4c_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4c_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4c_branch2b_filter_d';

% bn4c_branch2b layer
idx = net1.getLayerIndex('bn4c_branch2b');
net.addLayer('bn4c_branch2b_rgb',net1.layers(idx).block,{'res4c_branch2b_rgb'},{'res4c_branch2bx_rgb'},{'bn4c_branch2b_mult_rgb','bn4c_branch2b_bias_rgb','bn4c_branch2b_moments_rgb'});
net.addLayer('bn4c_branch2b_d',net2.layers(idx).block,{'res4c_branch2b_d'},{'res4c_branch2bx_d'},{'bn4c_branch2b_mult_d','bn4c_branch2b_bias_d','bn4c_branch2b_moments_d'});


f1 = net.getParamIndex('bn4c_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4c_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4c_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4c_branch2b_mult_d');
f5 = net.getParamIndex('bn4c_branch2b_bias_d');
f6 = net.getParamIndex('bn4c_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4c_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4c_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4c_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4c_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4c_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4c_branch2b_moments_d';

% res4c_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4c_branch2b_relu_rgb', reluBlock_rgb, {'res4c_branch2bx_rgb'}, {'res4c_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4c_branch2b_relu_d', reluBlock_d, {'res4c_branch2bx_d'}, {'res4c_branch2bxxx_d'}, {}) ;

% res4c_branch2c Layer
idx = net1.getLayerIndex('res4c_branch2c');
net.addLayer('res4c_branch2c_rgb',net1.layers(idx).block,{'res4c_branch2bxxx_rgb'},{'res4c_branch2c_rgb'},{'res4c_branch2c_filter_rgb'});
net.addLayer('res4c_branch2c_d',net2.layers(idx).block,{'res4c_branch2bxxx_d'},{'res4c_branch2c_d'},{'res4c_branch2c_filter_d'});

f1 = net.getParamIndex('res4c_branch2c_filter_rgb');
f2 = net.getParamIndex('res4c_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4c_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4c_branch2c_filter_d';

% bn4c_branch2c layer
idx = net1.getLayerIndex('bn4c_branch2c');
net.addLayer('bn4c_branch2c_rgb',net1.layers(idx).block,{'res4c_branch2c_rgb'},{'res4c_branch2cx_rgb'},{'bn4c_branch2c_mult_rgb','bn4c_branch2c_bias_rgb','bn4c_branch2c_moments_rgb'});
net.addLayer('bn4c_branch2c_d',net2.layers(idx).block,{'res4c_branch2c_d'},{'res4c_branch2cx_d'},{'bn4c_branch2c_mult_d','bn4c_branch2c_bias_d','bn4c_branch2c_moments_d'});


f1 = net.getParamIndex('bn4c_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4c_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4c_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4c_branch2c_mult_d');
f5 = net.getParamIndex('bn4c_branch2c_bias_d');
f6 = net.getParamIndex('bn4c_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4c_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4c_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4c_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4c_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4c_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4c_branch2c_moments_d';

% res4c layer(sum)
idx = net1.getLayerIndex('res4c');
net.addLayer('res4c_rgb',net1.layers(idx).block,{'res4bx_rgb','res4c_branch2cx_rgb'},{'res4c_rgb'},{});
net.addLayer('res4c_d',net2.layers(idx).block,{'res4bx_d','res4c_branch2cx_d'},{'res4c_d'},{});

% res4c_relu layer
idx = net1.getLayerIndex('res4c_relu');
net.addLayer('res4c_relu_rgb',net1.layers(idx).block,{'res4c_rgb'},{'res4cx_rgb'},{});
net.addLayer('res4c_relu_d',net2.layers(idx).block,{'res4c_d'},{'res4cx_d'},{});





% res4d_branch2a layer
idx = net1.getLayerIndex('res4d_branch2a');
net.addLayer('res4d_branch2a_rgb',net1.layers(idx).block,{'res4cx_rgb'},{'res4d_branch2a_rgb'},{'res4d_branch2a_filter_rgb'});
net.addLayer('res4d_branch2a_d',net2.layers(idx).block,{'res4cx_d'},{'res4d_branch2a_d'},{'res4d_branch2a_filter_d'});

f1 = net.getParamIndex('res4d_branch2a_filter_rgb');
f2 = net.getParamIndex('res4d_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4d_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4d_branch2a_filter_d';

% bn4d_branch2a layer
idx = net1.getLayerIndex('bn4d_branch2a');
net.addLayer('bn4d_branch2a_rgb',net1.layers(idx).block,{'res4d_branch2a_rgb'},{'res4d_branch2ax_rgb'},{'bn4d_branch2a_mult_rgb','bn4d_branch2a_bias_rgb','bn4d_branch2a_moments_rgb'});
net.addLayer('bn4d_branch2a_d',net2.layers(idx).block,{'res4d_branch2a_d'},{'res4d_branch2ax_d'},{'bn4d_branch2a_mult_d','bn4d_branch2a_bias_d','bn4d_branch2a_moments_d'});


f1 = net.getParamIndex('bn4d_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4d_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4d_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4d_branch2a_mult_d');
f5 = net.getParamIndex('bn4d_branch2a_bias_d');
f6 = net.getParamIndex('bn4d_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4d_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4d_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4d_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4d_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4d_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4d_branch2a_moments_d';

% res4d_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4d_branch2a_relu_rgb', reluBlock_rgb, {'res4d_branch2ax_rgb'}, {'res4d_branch2axxx_rgb'}, {}) ;
net.addLayer('res4d_branch2a_relu_d', reluBlock_d, {'res4d_branch2ax_d'}, {'res4d_branch2axxx_d'}, {}) ;

% res4d_branch2b Layer
idx = net1.getLayerIndex('res4d_branch2b');
net.addLayer('res4d_branch2b_rgb',net1.layers(idx).block,{'res4d_branch2axxx_rgb'},{'res4d_branch2b_rgb'},{'res4d_branch2b_filter_rgb'});
net.addLayer('res4d_branch2b_d',net2.layers(idx).block,{'res4d_branch2axxx_d'},{'res4d_branch2b_d'},{'res4d_branch2b_filter_d'});

f1 = net.getParamIndex('res4d_branch2b_filter_rgb');
f2 = net.getParamIndex('res4d_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4d_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4d_branch2b_filter_d';

% bn4d_branch2b layer
idx = net1.getLayerIndex('bn4d_branch2b');
net.addLayer('bn4d_branch2b_rgb',net1.layers(idx).block,{'res4d_branch2b_rgb'},{'res4d_branch2bx_rgb'},{'bn4d_branch2b_mult_rgb','bn4d_branch2b_bias_rgb','bn4d_branch2b_moments_rgb'});
net.addLayer('bn4d_branch2b_d',net2.layers(idx).block,{'res4d_branch2b_d'},{'res4d_branch2bx_d'},{'bn4d_branch2b_mult_d','bn4d_branch2b_bias_d','bn4d_branch2b_moments_d'});


f1 = net.getParamIndex('bn4d_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4d_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4d_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4d_branch2b_mult_d');
f5 = net.getParamIndex('bn4d_branch2b_bias_d');
f6 = net.getParamIndex('bn4d_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4d_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4d_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4d_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4d_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4d_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4d_branch2b_moments_d';

% res4d_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4d_branch2b_relu_rgb', reluBlock_rgb, {'res4d_branch2bx_rgb'}, {'res4d_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4d_branch2b_relu_d', reluBlock_d, {'res4d_branch2bx_d'}, {'res4d_branch2bxxx_d'}, {}) ;

% res4d_branch2c Layer
idx = net1.getLayerIndex('res4d_branch2c');
net.addLayer('res4d_branch2c_rgb',net1.layers(idx).block,{'res4d_branch2bxxx_rgb'},{'res4d_branch2c_rgb'},{'res4d_branch2c_filter_rgb'});
net.addLayer('res4d_branch2c_d',net2.layers(idx).block,{'res4d_branch2bxxx_d'},{'res4d_branch2c_d'},{'res4d_branch2c_filter_d'});

f1 = net.getParamIndex('res4d_branch2c_filter_rgb');
f2 = net.getParamIndex('res4d_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4d_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4d_branch2c_filter_d';

% bn4d_branch2c layer
idx = net1.getLayerIndex('bn4d_branch2c');
net.addLayer('bn4d_branch2c_rgb',net1.layers(idx).block,{'res4d_branch2c_rgb'},{'res4d_branch2cx_rgb'},{'bn4d_branch2c_mult_rgb','bn4d_branch2c_bias_rgb','bn4d_branch2c_moments_rgb'});
net.addLayer('bn4d_branch2c_d',net2.layers(idx).block,{'res4d_branch2c_d'},{'res4d_branch2cx_d'},{'bn4d_branch2c_mult_d','bn4d_branch2c_bias_d','bn4d_branch2c_moments_d'});


f1 = net.getParamIndex('bn4d_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4d_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4d_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4d_branch2c_mult_d');
f5 = net.getParamIndex('bn4d_branch2c_bias_d');
f6 = net.getParamIndex('bn4d_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4d_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4d_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4d_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4d_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4d_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4d_branch2c_moments_d';

% res4d layer(sum)
idx = net1.getLayerIndex('res4d');
net.addLayer('res4d_rgb',net1.layers(idx).block,{'res4cx_rgb','res4d_branch2cx_rgb'},{'res4d_rgb'},{});
net.addLayer('res4d_d',net2.layers(idx).block,{'res4cx_d','res4d_branch2cx_d'},{'res4d_d'},{});

% res4d_relu layer
idx = net1.getLayerIndex('res4d_relu');
net.addLayer('res4d_relu_rgb',net1.layers(idx).block,{'res4d_rgb'},{'res4dx_rgb'},{});
net.addLayer('res4d_relu_d',net2.layers(idx).block,{'res4d_d'},{'res4dx_d'},{});





% res4e_branch2a layer
idx = net1.getLayerIndex('res4e_branch2a');
net.addLayer('res4e_branch2a_rgb',net1.layers(idx).block,{'res4dx_rgb'},{'res4e_branch2a_rgb'},{'res4e_branch2a_filter_rgb'});
net.addLayer('res4e_branch2a_d',net2.layers(idx).block,{'res4dx_d'},{'res4e_branch2a_d'},{'res4e_branch2a_filter_d'});

f1 = net.getParamIndex('res4e_branch2a_filter_rgb');
f2 = net.getParamIndex('res4e_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4e_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4e_branch2a_filter_d';

% bn4e_branch2a layer
idx = net1.getLayerIndex('bn4e_branch2a');
net.addLayer('bn4e_branch2a_rgb',net1.layers(idx).block,{'res4e_branch2a_rgb'},{'res4e_branch2ax_rgb'},{'bn4e_branch2a_mult_rgb','bn4e_branch2a_bias_rgb','bn4e_branch2a_moments_rgb'});
net.addLayer('bn4e_branch2a_d',net2.layers(idx).block,{'res4e_branch2a_d'},{'res4e_branch2ax_d'},{'bn4e_branch2a_mult_d','bn4e_branch2a_bias_d','bn4e_branch2a_moments_d'});


f1 = net.getParamIndex('bn4e_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4e_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4e_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4e_branch2a_mult_d');
f5 = net.getParamIndex('bn4e_branch2a_bias_d');
f6 = net.getParamIndex('bn4e_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4e_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4e_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4e_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4e_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4e_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4e_branch2a_moments_d';

% res4e_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4e_branch2a_relu_rgb', reluBlock_rgb, {'res4e_branch2ax_rgb'}, {'res4e_branch2axxx_rgb'}, {}) ;
net.addLayer('res4e_branch2a_relu_d', reluBlock_d, {'res4e_branch2ax_d'}, {'res4e_branch2axxx_d'}, {}) ;

% res4e_branch2b Layer
idx = net1.getLayerIndex('res4e_branch2b');
net.addLayer('res4e_branch2b_rgb',net1.layers(idx).block,{'res4e_branch2axxx_rgb'},{'res4e_branch2b_rgb'},{'res4e_branch2b_filter_rgb'});
net.addLayer('res4e_branch2b_d',net2.layers(idx).block,{'res4e_branch2axxx_d'},{'res4e_branch2b_d'},{'res4e_branch2b_filter_d'});

f1 = net.getParamIndex('res4e_branch2b_filter_rgb');
f2 = net.getParamIndex('res4e_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4e_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4e_branch2b_filter_d';

% bn4e_branch2b layer
idx = net1.getLayerIndex('bn4e_branch2b');
net.addLayer('bn4e_branch2b_rgb',net1.layers(idx).block,{'res4e_branch2b_rgb'},{'res4e_branch2bx_rgb'},{'bn4e_branch2b_mult_rgb','bn4e_branch2b_bias_rgb','bn4e_branch2b_moments_rgb'});
net.addLayer('bn4e_branch2b_d',net2.layers(idx).block,{'res4e_branch2b_d'},{'res4e_branch2bx_d'},{'bn4e_branch2b_mult_d','bn4e_branch2b_bias_d','bn4e_branch2b_moments_d'});


f1 = net.getParamIndex('bn4e_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4e_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4e_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4e_branch2b_mult_d');
f5 = net.getParamIndex('bn4e_branch2b_bias_d');
f6 = net.getParamIndex('bn4e_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4e_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4e_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4e_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4e_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4e_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4e_branch2b_moments_d';

% res4e_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4e_branch2b_relu_rgb', reluBlock_rgb, {'res4e_branch2bx_rgb'}, {'res4e_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4e_branch2b_relu_d', reluBlock_d, {'res4e_branch2bx_d'}, {'res4e_branch2bxxx_d'}, {}) ;

% res4e_branch2c Layer
idx = net1.getLayerIndex('res4e_branch2c');
net.addLayer('res4e_branch2c_rgb',net1.layers(idx).block,{'res4e_branch2bxxx_rgb'},{'res4e_branch2c_rgb'},{'res4e_branch2c_filter_rgb'});
net.addLayer('res4e_branch2c_d',net2.layers(idx).block,{'res4e_branch2bxxx_d'},{'res4e_branch2c_d'},{'res4e_branch2c_filter_d'});

f1 = net.getParamIndex('res4e_branch2c_filter_rgb');
f2 = net.getParamIndex('res4e_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4e_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4e_branch2c_filter_d';

% bn4e_branch2c layer
idx = net1.getLayerIndex('bn4e_branch2c');
net.addLayer('bn4e_branch2c_rgb',net1.layers(idx).block,{'res4e_branch2c_rgb'},{'res4e_branch2cx_rgb'},{'bn4e_branch2c_mult_rgb','bn4e_branch2c_bias_rgb','bn4e_branch2c_moments_rgb'});
net.addLayer('bn4e_branch2c_d',net2.layers(idx).block,{'res4e_branch2c_d'},{'res4e_branch2cx_d'},{'bn4e_branch2c_mult_d','bn4e_branch2c_bias_d','bn4e_branch2c_moments_d'});


f1 = net.getParamIndex('bn4e_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4e_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4e_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4e_branch2c_mult_d');
f5 = net.getParamIndex('bn4e_branch2c_bias_d');
f6 = net.getParamIndex('bn4e_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4e_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4e_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4e_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4e_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4e_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4e_branch2c_moments_d';

% res4e layer(sum)
idx = net1.getLayerIndex('res4e');
net.addLayer('res4e_rgb',net1.layers(idx).block,{'res4dx_rgb','res4e_branch2cx_rgb'},{'res4e_rgb'},{});
net.addLayer('res4e_d',net2.layers(idx).block,{'res4dx_d','res4e_branch2cx_d'},{'res4e_d'},{});

% res4e_relu layer
idx = net1.getLayerIndex('res4e_relu');
net.addLayer('res4e_relu_rgb',net1.layers(idx).block,{'res4e_rgb'},{'res4ex_rgb'},{});
net.addLayer('res4e_relu_d',net2.layers(idx).block,{'res4e_d'},{'res4ex_d'},{});






% res4f_branch2a layer
idx = net1.getLayerIndex('res4f_branch2a');
net.addLayer('res4f_branch2a_rgb',net1.layers(idx).block,{'res4ex_rgb'},{'res4f_branch2a_rgb'},{'res4f_branch2a_filter_rgb'});
net.addLayer('res4f_branch2a_d',net2.layers(idx).block,{'res4ex_d'},{'res4f_branch2a_d'},{'res4f_branch2a_filter_d'});

f1 = net.getParamIndex('res4f_branch2a_filter_rgb');
f2 = net.getParamIndex('res4f_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4f_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4f_branch2a_filter_d';

% bn4f_branch2a layer
idx = net1.getLayerIndex('bn4f_branch2a');
net.addLayer('bn4f_branch2a_rgb',net1.layers(idx).block,{'res4f_branch2a_rgb'},{'res4f_branch2ax_rgb'},{'bn4f_branch2a_mult_rgb','bn4f_branch2a_bias_rgb','bn4f_branch2a_moments_rgb'});
net.addLayer('bn4f_branch2a_d',net2.layers(idx).block,{'res4f_branch2a_d'},{'res4f_branch2ax_d'},{'bn4f_branch2a_mult_d','bn4f_branch2a_bias_d','bn4f_branch2a_moments_d'});


f1 = net.getParamIndex('bn4f_branch2a_mult_rgb');
f2 = net.getParamIndex('bn4f_branch2a_bias_rgb');
f3 = net.getParamIndex('bn4f_branch2a_moments_rgb');
f4 = net.getParamIndex('bn4f_branch2a_mult_d');
f5 = net.getParamIndex('bn4f_branch2a_bias_d');
f6 = net.getParamIndex('bn4f_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4f_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4f_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4f_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4f_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4f_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4f_branch2a_moments_d';

% res4f_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4f_branch2a_relu_rgb', reluBlock_rgb, {'res4f_branch2ax_rgb'}, {'res4f_branch2axxx_rgb'}, {}) ;
net.addLayer('res4f_branch2a_relu_d', reluBlock_d, {'res4f_branch2ax_d'}, {'res4f_branch2axxx_d'}, {}) ;

% res4f_branch2b Layer
idx = net1.getLayerIndex('res4f_branch2b');
net.addLayer('res4f_branch2b_rgb',net1.layers(idx).block,{'res4f_branch2axxx_rgb'},{'res4f_branch2b_rgb'},{'res4f_branch2b_filter_rgb'});
net.addLayer('res4f_branch2b_d',net2.layers(idx).block,{'res4f_branch2axxx_d'},{'res4f_branch2b_d'},{'res4f_branch2b_filter_d'});

f1 = net.getParamIndex('res4f_branch2b_filter_rgb');
f2 = net.getParamIndex('res4f_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4f_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4f_branch2b_filter_d';

% bn4f_branch2b layer
idx = net1.getLayerIndex('bn4f_branch2b');
net.addLayer('bn4f_branch2b_rgb',net1.layers(idx).block,{'res4f_branch2b_rgb'},{'res4f_branch2bx_rgb'},{'bn4f_branch2b_mult_rgb','bn4f_branch2b_bias_rgb','bn4f_branch2b_moments_rgb'});
net.addLayer('bn4f_branch2b_d',net2.layers(idx).block,{'res4f_branch2b_d'},{'res4f_branch2bx_d'},{'bn4f_branch2b_mult_d','bn4f_branch2b_bias_d','bn4f_branch2b_moments_d'});


f1 = net.getParamIndex('bn4f_branch2b_mult_rgb');
f2 = net.getParamIndex('bn4f_branch2b_bias_rgb');
f3 = net.getParamIndex('bn4f_branch2b_moments_rgb');
f4 = net.getParamIndex('bn4f_branch2b_mult_d');
f5 = net.getParamIndex('bn4f_branch2b_bias_d');
f6 = net.getParamIndex('bn4f_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4f_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4f_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4f_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4f_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4f_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4f_branch2b_moments_d';

% res4f_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res4f_branch2b_relu_rgb', reluBlock_rgb, {'res4f_branch2bx_rgb'}, {'res4f_branch2bxxx_rgb'}, {}) ;
net.addLayer('res4f_branch2b_relu_d', reluBlock_d, {'res4f_branch2bx_d'}, {'res4f_branch2bxxx_d'}, {}) ;

% res4f_branch2c Layer
idx = net1.getLayerIndex('res4f_branch2c');
net.addLayer('res4f_branch2c_rgb',net1.layers(idx).block,{'res4f_branch2bxxx_rgb'},{'res4f_branch2c_rgb'},{'res4f_branch2c_filter_rgb'});
net.addLayer('res4f_branch2c_d',net2.layers(idx).block,{'res4f_branch2bxxx_d'},{'res4f_branch2c_d'},{'res4f_branch2c_filter_d'});

f1 = net.getParamIndex('res4f_branch2c_filter_rgb');
f2 = net.getParamIndex('res4f_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res4f_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res4f_branch2c_filter_d';

% bn4f_branch2c layer
idx = net1.getLayerIndex('bn4f_branch2c');
net.addLayer('bn4f_branch2c_rgb',net1.layers(idx).block,{'res4f_branch2c_rgb'},{'res4f_branch2cx_rgb'},{'bn4f_branch2c_mult_rgb','bn4f_branch2c_bias_rgb','bn4f_branch2c_moments_rgb'});
net.addLayer('bn4f_branch2c_d',net2.layers(idx).block,{'res4f_branch2c_d'},{'res4f_branch2cx_d'},{'bn4f_branch2c_mult_d','bn4f_branch2c_bias_d','bn4f_branch2c_moments_d'});


f1 = net.getParamIndex('bn4f_branch2c_mult_rgb');
f2 = net.getParamIndex('bn4f_branch2c_bias_rgb');
f3 = net.getParamIndex('bn4f_branch2c_moments_rgb');
f4 = net.getParamIndex('bn4f_branch2c_mult_d');
f5 = net.getParamIndex('bn4f_branch2c_bias_d');
f6 = net.getParamIndex('bn4f_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn4f_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn4f_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn4f_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn4f_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn4f_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn4f_branch2c_moments_d';

% res4f layer(sum)
idx = net1.getLayerIndex('res4f');
net.addLayer('res4f_rgb',net1.layers(idx).block,{'res4ex_rgb','res4f_branch2cx_rgb'},{'res4f_rgb'},{});
net.addLayer('res4f_d',net2.layers(idx).block,{'res4ex_d','res4f_branch2cx_d'},{'res4f_d'},{});

% res4f_relu layer
idx = net1.getLayerIndex('res4f_relu');
net.addLayer('res4f_relu_rgb',net1.layers(idx).block,{'res4f_rgb'},{'res4fx_rgb'},{});
net.addLayer('res4f_relu_d',net2.layers(idx).block,{'res4f_d'},{'res4fx_d'},{});









% res5a_branch1 layer
idx = net1.getLayerIndex('res5a_branch1');
net.addLayer('res5a_branch1_rgb',net1.layers(idx).block,{'res4fx_rgb'},{'res5a_branch1_rgb'},{'res5a_branch1_filter_rgb'});
net.addLayer('res5a_branch1_d',net2.layers(idx).block,{'res4fx_d'},{'res5a_branch1_d'},{'res5a_branch1_filter_d'});

f1 = net.getParamIndex('res5a_branch1_filter_rgb');
f2 = net.getParamIndex('res5a_branch1_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5a_branch1_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5a_branch1_filter_d';

% bn5a_branch1 Layer
idx = net1.getLayerIndex('bn5a_branch1');
net.addLayer('bn5a_branch1_rgb',net1.layers(idx).block,{'res5a_branch1_rgb'},{'res5a_branch1x_rgb'},{'bn5a_branch1_mult_rgb','bn5a_branch1_bias_rgb','bn5a_branch1_moments_rgb'});
net.addLayer('bn5a_branch1_d',net2.layers(idx).block,{'res5a_branch1_d'},{'res5a_branch1x_d'},{'bn5a_branch1_mult_d','bn5a_branch1_bias_d','bn5a_branch1_moments_d'});


f1 = net.getParamIndex('bn5a_branch1_mult_rgb');
f2 = net.getParamIndex('bn5a_branch1_bias_rgb');
f3 = net.getParamIndex('bn5a_branch1_moments_rgb');
f4 = net.getParamIndex('bn5a_branch1_mult_d');
f5 = net.getParamIndex('bn5a_branch1_bias_d');
f6 = net.getParamIndex('bn5a_branch1_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5a_branch1_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5a_branch1_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5a_branch1_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5a_branch1_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5a_branch1_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5a_branch1_moments_d';

% res5a_branch2a layer
idx = net1.getLayerIndex('res5a_branch2a');
net.addLayer('res5a_branch2a_rgb',net1.layers(idx).block,{'res4fx_rgb'},{'res5a_branch2a_rgb'},{'res5a_branch2a_filter_rgb'});
net.addLayer('res5a_branch2a_d',net2.layers(idx).block,{'res4fx_d'},{'res5a_branch2a_d'},{'res5a_branch2a_filter_d'});

f1 = net.getParamIndex('res5a_branch2a_filter_rgb');
f2 = net.getParamIndex('res5a_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5a_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5a_branch2a_filter_d';

% bn5a_branch2a layer
idx = net1.getLayerIndex('bn5a_branch2a');
net.addLayer('bn5a_branch2a_rgb',net1.layers(idx).block,{'res5a_branch2a_rgb'},{'res5a_branch2ax_rgb'},{'bn5a_branch2a_mult_rgb','bn5a_branch2a_bias_rgb','bn5a_branch2a_moments_rgb'});
net.addLayer('bn5a_branch2a_d',net2.layers(idx).block,{'res5a_branch2a_d'},{'res5a_branch2ax_d'},{'bn5a_branch2a_mult_d','bn5a_branch2a_bias_d','bn5a_branch2a_moments_d'});


f1 = net.getParamIndex('bn5a_branch2a_mult_rgb');
f2 = net.getParamIndex('bn5a_branch2a_bias_rgb');
f3 = net.getParamIndex('bn5a_branch2a_moments_rgb');
f4 = net.getParamIndex('bn5a_branch2a_mult_d');
f5 = net.getParamIndex('bn5a_branch2a_bias_d');
f6 = net.getParamIndex('bn5a_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5a_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5a_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5a_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5a_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5a_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5a_branch2a_moments_d';

% res5a_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5a_branch2a_relu_rgb', reluBlock_rgb, {'res5a_branch2ax_rgb'}, {'res5a_branch2axxx_rgb'}, {}) ;
net.addLayer('res5a_branch2a_relu_d', reluBlock_d, {'res5a_branch2ax_d'}, {'res5a_branch2axxx_d'}, {}) ;

% res5a_branch2b Layer
idx = net1.getLayerIndex('res5a_branch2b');
net.addLayer('res5a_branch2b_rgb',net1.layers(idx).block,{'res5a_branch2axxx_rgb'},{'res5a_branch2b_rgb'},{'res5a_branch2b_filter_rgb'});
net.addLayer('res5a_branch2b_d',net2.layers(idx).block,{'res5a_branch2axxx_d'},{'res5a_branch2b_d'},{'res5a_branch2b_filter_d'});

f1 = net.getParamIndex('res5a_branch2b_filter_rgb');
f2 = net.getParamIndex('res5a_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5a_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5a_branch2b_filter_d';

% bn5a_branch2b layer
idx = net1.getLayerIndex('bn5a_branch2b');
net.addLayer('bn5a_branch2b_rgb',net1.layers(idx).block,{'res5a_branch2b_rgb'},{'res5a_branch2bx_rgb'},{'bn5a_branch2b_mult_rgb','bn5a_branch2b_bias_rgb','bn5a_branch2b_moments_rgb'});
net.addLayer('bn5a_branch2b_d',net2.layers(idx).block,{'res5a_branch2b_d'},{'res5a_branch2bx_d'},{'bn5a_branch2b_mult_d','bn5a_branch2b_bias_d','bn5a_branch2b_moments_d'});


f1 = net.getParamIndex('bn5a_branch2b_mult_rgb');
f2 = net.getParamIndex('bn5a_branch2b_bias_rgb');
f3 = net.getParamIndex('bn5a_branch2b_moments_rgb');
f4 = net.getParamIndex('bn5a_branch2b_mult_d');
f5 = net.getParamIndex('bn5a_branch2b_bias_d');
f6 = net.getParamIndex('bn5a_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5a_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5a_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5a_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5a_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5a_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5a_branch2b_moments_d';

% res5a_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5a_branch2b_relu_rgb', reluBlock_rgb, {'res5a_branch2bx_rgb'}, {'res5a_branch2bxxx_rgb'}, {}) ;
net.addLayer('res5a_branch2b_relu_d', reluBlock_d, {'res5a_branch2bx_d'}, {'res5a_branch2bxxx_d'}, {}) ;

% res5a_branch2c Layer
idx = net1.getLayerIndex('res5a_branch2c');
net.addLayer('res5a_branch2c_rgb',net1.layers(idx).block,{'res5a_branch2bxxx_rgb'},{'res5a_branch2c_rgb'},{'res5a_branch2c_filter_rgb'});
net.addLayer('res5a_branch2c_d',net2.layers(idx).block,{'res5a_branch2bxxx_d'},{'res5a_branch2c_d'},{'res5a_branch2c_filter_d'});

f1 = net.getParamIndex('res5a_branch2c_filter_rgb');
f2 = net.getParamIndex('res5a_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5a_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5a_branch2c_filter_d';

% bn5a_branch2c layer
idx = net1.getLayerIndex('bn5a_branch2c');
net.addLayer('bn5a_branch2c_rgb',net1.layers(idx).block,{'res5a_branch2c_rgb'},{'res5a_branch2cx_rgb'},{'bn5a_branch2c_mult_rgb','bn5a_branch2c_bias_rgb','bn5a_branch2c_moments_rgb'});
net.addLayer('bn5a_branch2c_d',net2.layers(idx).block,{'res5a_branch2c_d'},{'res5a_branch2cx_d'},{'bn5a_branch2c_mult_d','bn5a_branch2c_bias_d','bn5a_branch2c_moments_d'});


f1 = net.getParamIndex('bn5a_branch2c_mult_rgb');
f2 = net.getParamIndex('bn5a_branch2c_bias_rgb');
f3 = net.getParamIndex('bn5a_branch2c_moments_rgb');
f4 = net.getParamIndex('bn5a_branch2c_mult_d');
f5 = net.getParamIndex('bn5a_branch2c_bias_d');
f6 = net.getParamIndex('bn5a_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5a_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5a_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5a_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5a_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5a_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5a_branch2c_moments_d';

% res5a layer(sum)
idx = net1.getLayerIndex('res5a');
net.addLayer('res5a_rgb',net1.layers(idx).block,{'res5a_branch1x_rgb','res5a_branch2cx_rgb'},{'res5a_rgb'},{});
net.addLayer('res5a_d',net2.layers(idx).block,{'res5a_branch1x_d','res5a_branch2cx_d'},{'res5a_d'},{});

% res5a_relu layer
idx = net1.getLayerIndex('res5a_relu');
net.addLayer('res5a_relu_rgb',net1.layers(idx).block,{'res5a_rgb'},{'res5ax_rgb'},{});
net.addLayer('res5a_relu_d',net2.layers(idx).block,{'res5a_d'},{'res5ax_d'},{});




% res5b_branch2a layer
idx = net1.getLayerIndex('res5b_branch2a');
net.addLayer('res5b_branch2a_rgb',net1.layers(idx).block,{'res5ax_rgb'},{'res5b_branch2a_rgb'},{'res5b_branch2a_filter_rgb'});
net.addLayer('res5b_branch2a_d',net2.layers(idx).block,{'res5ax_d'},{'res5b_branch2a_d'},{'res5b_branch2a_filter_d'});

f1 = net.getParamIndex('res5b_branch2a_filter_rgb');
f2 = net.getParamIndex('res5b_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5b_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5b_branch2a_filter_d';

% bn5b_branch2a layer
idx = net1.getLayerIndex('bn5b_branch2a');
net.addLayer('bn5b_branch2a_rgb',net1.layers(idx).block,{'res5b_branch2a_rgb'},{'res5b_branch2ax_rgb'},{'bn5b_branch2a_mult_rgb','bn5b_branch2a_bias_rgb','bn5b_branch2a_moments_rgb'});
net.addLayer('bn5b_branch2a_d',net2.layers(idx).block,{'res5b_branch2a_d'},{'res5b_branch2ax_d'},{'bn5b_branch2a_mult_d','bn5b_branch2a_bias_d','bn5b_branch2a_moments_d'});


f1 = net.getParamIndex('bn5b_branch2a_mult_rgb');
f2 = net.getParamIndex('bn5b_branch2a_bias_rgb');
f3 = net.getParamIndex('bn5b_branch2a_moments_rgb');
f4 = net.getParamIndex('bn5b_branch2a_mult_d');
f5 = net.getParamIndex('bn5b_branch2a_bias_d');
f6 = net.getParamIndex('bn5b_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5b_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5b_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5b_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5b_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5b_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5b_branch2a_moments_d';

% res5b_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5b_branch2a_relu_rgb', reluBlock_rgb, {'res5b_branch2ax_rgb'}, {'res5b_branch2axxx_rgb'}, {}) ;
net.addLayer('res5b_branch2a_relu_d', reluBlock_d, {'res5b_branch2ax_d'}, {'res5b_branch2axxx_d'}, {}) ;

% res5b_branch2b Layer
idx = net1.getLayerIndex('res5b_branch2b');
net.addLayer('res5b_branch2b_rgb',net1.layers(idx).block,{'res5b_branch2axxx_rgb'},{'res5b_branch2b_rgb'},{'res5b_branch2b_filter_rgb'});
net.addLayer('res5b_branch2b_d',net2.layers(idx).block,{'res5b_branch2axxx_d'},{'res5b_branch2b_d'},{'res5b_branch2b_filter_d'});

f1 = net.getParamIndex('res5b_branch2b_filter_rgb');
f2 = net.getParamIndex('res5b_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5b_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5b_branch2b_filter_d';

% bn5b_branch2b layer
idx = net1.getLayerIndex('bn5b_branch2b');
net.addLayer('bn5b_branch2b_rgb',net1.layers(idx).block,{'res5b_branch2b_rgb'},{'res5b_branch2bx_rgb'},{'bn5b_branch2b_mult_rgb','bn5b_branch2b_bias_rgb','bn5b_branch2b_moments_rgb'});
net.addLayer('bn5b_branch2b_d',net2.layers(idx).block,{'res5b_branch2b_d'},{'res5b_branch2bx_d'},{'bn5b_branch2b_mult_d','bn5b_branch2b_bias_d','bn5b_branch2b_moments_d'});


f1 = net.getParamIndex('bn5b_branch2b_mult_rgb');
f2 = net.getParamIndex('bn5b_branch2b_bias_rgb');
f3 = net.getParamIndex('bn5b_branch2b_moments_rgb');
f4 = net.getParamIndex('bn5b_branch2b_mult_d');
f5 = net.getParamIndex('bn5b_branch2b_bias_d');
f6 = net.getParamIndex('bn5b_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5b_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5b_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5b_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5b_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5b_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5b_branch2b_moments_d';

% res5b_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5b_branch2b_relu_rgb', reluBlock_rgb, {'res5b_branch2bx_rgb'}, {'res5b_branch2bxxx_rgb'}, {}) ;
net.addLayer('res5b_branch2b_relu_d', reluBlock_d, {'res5b_branch2bx_d'}, {'res5b_branch2bxxx_d'}, {}) ;

% res5b_branch2c Layer
idx = net1.getLayerIndex('res5b_branch2c');
net.addLayer('res5b_branch2c_rgb',net1.layers(idx).block,{'res5b_branch2bxxx_rgb'},{'res5b_branch2c_rgb'},{'res5b_branch2c_filter_rgb'});
net.addLayer('res5b_branch2c_d',net2.layers(idx).block,{'res5b_branch2bxxx_d'},{'res5b_branch2c_d'},{'res5b_branch2c_filter_d'});

f1 = net.getParamIndex('res5b_branch2c_filter_rgb');
f2 = net.getParamIndex('res5b_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5b_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5b_branch2c_filter_d';

% bn5b_branch2c layer
idx = net1.getLayerIndex('bn5b_branch2c');
net.addLayer('bn5b_branch2c_rgb',net1.layers(idx).block,{'res5b_branch2c_rgb'},{'res5b_branch2cx_rgb'},{'bn5b_branch2c_mult_rgb','bn5b_branch2c_bias_rgb','bn5b_branch2c_moments_rgb'});
net.addLayer('bn5b_branch2c_d',net2.layers(idx).block,{'res5b_branch2c_d'},{'res5b_branch2cx_d'},{'bn5b_branch2c_mult_d','bn5b_branch2c_bias_d','bn5b_branch2c_moments_d'});


f1 = net.getParamIndex('bn5b_branch2c_mult_rgb');
f2 = net.getParamIndex('bn5b_branch2c_bias_rgb');
f3 = net.getParamIndex('bn5b_branch2c_moments_rgb');
f4 = net.getParamIndex('bn5b_branch2c_mult_d');
f5 = net.getParamIndex('bn5b_branch2c_bias_d');
f6 = net.getParamIndex('bn5b_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5b_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5b_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5b_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5b_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5b_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5b_branch2c_moments_d';

% res5b layer(sum)
idx = net1.getLayerIndex('res5b');
net.addLayer('res5b_rgb',net1.layers(idx).block,{'res5ax_rgb','res5b_branch2cx_rgb'},{'res5b_rgb'},{});
net.addLayer('res5b_d',net2.layers(idx).block,{'res5ax_d','res5b_branch2cx_d'},{'res5b_d'},{});

% res5b_relu layer
idx = net1.getLayerIndex('res5b_relu');
net.addLayer('res5b_relu_rgb',net1.layers(idx).block,{'res5b_rgb'},{'res5bx_rgb'},{});
net.addLayer('res5b_relu_d',net2.layers(idx).block,{'res5b_d'},{'res5bx_d'},{});




% res5c_branch2a layer
idx = net1.getLayerIndex('res5c_branch2a');
net.addLayer('res5c_branch2a_rgb',net1.layers(idx).block,{'res5bx_rgb'},{'res5c_branch2a_rgb'},{'res5c_branch2a_filter_rgb'});
net.addLayer('res5c_branch2a_d',net2.layers(idx).block,{'res5bx_d'},{'res5c_branch2a_d'},{'res5c_branch2a_filter_d'});

f1 = net.getParamIndex('res5c_branch2a_filter_rgb');
f2 = net.getParamIndex('res5c_branch2a_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5c_branch2a_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5c_branch2a_filter_d';

% bn5c_branch2a layer
idx = net1.getLayerIndex('bn5c_branch2a');
net.addLayer('bn5c_branch2a_rgb',net1.layers(idx).block,{'res5c_branch2a_rgb'},{'res5c_branch2ax_rgb'},{'bn5c_branch2a_mult_rgb','bn5c_branch2a_bias_rgb','bn5c_branch2a_moments_rgb'});
net.addLayer('bn5c_branch2a_d',net2.layers(idx).block,{'res5c_branch2a_d'},{'res5c_branch2ax_d'},{'bn5c_branch2a_mult_d','bn5c_branch2a_bias_d','bn5c_branch2a_moments_d'});


f1 = net.getParamIndex('bn5c_branch2a_mult_rgb');
f2 = net.getParamIndex('bn5c_branch2a_bias_rgb');
f3 = net.getParamIndex('bn5c_branch2a_moments_rgb');
f4 = net.getParamIndex('bn5c_branch2a_mult_d');
f5 = net.getParamIndex('bn5c_branch2a_bias_d');
f6 = net.getParamIndex('bn5c_branch2a_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5c_branch2a_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5c_branch2a_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5c_branch2a_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5c_branch2a_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5c_branch2a_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5c_branch2a_moments_d';

% res5c_branch2a_relu Layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5c_branch2a_relu_rgb', reluBlock_rgb, {'res5c_branch2ax_rgb'}, {'res5c_branch2axxx_rgb'}, {}) ;
net.addLayer('res5c_branch2a_relu_d', reluBlock_d, {'res5c_branch2ax_d'}, {'res5c_branch2axxx_d'}, {}) ;

% res5c_branch2b Layer
idx = net1.getLayerIndex('res5c_branch2b');
net.addLayer('res5c_branch2b_rgb',net1.layers(idx).block,{'res5c_branch2axxx_rgb'},{'res5c_branch2b_rgb'},{'res5c_branch2b_filter_rgb'});
net.addLayer('res5c_branch2b_d',net2.layers(idx).block,{'res5c_branch2axxx_d'},{'res5c_branch2b_d'},{'res5c_branch2b_filter_d'});

f1 = net.getParamIndex('res5c_branch2b_filter_rgb');
f2 = net.getParamIndex('res5c_branch2b_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5c_branch2b_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5c_branch2b_filter_d';

% bn5c_branch2b layer
idx = net1.getLayerIndex('bn5c_branch2b');
net.addLayer('bn5c_branch2b_rgb',net1.layers(idx).block,{'res5c_branch2b_rgb'},{'res5c_branch2bx_rgb'},{'bn5c_branch2b_mult_rgb','bn5c_branch2b_bias_rgb','bn5c_branch2b_moments_rgb'});
net.addLayer('bn5c_branch2b_d',net2.layers(idx).block,{'res5c_branch2b_d'},{'res5c_branch2bx_d'},{'bn5c_branch2b_mult_d','bn5c_branch2b_bias_d','bn5c_branch2b_moments_d'});


f1 = net.getParamIndex('bn5c_branch2b_mult_rgb');
f2 = net.getParamIndex('bn5c_branch2b_bias_rgb');
f3 = net.getParamIndex('bn5c_branch2b_moments_rgb');
f4 = net.getParamIndex('bn5c_branch2b_mult_d');
f5 = net.getParamIndex('bn5c_branch2b_bias_d');
f6 = net.getParamIndex('bn5c_branch2b_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5c_branch2b_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5c_branch2b_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5c_branch2b_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5c_branch2b_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5c_branch2b_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5c_branch2b_moments_d';

% res5c_branch2b_relu layer
reluBlock_rgb = dagnn.ReLU() ;
reluBlock_d = dagnn.ReLU() ;
net.addLayer('res5c_branch2b_relu_rgb', reluBlock_rgb, {'res5c_branch2bx_rgb'}, {'res5c_branch2bxxx_rgb'}, {}) ;
net.addLayer('res5c_branch2b_relu_d', reluBlock_d, {'res5c_branch2bx_d'}, {'res5c_branch2bxxx_d'}, {}) ;

% res5c_branch2c Layer
idx = net1.getLayerIndex('res5c_branch2c');
net.addLayer('res5c_branch2c_rgb',net1.layers(idx).block,{'res5c_branch2bxxx_rgb'},{'res5c_branch2c_rgb'},{'res5c_branch2c_filter_rgb'});
net.addLayer('res5c_branch2c_d',net2.layers(idx).block,{'res5c_branch2bxxx_d'},{'res5c_branch2c_d'},{'res5c_branch2c_filter_d'});

f1 = net.getParamIndex('res5c_branch2c_filter_rgb');
f2 = net.getParamIndex('res5c_branch2c_filter_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'res5c_branch2c_filter_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'res5c_branch2c_filter_d';

% bn5c_branch2c layer
idx = net1.getLayerIndex('bn5c_branch2c');
net.addLayer('bn5c_branch2c_rgb',net1.layers(idx).block,{'res5c_branch2c_rgb'},{'res5c_branch2cx_rgb'},{'bn5c_branch2c_mult_rgb','bn5c_branch2c_bias_rgb','bn5c_branch2c_moments_rgb'});
net.addLayer('bn5c_branch2c_d',net2.layers(idx).block,{'res5c_branch2c_d'},{'res5c_branch2cx_d'},{'bn5c_branch2c_mult_d','bn5c_branch2c_bias_d','bn5c_branch2c_moments_d'});


f1 = net.getParamIndex('bn5c_branch2c_mult_rgb');
f2 = net.getParamIndex('bn5c_branch2c_bias_rgb');
f3 = net.getParamIndex('bn5c_branch2c_moments_rgb');
f4 = net.getParamIndex('bn5c_branch2c_mult_d');
f5 = net.getParamIndex('bn5c_branch2c_bias_d');
f6 = net.getParamIndex('bn5c_branch2c_moments_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'bn5c_branch2c_mult_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'bn5c_branch2c_bias_rgb';
net.params(f3) = f_rgb(3);
net.params(f3).name = 'bn5c_branch2c_moments_rgb';

net.params(f4) = f_d(1);
net.params(f4).name = 'bn5c_branch2c_mult_d';
net.params(f5) = f_d(2);
net.params(f5).name = 'bn5c_branch2c_bias_d';
net.params(f6) = f_d(3);
net.params(f6).name = 'bn5c_branch2c_moments_d';

% res5c layer(sum)
idx = net1.getLayerIndex('res5c');
net.addLayer('res5c_rgb',net1.layers(idx).block,{'res5bx_rgb','res5c_branch2cx_rgb'},{'res5c_rgb'},{});
net.addLayer('res5c_d',net2.layers(idx).block,{'res5bx_d','res5c_branch2cx_d'},{'res5c_d'},{});

% res5c_relu layer
idx = net1.getLayerIndex('res5c_relu');
net.addLayer('res5c_relu_rgb',net1.layers(idx).block,{'res5c_rgb'},{'res5cx_rgb'},{});
net.addLayer('res5c_relu_d',net2.layers(idx).block,{'res5c_d'},{'res5cx_d'},{});


% fc1 layer
idx = net1.getLayerIndex('fc1');
net.addLayer('fc1_rgb',net1.layers(idx).block,{'res5cx_rgb'},{'fc1_o_rgb'},{'fc1f_rgb','fc1b_rgb'});
net.addLayer('fc1_d',net2.layers(idx).block,{'res5cx_d'},{'fc1_o_d'},{'fc1f_d','fc1b_d'});

f1 = net.getParamIndex('fc1f_rgb');
f2 = net.getParamIndex('fc1b_rgb');
f3 = net.getParamIndex('fc1f_d');
f4 = net.getParamIndex('fc1b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'fc1f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'fc1b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'fc1f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'fc1b_d';

% deconv1 layer
idx = net1.getLayerIndex('deconv_1');
net.addLayer('deconv1_rgb',net1.layers(idx).block,{'fc1_o_rgb'},{'deconv1_o_rgb'},{'deconv1_rgb'});
net.addLayer('deconv1_d',net2.layers(idx).block,{'fc1_o_d'},{'deconv1_o_d'},{'deconv1_d'});

f1 = net.getParamIndex('deconv1_rgb');
f2 = net.getParamIndex('deconv1_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv1_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv1_d';

% skip layer
idx = net1.getLayerIndex('skip1');
net.addLayer('skip1_rgb',net1.layers(idx).block,{'res4fx_rgb'},{'skip1_o_rgb'},{'skip1f_rgb','skip1b_rgb'});
net.addLayer('skip1_d',net2.layers(idx).block,{'res4fx_d'},{'skip1_o_d'},{'skip1f_d','skip1b_d'});

f1 = net.getParamIndex('skip1f_rgb');
f2 = net.getParamIndex('skip1b_rgb');
f3 = net.getParamIndex('skip1f_d');
f4 = net.getParamIndex('skip1b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip1f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip1b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip1f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip1b_d';

% sum1 layer
idx = net1.getLayerIndex('sum1_rgb');
net.addLayer('sum1_rgb',net1.layers(idx).block,{'skip1_o_rgb','deconv1_o_rgb'},{'sum1_o_rgb'},{});
net.addLayer('sum1_d',net2.layers(idx).block,{'skip1_o_d','deconv1_o_d'},{'sum1_o_d'},{});

% deconv2 layer
idx = net1.getLayerIndex('deconv2bis_rgb');
net.addLayer('deconv2_rgb',net1.layers(idx).block,{'sum1_o_rgb'},{'deconv2_o_rgb'},{'deconv2_rgb'});
net.addLayer('deconv2_d',net2.layers(idx).block,{'sum1_o_d'},{'deconv2_o_d'},{'deconv2_d'});

f1 = net.getParamIndex('deconv2_rgb');
f2 = net.getParamIndex('deconv2_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv2_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv2_d';

% skip2 layer
idx = net1.getLayerIndex('skip2');
net.addLayer('skip2_rgb',net1.layers(idx).block,{'res3dx_rgb'},{'skip2_o_rgb'},{'skip2f_rgb','skip2b_rgb'});
net.addLayer('skip2_d',net2.layers(idx).block,{'res3dx_d'},{'skip2_o_d'},{'skip2f_d','skip2b_d'});

f1 = net.getParamIndex('skip2f_rgb');
f2 = net.getParamIndex('skip2b_rgb');
f3 = net.getParamIndex('skip2f_d');
f4 = net.getParamIndex('skip2b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip2f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip2b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip2f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip2b_d';

% sum2 layer
idx = net1.getLayerIndex('sum2_rgb');
net.addLayer('sum2_rgb',net1.layers(idx).block,{'skip2_o_rgb','deconv2_o_rgb'},{'sum2_o_rgb'},{});
net.addLayer('sum2_d',net2.layers(idx).block,{'skip2_o_d','deconv2_o_d'},{'sum2_o_d'},{});

% deconv3 layer 
idx = net1.getLayerIndex('deconv3');
net.addLayer('deconv3_rgb',net1.layers(idx).block,{'sum2_o_rgb'},{'deconv3_o_rgb'},{'deconv3_rgb'});
net.addLayer('deconv3_d',net2.layers(idx).block,{'sum2_o_d'},{'deconv3_o_d'},{'deconv3_d'});

f1 = net.getParamIndex('deconv3_rgb');
f2 = net.getParamIndex('deconv3_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv3_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv3_d';

% add skip3 layer
idx = net1.getLayerIndex('skip3');
net.addLayer('skip3_rgb',net1.layers(idx).block,{'res2cx_rgb'},{'skip3_o_rgb'},{'skip3f_rgb','skip3b_rgb'});
net.addLayer('skip3_d',net2.layers(idx).block,{'res2cx_d'},{'skip3_o_d'},{'skip3f_d','skip3b_d'});

f1 = net.getParamIndex('skip3f_rgb');
f2 = net.getParamIndex('skip3b_rgb');
f3 = net.getParamIndex('skip3f_d');
f4 = net.getParamIndex('skip3b_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip3f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip3b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip3f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip3b_d';

% add sum layer(112*112*40*2)
idx = net1.getLayerIndex('sum3_rgb');
net.addLayer('sum3_rgb',net1.layers(idx).block,{'skip3_o_rgb','deconv3_o_rgb'},{'sum3_o_rgb'},{});
net.addLayer('sum3_d',net2.layers(idx).block,{'skip3_o_d','deconv3_o_d'},{'sum3_o_d'},{});

% add deconv layer
idx = net1.getLayerIndex('deconv4');
net.addLayer('deconv4_rgb',net1.layers(idx).block,{'sum3_o_rgb'},{'deconv4_o_rgb'},{'deconv4_rgb'});
net.addLayer('deconv4_d',net2.layers(idx).block,{'sum3_o_d'},{'deconv4_o_d'},{'deconv4_d'});

f1 = net.getParamIndex('deconv4_rgb');
f2 = net.getParamIndex('deconv4_d');

f_rgb = net1.getParam(net1.layers(idx).params);
f_d = net2.getParam(net2.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv4_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv4_d';

% fusion layer
fuseblock = dagnn.Concat('dim', 3);
net.addLayer('fuse_decision', fuseblock, {'deconv4_o_rgb','deconv4_o_d'}, {'fusion'}, {});

% fc layer after fusion
fuse_fc = dagnn.Conv('size', [1 1 80 40], 'hasBias', true);
net.addLayer('fuse_fc', fuse_fc, {'fusion'}, {'prediction'}, {'filters_f1', 'biases_f1'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 80 40];
  else
    sz = [40,1];
  end
  net.params(p).value = 0.001*randn(sz, 'single');
end


% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

for i = 1:444
    net.params(i).learningRate = 0;
end

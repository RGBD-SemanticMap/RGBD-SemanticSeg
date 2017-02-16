function net = multiModel_fus2(varargin)
dbstop if error;
% run matconvnet/matlab/vl_setupnn ;
opts.sourceModelPath_image = 'data/models/net-448-epoch-9.mat' ;
opts.sourceModelPath_depth = 'data/models/net-448-epoch-10.mat' ;
opts = vl_argparse(opts, varargin) ;

netStruct = load(opts.sourceModelPath_image) ;
net_rgb = dagnn.DagNN.loadobj(netStruct.net) ;
clear netStruct ;

netStruct = load(opts.sourceModelPath_depth) ;
net_d = dagnn.DagNN.loadobj(netStruct.net) ;
clear netStruct ;

removeNames = {net_rgb.layers(184:185).name};
for i = 1 : numel(removeNames)
    net_rgb.removeLayer(removeNames{i});
end
removeNames = {net_d.layers(192:193).name};
for i = 1 : numel(removeNames)
    net_d.removeLayer(removeNames{i});
end

net_rgb.layers(end).outputs = {'fuse_rgb'};
net_d.layers(end).outputs = {'fuse_d'};

net = dagnn.DagNN();
sliceBlock1 = dagnn.Slice('sta',1,'terminus',3);
sliceBlock2 = dagnn.Slice('sta',4,'terminus',6);
net.addLayer('slice1', sliceBlock1, {'input'}, {'rgb_input'}, {});
net.addLayer('slice2', sliceBlock2, {'input'}, {'d_input'}, {});

% add bottleneck in residual network

for i = 1 : 172

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


%-----------------------------------------------------------
%
% add trained parts of rgb and depth
%
%-----------------------------------------------------------

% fc1 layer
idx = net_rgb.getLayerIndex('fc1');
net.addLayer('fc1_rgb',net_rgb.layers(idx).block,{'rgb_res5cx'},{'fc1_o_rgb'},{'fc1f_rgb','fc1b_rgb'});
net.addLayer('fc1_d',net_d.layers(idx).block,{'d_res5cx'},{'fc1_o_d'},{'fc1f_d','fc1b_d'});
  
f1 = net.getParamIndex('fc1f_rgb');
f2 = net.getParamIndex('fc1b_rgb');
f3 = net.getParamIndex('fc1f_d');
f4 = net.getParamIndex('fc1b_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx).params);
f_d = net_d.getParam(net_d.layers(idx).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'fc1f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'fc1b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'fc1f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'fc1b_d';

% depth bn1 layer
idx = net_d.getLayerIndex('bn_fc1');
net.addLayer('bn_fc1_d',net_d.layers(idx).block,{'fc1_o_d'},{'bn_fc1_o_d'},{'bn1_mult','bn1_bias','bn1_moment'});

f1 = net.getParamIndex('bn1_mult');
f2 = net.getParamIndex('bn1_bias');
f3 = net.getParamIndex('bn1_moment');

f = net_d.getParam(net_d.layers(idx).params);

net.params(f1) = f(1);
net.params(f1).name = 'bn1_mult';
net.params(f2) = f(2);
net.params(f2).name = 'bn1_bias';
net.params(f3) = f(3);
net.params(f3).name = 'bn1_moment';

% depth relu1 layer
idx = net_d.getLayerIndex('relu_fc1');
net.addLayer('relu_fc1',net_d.layers(idx).block,{'bn_fc1_o_d'},{'relu1_o_d'},{});


% deconv1 layer
idx1 = net_rgb.getLayerIndex('deconv_1');
idx2 = net_d.getLayerIndex('deconv_1');
net.addLayer('deconv1_rgb',net_rgb.layers(idx1).block,{'fc1_o_rgb'},{'deconv1_o_rgb'},{'deconv1_rgb'});
net.addLayer('deconv1_d',net_d.layers(idx2).block,{'relu1_o_d'},{'deconv1_o_d'},{'deconv1_d'});

f1 = net.getParamIndex('deconv1_rgb');
f2 = net.getParamIndex('deconv1_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv1_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv1_d';

% skip layer
idx1 = net_rgb.getLayerIndex('skip1');
idx2 = net_d.getLayerIndex('skip1');

net.addLayer('skip1_rgb',net_rgb.layers(idx1).block,{'rgb_res4fx'},{'skip1_o_rgb'},{'skip1f_rgb','skip1b_rgb'});
net.addLayer('skip1_d',net_d.layers(idx2).block,{'d_res4fx'},{'skip1_o_d'},{'skip1f_d','skip1b_d'});

f1 = net.getParamIndex('skip1f_rgb');
f2 = net.getParamIndex('skip1b_rgb');
f3 = net.getParamIndex('skip1f_d');
f4 = net.getParamIndex('skip1b_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip1f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip1b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip1f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip1b_d';

% depth bn_skip1 layer
idx = net_d.getLayerIndex('bn_skip1');
net.addLayer('bn_skip1',net_d.layers(idx).block,{'skip1_o_d'},{'bn_skip1_o'},{'bns1_mult','bns1_bias','bns1_moment'});

f1 = net.getParamIndex('bns1_mult');
f2 = net.getParamIndex('bns1_bias');
f3 = net.getParamIndex('bns1_moment');

f = net_d.getParam(net_d.layers(idx).params);

net.params(f1) = f(1);
net.params(f1).name = 'bns1_mult';
net.params(f2) = f(2);
net.params(f2).name = 'bns1_bias';
net.params(f3) = f(3);
net.params(f3).name = 'bns1_moment';

% depth relu_skip1 layer
idx = net_d.getLayerIndex('relu_skip1');
net.addLayer('relu_skip1',net_d.layers(idx).block,{'bn_skip1_o'},{'relu1s_o'},{});


% sum1 layer
idx1 = net_rgb.getLayerIndex('sum1_rgb');
idx2 = net_d.getLayerIndex('sum1_rgb');

net.addLayer('sum1_rgb',net_rgb.layers(idx1).block,{'skip1_o_rgb','deconv1_o_rgb'},{'sum1_o_rgb'},{});
net.addLayer('sum1_d',net_d.layers(idx2).block,{'relu1s_o','deconv1_o_d'},{'sum1_o_d'},{});

% deconv2 layer
idx1 = net_rgb.getLayerIndex('deconv2bis_rgb');
idx2 = net_d.getLayerIndex('deconv2bis_rgb');

net.addLayer('deconv2_rgb',net_rgb.layers(idx1).block,{'sum1_o_rgb'},{'deconv2_o_rgb'},{'deconv2_rgb'});
net.addLayer('deconv2_d',net_d.layers(idx2).block,{'sum1_o_d'},{'deconv2_o_d'},{'deconv2_d'});

f1 = net.getParamIndex('deconv2_rgb');
f2 = net.getParamIndex('deconv2_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv2_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv2_d';


% skip2 layer
idx1 = net_rgb.getLayerIndex('skip2');
idx2 = net_d.getLayerIndex('skip2');

net.addLayer('skip2_rgb',net_rgb.layers(idx1).block,{'rgb_res3dx'},{'skip2_o_rgb'},{'skip2f_rgb','skip2b_rgb'});
net.addLayer('skip2_d',net_d.layers(idx2).block,{'d_res3dx'},{'skip2_o_d'},{'skip2f_d','skip2b_d'});

f1 = net.getParamIndex('skip2f_rgb');
f2 = net.getParamIndex('skip2b_rgb');
f3 = net.getParamIndex('skip2f_d');
f4 = net.getParamIndex('skip2b_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip2f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip2b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip2f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip2b_d';

% depth bn_skip2 layer
idx = net_d.getLayerIndex('bn_skip2');
net.addLayer('bn_skip2',net_d.layers(idx).block,{'skip2_o_d'},{'bn_skip2_o'},{'bns2_mult','bns2_bias','bns2_moment'});

f1 = net.getParamIndex('bns2_mult');
f2 = net.getParamIndex('bns2_bias');
f3 = net.getParamIndex('bns2_moment');

f = net_d.getParam(net_d.layers(idx).params);

net.params(f1) = f(1);
net.params(f1).name = 'bns2_mult';
net.params(f2) = f(2);
net.params(f2).name = 'bns2_bias';
net.params(f3) = f(3);
net.params(f3).name = 'bns2_moment';

% depth relu_skip2 layer
idx = net_d.getLayerIndex('relu_skip2');
net.addLayer('relu_skip2',net_d.layers(idx).block,{'bn_skip2_o'},{'relu2s_o'},{});

% sum2 layer
idx1= net_rgb.getLayerIndex('sum2_rgb');
idx2= net_d.getLayerIndex('sum2_rgb');

net.addLayer('sum2_rgb',net_rgb.layers(idx1).block,{'skip2_o_rgb','deconv2_o_rgb'},{'sum2_o_rgb'},{});
net.addLayer('sum2_d',net_d.layers(idx2).block,{'relu2s_o','deconv2_o_d'},{'sum2_o_d'},{});

% deconv3 layer 
idx1 = net_rgb.getLayerIndex('deconv3');
idx2 = net_d.getLayerIndex('deconv3');

net.addLayer('deconv3_rgb',net_rgb.layers(idx1).block,{'sum2_o_rgb'},{'deconv3_o_rgb'},{'deconv3_rgb'});
net.addLayer('deconv3_d',net_d.layers(idx2).block,{'sum2_o_d'},{'deconv3_o_d'},{'deconv3_d'});

f1 = net.getParamIndex('deconv3_rgb');
f2 = net.getParamIndex('deconv3_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv3_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv3_d';

% add skip3 layer
idx1 = net_rgb.getLayerIndex('skip3');
idx2 = net_d.getLayerIndex('skip3');

net.addLayer('skip3_rgb',net_rgb.layers(idx1).block,{'rgb_res2cx'},{'skip3_o_rgb'},{'skip3f_rgb','skip3b_rgb'});
net.addLayer('skip3_d',net_d.layers(idx2).block,{'d_res2cx'},{'skip3_o_d'},{'skip3f_d','skip3b_d'});

f1 = net.getParamIndex('skip3f_rgb');
f2 = net.getParamIndex('skip3b_rgb');
f3 = net.getParamIndex('skip3f_d');
f4 = net.getParamIndex('skip3b_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'skip3f_rgb';
net.params(f2) = f_rgb(2);
net.params(f2).name = 'skip3b_rgb';
net.params(f3) = f_d(1);
net.params(f3).name = 'skip3f_d';
net.params(f4) = f_d(2);
net.params(f4).name = 'skip3b_d';

% depth bn_skip3 layer
idx = net_d.getLayerIndex('bn_skip3');
net.addLayer('bn_skip3',net_d.layers(idx).block,{'skip3_o_d'},{'bn_skip3_o'},{'bns3_mult','bns3_bias','bns3_moment'});

f1 = net.getParamIndex('bns3_mult');
f2 = net.getParamIndex('bns3_bias');
f3 = net.getParamIndex('bns3_moment');

f = net_d.getParam(net_d.layers(idx).params);

net.params(f1) = f(1);
net.params(f1).name = 'bns3_mult';
net.params(f2) = f(2);
net.params(f2).name = 'bns3_bias';
net.params(f3) = f(3);
net.params(f3).name = 'bns3_moment';

% depth relu_skip3 layer
idx = net_d.getLayerIndex('relu_skip3');
net.addLayer('relu_skip3',net_d.layers(idx).block,{'bn_skip3_o'},{'relu3s_o'},{});


% add sum layer(112*112*40*2)
idx1 = net_rgb.getLayerIndex('sum3_rgb');
idx2 = net_d.getLayerIndex('sum3_rgb');

net.addLayer('sum3_rgb',net_rgb.layers(idx1).block,{'skip3_o_rgb','deconv3_o_rgb'},{'sum3_o_rgb'},{});
net.addLayer('sum3_d',net_d.layers(idx2).block,{'relu3s_o','deconv3_o_d'},{'sum3_o_d'},{});

% add deconv layer
idx1 = net_rgb.getLayerIndex('deconv4');
idx2 = net_d.getLayerIndex('deconv4');

net.addLayer('deconv4_rgb',net_rgb.layers(idx1).block,{'sum3_o_rgb'},{'rgb_fuse_rgb'},{'deconv4_rgb'});
net.addLayer('deconv4_d',net_d.layers(idx2).block,{'sum3_o_d'},{'d_fuse_d'},{'deconv4_d'});

f1 = net.getParamIndex('deconv4_rgb');
f2 = net.getParamIndex('deconv4_d');

f_rgb = net_rgb.getParam(net_rgb.layers(idx1).params);
f_d = net_d.getParam(net_d.layers(idx2).params);

net.params(f1) = f_rgb(1);
net.params(f1).name = 'deconv4_rgb';
net.params(f2) = f_d(1);
net.params(f2).name = 'deconv4_d';





for i = 1 : numel(net.params)
    net.params(i).learningRate = 0;
end


%------------------------------------------------------------------
% 
% add denoising bottleneck layers
%
%------------------------------------------------------------------

% add softmax layer
rgb_softmax = dagnn.SoftMax();
net.addLayer('rgb_softmax',rgb_softmax,{'rgb_fuse_rgb'},{'rgb_fuse_softmax'},{});

d_softmax = dagnn.SoftMax();
net.addLayer('d_softmax',d_softmax,{'d_fuse_d'},{'d_fuse_softmax'},{});

% add entropy layer
entropy_rgb_block = dagnn.Entropy();
net.addLayer('rgb_entro', entropy_rgb_block, {'rgb_fuse_softmax'}, {'rgb_heatmap'}, {});
entropy_d_block = dagnn.Entropy();
net.addLayer('d_entro', entropy_d_block, {'d_fuse_softmax'}, {'d_heatmap'}, {});


rgb_conv = dagnn.Conv('size',[1,1,1,64],'hasBias',true);
rgb_conv_name = horzcat('rgb_conv',num2str(m));
rgb_input_name = 'rgb_heatmap';
rgb_output_name = horzcat(rgb_conv_name,'_out');
filter_name = horzcat(rgb_conv_name,'_f');
bias_name = horzcat(rgb_conv_name,'_b');
net.addLayer(rgb_conv_name,rgb_conv,{rgb_input_name},{rgb_output_name},{filter_name,bias_name});

p = net.getParamIndex(net.layers(end).params{1});
net.params(p).value = 0.3*sqrt(2/(1*1*1))*randn([1,1,1,64],'single');
p = net.getParamIndex(net.layers(end).params{2});
net.params(p).value = zeros([1,64],'single');


d_conv = dagnn.Conv('size',[1,1,1,64],'hasBias',true);

d_conv_name = horzcat('d_conv',num2str(m));
d_input_name = 'd_heatmap';
d_output_name = horzcat(d_conv_name,'_out');
filter_name = horzcat(d_conv_name,'_f');
bias_name = horzcat(d_conv_name,'_b');
net.addLayer(d_conv_name,d_conv,{d_input_name},{d_output_name},{filter_name,bias_name});

p = net.getParamIndex(net.layers(end).params{1});
net.params(p).value = 0.3*sqrt(2/(1*1*1))*randn([1,1,1,64],'single');
p = net.getParamIndex(net.layers(end).params{2});
net.params(p).value = zeros([1,64],'single');

% bottleneck1:batch normalization
rgb_bn = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
rgb_bn_name = horzcat('rgb_bn',num2str(m));
rgb_input_name = rgb_output_name;
rgb_output_name = horzcat(rgb_bn_name,'_out');
mult_name = horzcat(rgb_bn_name,'_mult');
bias_name = horzcat(rgb_bn_name,'_bias');
moment_name = horzcat(rgb_bn_name,'_moment');
net.addLayer(rgb_bn_name,rgb_bn,{rgb_input_name},{rgb_output_name},{mult_name,bias_name,moment_name});

for i = [1,2,3]
	p = net.getParamIndex(net.layers(end).params{i});
	if i == 1
		net.params(p).value = rand([64,1],'single');
	elseif i == 2
		net.params(p).value = randn([64,1],'single');
	else
		net.params(p).value = [randn([1,64],'single');rand([1,64],'single')]';
	end
end


d_bn = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
d_bn_name = horzcat('d_bn',num2str(m));
d_input_name = d_output_name;
d_output_name = horzcat(d_bn_name,'_out');
mult_name = horzcat(d_bn_name,'_mult');
bias_name = horzcat(d_bn_name,'_bias');
moment_name = horzcat(d_bn_name,'_moment');
net.addLayer(d_bn_name,d_bn,{d_input_name},{d_output_name},{mult_name,bias_name,moment_name});

for i = [1,2,3]
	p = net.getParamIndex(net.layers(end).params{i});
	if i == 1
		net.params(p).value = rand([64,1],'single');
	elseif i == 2
		net.params(p).value = randn([64,1],'single');
	else
		net.params(p).value = [randn([1,64],'single');rand([1,64],'single')]';
	end
end

% bottleneck1:relu
rgb_relu = dagnn.ReLU();
rgb_relu_name = horzcat('rgb_relu',num2str(m));
rgb_input_name = rgb_output_name;
rgb_output_name = horzcat(rgb_relu_name,'_out');
net.addLayer(rgb_relu_name,rgb_relu,{rgb_input_name},{rgb_output_name});

d_relu = dagnn.ReLU();
d_relu_name = horzcat('d_relu',num2str(m));
d_input_name = d_output_name;
d_output_name = horzcat(d_relu_name,'_out');
net.addLayer(d_relu_name,d_relu,{d_input_name},{d_output_name});


bottleneck_num = 2;

for m = 1:bottleneck_num
	% bottleneck1:conv 1*1 pad 0
	rgb_conv1 = dagnn.Conv('size',[1,1,64,32],'hasBias',true);

	rgb_conv_name = horzcat('rgb_conv',num2str(m),'_1');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_conv_name,'_out');
	filter_name = horzcat(rgb_conv_name,'_f');
	bias_name = horzcat(rgb_conv_name,'_b');
	net.addLayer(rgb_conv_name,rgb_conv1,{rgb_input_name},{rgb_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(1*1*64))*randn([1,1,64,32],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,32],'single');


	d_conv1 = dagnn.Conv('size',[1,1,64,32],'hasBias',true);

	d_conv_name = horzcat('d_conv',num2str(m),'_1');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_conv_name,'_out');
	filter_name = horzcat(d_conv_name,'_f');
	bias_name = horzcat(d_conv_name,'_b');
	net.addLayer(d_conv_name,d_conv1,{d_input_name},{d_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(1*1*64))*randn([1,1,64,32],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,32],'single');

	% bottleneck1:batch normalization
	rgb_bn1 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	rgb_bn_name = horzcat('rgb_bn',num2str(m),'_1');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_bn_name,'_out');
	mult_name = horzcat(rgb_bn_name,'_mult');
	bias_name = horzcat(rgb_bn_name,'_bias');
	moment_name = horzcat(rgb_bn_name,'_moment');
	net.addLayer(rgb_bn_name,rgb_bn1,{rgb_input_name},{rgb_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([32,1],'single');
		elseif i == 2
			net.params(p).value = randn([32,1],'single');
		else
			net.params(p).value = [randn([1,32],'single');rand([1,32],'single')]';
		end
	end


	d_bn1 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	d_bn_name = horzcat('d_bn',num2str(m),'_1');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_bn_name,'_out');
	mult_name = horzcat(d_bn_name,'_mult');
	bias_name = horzcat(d_bn_name,'_bias');
	moment_name = horzcat(d_bn_name,'_moment');
	net.addLayer(d_bn_name,d_bn1,{d_input_name},{d_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([32,1],'single');
		elseif i == 2
			net.params(p).value = randn([32,1],'single');
		else
			net.params(p).value = [randn([1,32],'single');rand([1,32],'single')]';
		end
	end

	% bottleneck1:relu
	rgb_relu1 = dagnn.ReLU();
	rgb_relu_name = horzcat('rgb_relu',num2str(m),'_1');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_relu_name,'_out');
	net.addLayer(rgb_relu_name,rgb_relu1,{rgb_input_name},{rgb_output_name});

	d_relu1 = dagnn.ReLU();
	d_relu_name = horzcat('d_relu',num2str(m),'_1');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_relu_name,'_out');
	net.addLayer(d_relu_name,d_relu1,{d_input_name},{d_output_name});

	% bottleneck1:conv 3*3 pad 1
	rgb_conv2 = dagnn.Conv('size',[3,3,32,32],'hasBias',true);

	rgb_conv_name = horzcat('rgb_conv',num2str(m),'_2');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_conv_name,'_out');
	filter_name = horzcat(rgb_conv_name,'_f');
	bias_name = horzcat(rgb_conv_name,'_b');
	net.addLayer(rgb_conv_name,rgb_conv2,{rgb_input_name},{rgb_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(3*3*32))*randn([3,3,32,32],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,32],'single');
	net.layers(end).block.pad = [1,1,1,1];

	d_conv2 = dagnn.Conv('size',[3,3,32,32],'hasBias',true);

	d_conv_name = horzcat('d_conv',num2str(m),'_2');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_conv_name,'_out');
	filter_name = horzcat(d_conv_name,'_f');
	bias_name = horzcat(d_conv_name,'_b');
	net.addLayer(d_conv_name,d_conv2,{d_input_name},{d_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(3*3*32))*randn([3,3,32,32],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,32],'single');
	net.layers(end).block.pad = [1,1,1,1];


	% bottleneck1:batch normalization
	rgb_bn2 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	rgb_bn_name = horzcat('rgb_bn',num2str(m),'_2');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_bn_name,'_out');
	mult_name = horzcat(rgb_bn_name,'_mult');
	bias_name = horzcat(rgb_bn_name,'_bias');
	moment_name = horzcat(rgb_bn_name,'_moment');
	net.addLayer(rgb_bn_name,rgb_bn2,{rgb_input_name},{rgb_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([32,1],'single');
		elseif i == 2
			net.params(p).value = randn([32,1],'single');
		else
			net.params(p).value = [randn([1,32],'single');rand([1,32],'single')]';
		end
	end


	d_bn2 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	d_bn_name = horzcat('d_bn',num2str(m),'_2');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_bn_name,'_out');
	mult_name = horzcat(d_bn_name,'_mult');
	bias_name = horzcat(d_bn_name,'_bias');
	moment_name = horzcat(d_bn_name,'_moment');
	net.addLayer(d_bn_name,d_bn2,{d_input_name},{d_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([32,1],'single');
		elseif i == 2
			net.params(p).value = randn([32,1],'single');
		else
			net.params(p).value = [randn([1,32],'single');rand([1,32],'single')]';
		end
	end


	% bottleneck1:relu
	rgb_relu2 = dagnn.ReLU();
	rgb_relu_name = horzcat('rgb_relu',num2str(m),'_2');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_relu_name,'_out');
	net.addLayer(rgb_relu_name,rgb_relu2,{rgb_input_name},{rgb_output_name});

	d_relu2 = dagnn.ReLU();
	d_relu_name = horzcat('d_relu',num2str(m),'_2');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_relu_name,'_out');
	net.addLayer(d_relu_name,d_relu2,{d_input_name},{d_output_name});



	% bottleneck1:conv 1*1 
	rgb_conv3 = dagnn.Conv('size',[1,1,32,64],'hasBias',true);

	rgb_conv_name = horzcat('rgb_conv',num2str(m),'_3');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_conv_name,'_out');
	filter_name = horzcat(rgb_conv_name,'_f');
	bias_name = horzcat(rgb_conv_name,'_b');
	net.addLayer(rgb_conv_name,rgb_conv3,{rgb_input_name},{rgb_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(1*1*32))*randn([1,1,32,64],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,64],'single');


	d_conv3 = dagnn.Conv('size',[1,1,32,64],'hasBias',true);

	d_conv_name = horzcat('d_conv',num2str(m),'_3');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_conv_name,'_out');
	filter_name = horzcat(d_conv_name,'_f');
	bias_name = horzcat(d_conv_name,'_b');
	net.addLayer(d_conv_name,d_conv3,{d_input_name},{d_output_name},{filter_name,bias_name});

	p = net.getParamIndex(net.layers(end).params{1});
	net.params(p).value = 0.3*sqrt(2/(1*1*32))*randn([1,1,32,64],'single');
	p = net.getParamIndex(net.layers(end).params{2});
	net.params(p).value = zeros([1,64],'single');



	% bottleneck1:batch normalization
	rgb_bn3 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	rgb_bn_name = horzcat('rgb_bn',num2str(m),'_3');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_bn_name,'_out');
	mult_name = horzcat(rgb_bn_name,'_mult');
	bias_name = horzcat(rgb_bn_name,'_bias');
	moment_name = horzcat(rgb_bn_name,'_moment');
	net.addLayer(rgb_bn_name,rgb_bn3,{rgb_input_name},{rgb_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([64,1],'single');
		elseif i == 2
			net.params(p).value = randn([64,1],'single');
		else
			net.params(p).value = [randn([1,64],'single');rand([1,64],'single')]';
		end
	end


	d_bn3 = dagnn.BatchNorm('numChannels',[],'epsilon',1e-5);
	d_bn_name = horzcat('d_bn',num2str(m),'_3');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_bn_name,'_out');
	mult_name = horzcat(d_bn_name,'_mult');
	bias_name = horzcat(d_bn_name,'_bias');
	moment_name = horzcat(d_bn_name,'_moment');
	net.addLayer(d_bn_name,d_bn3,{d_input_name},{d_output_name},{mult_name,bias_name,moment_name});

	for i = [1,2,3]
		p = net.getParamIndex(net.layers(end).params{i});
		if i == 1
			net.params(p).value = rand([64,1],'single');
		elseif i == 2
			net.params(p).value = randn([64,1],'single');
		else
			net.params(p).value = [randn([1,64],'single');rand([1,64],'single')]';
		end
	end


	% bottleneck1:relu
	rgb_relu3 = dagnn.ReLU();
	rgb_relu_name = horzcat('rgb_relu',num2str(m),'_3');
	rgb_input_name = rgb_output_name;
	rgb_output_name = horzcat(rgb_relu_name,'_out');
	net.addLayer(rgb_relu_name,rgb_relu3,{rgb_input_name},{rgb_output_name});

	d_relu3 = dagnn.ReLU();
	d_relu_name = horzcat('d_relu',num2str(m),'_3');
	d_input_name = d_output_name;
	d_output_name = horzcat(d_relu_name,'_out');
	net.addLayer(d_relu_name,d_relu3,{d_input_name},{d_output_name});
end


rgb_conv = dagnn.Conv('size',[1,1,64,1],'hasBias',true);

rgb_conv_name = 'rgb_conv_x';
rgb_input_name = rgb_output_name;
rgb_output_name = horzcat(rgb_conv_name,'_out');
filter_name = horzcat(rgb_conv_name,'_f');
bias_name = horzcat(rgb_conv_name,'_b');
net.addLayer(rgb_conv_name,rgb_conv,{rgb_input_name},{rgb_output_name},{filter_name,bias_name});

p = net.getParamIndex(net.layers(end).params{1});
net.params(p).value = 0.3*sqrt(2/(1*1*64))*randn([1,1,64,1],'single');
p = net.getParamIndex(net.layers(end).params{2});
net.params(p).value = zeros([1,1],'single');


d_conv = dagnn.Conv('size',[1,1,64,1],'hasBias',true);

d_conv_name = 'd_conv_x';
d_input_name = d_output_name;
d_output_name = horzcat(d_conv_name,'_out');
filter_name = horzcat(d_conv_name,'_f');
bias_name = horzcat(d_conv_name,'_b');
net.addLayer(d_conv_name,d_conv,{d_input_name},{d_output_name},{filter_name,bias_name});

p = net.getParamIndex(net.layers(end).params{1});
net.params(p).value = 0.3*sqrt(2/(1*1*64))*randn([1,1,64,1],'single');
p = net.getParamIndex(net.layers(end).params{2});
net.params(p).value = zeros([1,1],'single');


rgb_denoise_sig = dagnn.Sigmoid();
net.addLayer('rgb_denoise_sig', rgb_denoise_sig, {rgb_output_name}, {'rgb_denoise_o5'}, {});
d_denoise_sig = dagnn.Sigmoid();
net.addLayer('d_denoise_sig', d_denoise_sig, {d_output_name}, {'d_denoise_o5'}, {});


% add supervision layers
rgb_filter_block = dagnn.FilterPred();
net.addLayer('rgb_fgt', rgb_filter_block, {'rgb_fuse_rgb', 'label'}, {'rgb_filter_gt'}, {});
d_filter_block = dagnn.FilterPred();
net.addLayer('d_fgt', d_filter_block, {'d_fuse_d', 'label'}, {'d_filter_gt'}, {});

filter_gen_block = dagnn.FilterGen();
net.addLayer('filter_gen', filter_gen_block, {'rgb_filter_gt', 'd_filter_gt'}, {'rgb_f_final', 'd_f_final'});


net.addLayer('rgb_objective', ...
  FilterLoss('loss', 'filterloss'), ...
  {'rgb_denoise_o5', 'rgb_f_final'}, 'rgb_objective') ;


net.addLayer('d_objective', ...
  FilterLoss('loss', 'filterloss'), ...
  {'d_denoise_o5', 'd_f_final'}, 'd_objective') ;



% add filtering layers
rgb_filterProcess_block = dagnn.FilterProcess();
net.addLayer('rgb_fp', rgb_filterProcess_block, {'rgb_fuse_rgb',  'rgb_denoise_o5'}, {'rgb_fuse_in'}, {});

d_filterProcess_block = dagnn.FilterProcess();
net.addLayer('d_fp', d_filterProcess_block, {'d_fuse_d',  'd_denoise_o5'}, {'d_fuse_in'}, {});

% add fusing layers
fuseblock = dagnn.Concat('dim', 3);
net.addLayer('fuse', fuseblock, {'rgb_fuse_in','d_fuse_in'}, {'fusion_1'}, {});

fuse_conv1_block = dagnn.Conv('size', [1 1 80 40], 'hasBias', true);
net.addLayer('fc1_fus', fuse_conv1_block, {'fusion_1'}, {'fusion_2'}, {'filters_f1', 'biases_f1'});

% for i = [1 2]
%   p = net.getParamIndex(net.layers(end).params{i}) ;
%   if i == 1
%     sz = [1 1 80 40];
%   else
%     sz = [1 40];
%   end
%   net.params(p).value = single(normrnd(0, 0.05, sz));
% end
% net.params(p).value = ones(sz, 'single');
p = net.getParamIndex(net.layers(end).params{1});
net.params(p).value = 0.3*sqrt(2/(1*1*80))*randn([1,1,80,40],'single');
p = net.getParamIndex(net.layers(end).params{2});
net.params(p).value = zeros([1,40],'single');


% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'fusion_2', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'fusion_2', 'label'}, 'accuracy') ;
end


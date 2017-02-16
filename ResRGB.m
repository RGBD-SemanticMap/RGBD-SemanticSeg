function net = ResRGB(varargin)
dbstop if error;
res_path = 'data/NYU/imagenet-resnet-50-dag.mat';
image_path = 'data/NYU/image_net.mat';
resnet_mode = '4s';

net = dagnn.DagNN.loadobj(load(res_path));
net1p = load(image_path);
net1 = dagnn.DagNN.loadobj(net1p.net);

% remove softmax fc and pooling layer
net.removeLayer(net.layers(end).name);
net.removeLayer(net.layers(end).name);
net.removeLayer(net.layers(end).name);

% add fc1 layer(original input size:448*448*3*2 fc1 output size:14*14*40*2)
fc1 = dagnn.Conv('size', [1 1 2048 40], 'hasBias', true);
net.addLayer('fc1', fc1, {'res5cx'}, {'fc1_o'}, {'fc1_filters', 'fc1_biases'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 2048 40];
  else
    sz = [40 1];
  end
  net.params(p).value = 0.001*randn(sz, 'single');
end



% add deconv layer(output size 28*28*40*2)
idx = net1.getLayerIndex('deconv2');
net.addLayer('deconv_1',net1.layers(idx).block,{'fc1_o'},{'deconv1_o'},{'deconv1_rgb'});

f1 = net.getParamIndex('deconv1_rgb');

f_rgb = net1.getParam(net1.layers(idx).params);

net.params(f1) = f_rgb;
net.params(f1).name = 'deconv1_rgb';

% add skip layer
skip1 = dagnn.Conv('size',[1,1,1024,40],'hasBias',true);
net.addLayer('skip1',skip1,{'res4fx'},{'skip1_o'},{'skip1_f','skip1_b'});

for i = [1 2]
  p = net.getParamIndex(net.layers(end).params{i}) ;
  if i == 1
    sz = [1 1 1024 40];
  else
    sz = [40 1];
  end
  net.params(p).value = 0.001*randn(sz, 'single');
end


% add sum layer(28*28*40*2)
idx = net1.getLayerIndex('sum1');
net.addLayer('sum1_rgb',net1.layers(idx).block,{'skip1_o','deconv1_o'},{'sum1_o'},{});

if strcmp(resnet_mode,'16s')
	filters = single(bilinear_u(32, 40, 40)) ;
	net.addLayer('deconv16', ...
    dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 16, ...
                      'crop', 8, ...
                      'numGroups', 40, ...
                      'hasBias', false, ...
                      'opts', {}), ...
             'sum1_o', 'prediction', 'deconv16f') ;

	f = net.getParamIndex('deconv16f') ;
	net.params(f).value = filters ;
	net.params(f).learningRate = 0 ;
	net.params(f).weightDecay = 1 ;

	% Make sure that the output of the bilinear interpolator is not discared for
	% visualization purposes
	net.vars(net.getVarIndex('prediction')).precious = 1 ;
else
    % add deconv layer(output 56*56*40*2)
    idx = net1.getLayerIndex('deconv2bis');
    net.addLayer('deconv2bis_rgb',net1.layers(idx).block,{'sum1_o'},{'deconv2_o'},{'deconv2bisf_rgb'});

    f1 = net.getParamIndex('deconv2bisf_rgb');

    f_rgb = net1.getParam(net1.layers(idx).params);

    net.params(f1) = f_rgb;
    net.params(f1).name = 'deconv2bisf_rgb';

    % add skip layer(output size 56*56*40*2)
    skip2 = dagnn.Conv('size',[1,1,512,40],'hasBias','true');
    net.addLayer('skip2',skip2,{'res3dx'},{'skip2_o'},{'skip2_f','skip2_b'});

    for i = [1 2]
      p = net.getParamIndex(net.layers(end).params{i}) ;
      if i == 1
        sz = [1 1 512 40];
      else
        sz = [40 1];
      end
      net.params(p).value = 0.001*randn(sz, 'single');
    end

	
    % add sum layer
    idx = net1.getLayerIndex('sum2');
    net.addLayer('sum2_rgb',net1.layers(idx).block,{'skip2_o','deconv2_o'},{'sum2_o'},{});

    % % add deconv layer
    % idx = net1.getLayerIndex('deconv8');
    % net.addLayer('deconv3_rgb',net1.layers(idx).block,{'sum2_o'},{'prediction'},{'deconv3f_rgb'});


    % f1 = net.getParamIndex('deconv3f_rgb');

    % f_rgb = net1.getParam(net1.layers(idx).params);

    % net.params(f1) = f_rgb;
    % net.params(f1).name = 'deconv3f_rgb';
	
	% add deconv layer(output size 112*112*40*2)
    idx = net1.getLayerIndex('deconv2bis');
    net.addLayer('deconv3',net1.layers(idx).block,{'sum2_o'},{'deconv3_o'},{'deconv3_rgb'});

    f1 = net.getParamIndex('deconv3_rgb');

    f_rgb = net1.getParam(net1.layers(idx).params);

    net.params(f1) = f_rgb;
    net.params(f1).name = 'deconv3_rgb';

	% add skip layer
	skip3 = dagnn.Conv('size',[1,1,256,40],'hasBias','true');
    net.addLayer('skip3',skip3,{'res2cx'},{'skip3_o'},{'skip3_f','skip3_b'});

    for i = [1 2]
      p = net.getParamIndex(net.layers(end).params{i}) ;
      if i == 1
        sz = [1 1 256 40];
      else
        sz = [40 1];
      end
      net.params(p).value = 0.001*randn(sz, 'single');
    end
	
	% add sum layer(112*112*40*2)
	idx = net1.getLayerIndex('sum2');
    net.addLayer('sum3_rgb',net1.layers(idx).block,{'skip3_o','deconv3_o'},{'sum3_o'},{});
	
	% add deconv layer(out 448*448*40*2)
	filters = single(bilinear_u(8, 40, 40)) ;
	net.addLayer('deconv4', ...
    dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 4, ...
                      'crop', 2, ...
                      'numGroups', 40, ...
                      'hasBias', false, ...
                      'opts', {}), ...
             'sum3_o', 'prediction', 'deconv4f') ;

	f = net.getParamIndex('deconv4f') ;
	net.params(f).value = filters ;
	net.params(f).learningRate = 0 ;
	net.params(f).weightDecay = 1 ;

	% Make sure that the output of the bilinear interpolator is not discared for
	% visualization purposes
	net.vars(net.getVarIndex('prediction')).precious = 1 ;
end

	

% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

% modify layer element name
input1 = net.getVarIndex('data');
net.vars(input1).name = 'input';
inp = net.getLayerIndex('conv1');
net.layers(inp).inputs = 'input';


net_ = net;
net = net_.saveobj();
save(['imagenet-resnet-50-',resnet_mode,'-dag.mat'],'-struct', 'net') ;

classdef FilterGen < dagnn.Layer
%% to generate the filter groundtruth
  methods
  	function outputs = forward(obj, inputs, params)
      rgb = inputs{1};
      depth = inputs{2};
  	  outputs{1} = zeros(size(inputs{1}),'single') + 0.5 + 0.5*single(rgb - depth);
  	  outputs{2} = zeros(size(inputs{1}),'single') + 0.5 + 0.5*single(depth - rgb);
  	end

  	function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
  	  derInputs = {};
  	  derParams = {};
  	end

  	function obj = FilterGen(varargin)
  	  obj.load(varargin);
    end
  end
end
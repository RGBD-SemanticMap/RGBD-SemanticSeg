classdef FilterProcess < dagnn.Layer
%% to generate the filter groundtruth
  methods
  	function outputs = forward(obj, inputs, params)
      outputs{1} = bsxfun(@times, inputs{1}, inputs{2});
  	end

  	function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
  	  derInputs{1} = bsxfun(@times, derOutputs{1}, inputs{2});
      derInputs{2} = {};
  	  derParams = {};
  	end

  	function obj = FilterProcess(varargin)
  	  obj.load(varargin);
    end
  end
end
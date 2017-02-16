classdef FilterPred < dagnn.Layer
%% to predict the label using unary block
  methods
  	function outputs = forward(obj, inputs, params)
  	  unary = abs(inputs{1}) ;
  	  [~, pred] = max(unary, [], 3);
  	  lb = inputs{2};
  	  outputs{1} =  single(lb == pred);
  	end

  	function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
  	  derInputs = {};
  	  derParams = {};
  	end

  	function obj = FilterPred(varargin)
  	  obj.load(varargin);
    end
  end
end

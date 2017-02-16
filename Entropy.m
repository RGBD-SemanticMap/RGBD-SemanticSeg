classdef Entropy < dagnn.Layer
%% calculate the entropy of input

  methods
  	function outputs = forward(obj, inputs, params)
      etp = inputs{1};
      etp = abs(etp .* log2(complex(etp)));
      etp = sum(etp, 3);
  	  outputs{1} = etp;   
  	end

  	function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
  	  derInputs{1} = {};
  	  derParams = {};
  	end

  	function obj = Entropy(varargin)
  	  obj.load(varargin);
    end
  end
end
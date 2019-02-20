classdef Gaus < dagnn.ElementWise
    properties
        slope = 0.5
    end
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnGaus(inputs{1},obj.slope) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnGaus(inputs{1}, obj.slope, derOutputs{1}) ;
            derParams = {} ;
        end
        function obj = Gaus(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.slope = obj.slope ;
        end
    end
end

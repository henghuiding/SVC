classdef Gaussian < dagnn.ElementWise
    properties
        method = 'gaussian';
        size = 20;
        sigma = 5;
        Boundaryoptions = 0;
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            gausFilter = fspecial('gaussian', [obj.size,obj.size], obj.sigma);%b=gather(inputs{1}(:,:,:,1));
            outputs{1} = imfilter(inputs{1}, gausFilter, obj.Boundaryoptions);
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            gausFilter = fspecial('gaussian', [obj.size,obj.size], obj.sigma);
            derParams={};
            derInputs{1}= imfilter(derOutputs{1}, gausFilter, obj.Boundaryoptions);
        end
        
        function obj = Gaussian(varargin)
            obj.load(varargin) ;
        end
    end
end

% - Boundary options
%  
%         X            Input array values outside the bounds of the array
%                      are implicitly assumed to have the value X.  When no
%                      boundary option is specified, imfilter uses X = 0.
%  
%         'symmetric'  Input array values outside the bounds of the array
%                      are computed by mirror-reflecting the array across
%                      the array border.
%  
%         'replicate'  Input array values outside the bounds of the array
%                      are assumed to equal the nearest array border
%                      value.
%  
%         'circular'   Input array values outside the bounds of the array
%                      are computed by implicitly assuming the input array
%                      is periodic.
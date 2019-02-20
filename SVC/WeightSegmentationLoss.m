classdef WeightSegmentationLoss < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, params)
            mass = max(sum(sum(inputs{2} > 0,2),1), 1) ;
                  outputs{1} = vl_nnloss_new(inputs{1}, inputs{2}, [], ...
                      'loss', obj.loss, ...
                      'instanceWeights', 1./mass, ...
                      'classWeights', inputs{3}) ;
            
%             outputs{1} = vl_fcnidfsoftmaxloss(inputs{1}, inputs{2}, inputs{3}(2:end));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            mass = max(sum(sum(inputs{2} > 0,2),1) , 1) ;
            derInputs{1} = vl_nnloss_new(inputs{1}, inputs{2}, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', 1./mass, ...
                'classWeights', inputs{3}) ;
            
%             derInputs{1} = vl_fcnidfsoftmaxloss(inputs{1}, inputs{2}, inputs{3}(2:end), derOutputs{1});
            derInputs{2} = [] ;
            derInputs{3} = [];
            derParams = {};
            
        end
        
        function obj = WeightSegmentationLoss(varargin)
            obj.load(varargin) ;
        end
    end
end

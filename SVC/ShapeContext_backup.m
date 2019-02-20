classdef ShapeContext < dagnn.ElementWise
    properties
        method = 'Gate';
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            features = inputs{1};
            gates = inputs{2};
            outputs{1} = 0*features;
            [h, w, ~, ~] = size (features);
            assert(isequal(h*w, size(gates, 3)));
            for ii=1:h
                for jj=1:w
                    outputs{1}(ii,jj,:,:) = sum(sum(bsxfun(@times, features, gates(:,:,(ii-1)*w+jj,:)),1),2);
                end
            end
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            features = inputs{1};
            gates = inputs{2};
            [h, w, ~, ~] = size (features);
            assert(isequal(h*w, size(gates, 3)));
            derInputs{1}=0*derOutputs{1};
            derInputs{2}=0*derOutputs{1}(:,:,1,:);
            for ii=1:h
                for jj=1:w
                    for nn=1:h
                        for mm=1:w
                          derInputs{2}(ii,jj,(nn-1)*w+mm,:) = squeeze(mean(bsxfun(@times, features(ii,jj,:,:), derOutputs{1}(ii,jj,:,:)),3));
                          derInputs{1}(ii,jj,:,:) = derInputs{1}(ii,jj,:,:) + bsxfun(@times, gates(ii,jj,(nn-1)*w+mm,:), derOutputs{1}(nn,mm,:,:));
                        end
                    end
                end
            end
        end
        
        function obj = ShapeContext(varargin)
            obj.load(varargin) ;
        end
    end
end
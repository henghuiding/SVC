classdef ShapeContext < dagnn.ElementWise
    properties
        method = 'Gate';
        padsize = 0;
        kersize = 0;
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            features = gather(inputs{1});
            outputs{1} = 0*features;
            [h, w, c, batch] = size (features);
            features = padarray(features, [obj.padsize, obj.padsize],0);
            gates = gather(inputs{2});
            assert(isequal(floor(obj.kersize/2), obj.padsize));
            for jj = obj.padsize+1:obj.padsize+w
                for ii = obj.padsize+1:obj.padsize+h
                    gates_ = gates(ii-obj.padsize,jj-obj.padsize,:,:);
                    gates_ = reshape(gates_, [obj.kersize, obj.kersize, 1, batch]);
                    l= ii-obj.padsize; r= ii + obj.padsize; u= jj -obj.padsize; d= jj + obj.padsize;
                    features_ = features(l:r,u:d,:,:);
                    outputs{1}(ii-obj.padsize,jj-obj.padsize,:,:) = sum(sum(bsxfun(@times, features_, gates_),1),2);
                end
            end
            outputs{1} = gpuArray(outputs{1});
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            features = gather(inputs{1});
            [h, w, c, batch] = size (features);
            features = padarray(features, [obj.padsize, obj.padsize],0);
            gates = gather(inputs{2});
            gates_ = zeros(obj.kersize, obj.kersize, 1, batch);
            derInputs{2}=0*gates;
            derOutputs1 = gather(derOutputs{1});
            derInputs{1}=0*derOutputs1;
            derOutputs1 = padarray(derOutputs1, [obj.padsize, obj.padsize],0);
            for jj=obj.padsize+1:obj.padsize+w
                for ii=obj.padsize+1:obj.padsize+h
                    l= ii-obj.padsize; r= ii + obj.padsize; u= jj -obj.padsize; d= jj + obj.padsize;
                    features_ = features(l:r,u:d,:,:);
                    derOutputs1_ = derOutputs1(ii,jj,:,:);
                    derOutputs1_1 = derOutputs1(l:r,u:d,:,:);
                    derInputs2_ = sum(bsxfun(@times, features_, derOutputs1_),3);
                    derInputs{2}(ii-obj.padsize,jj-obj.padsize,:,:) = reshape(derInputs2_, [1, 1, obj.kersize*obj.kersize, batch]);
                    index = obj.kersize * obj.kersize;
                    for nn=1:obj.kersize
                        for mm=1:obj.kersize
                            if (ii-2*obj.padsize+mm-1)>0 && (jj-2*obj.padsize+nn-1)>0 && (ii-2*obj.padsize+mm-1)<=h && (jj-2*obj.padsize+nn-1)<=w
                                gates_(mm,nn,:,:) = gates (ii-2*obj.padsize+mm-1,jj-2*obj.padsize+nn-1, index, :);
                            else
                                gates_(mm,nn,:,:) = 0;
                            end
                            index = index -1;
                        end
                    end
                    derInputs{1}(ii-obj.padsize,jj-obj.padsize,:,:) = sum(sum(bsxfun(@times, gates_, derOutputs1_1),1),2);
                    gates_ = gates_ * 0;
                end
            end
            derInputs{1} = gpuArray(derInputs{1});
            derInputs{2} = gpuArray(derInputs{2});
        end
        
        function obj = ShapeContext(varargin)
            obj.load(varargin) ;
        end
    end
end
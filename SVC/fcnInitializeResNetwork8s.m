
function net = fcnInitializeResNetwork8s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;

%% Remove the last layer
net.removeLayer('deconv16') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x6', 'x7', 'deconvf_2') ;

f = net.getParamIndex('deconvf_2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% Add direct output from pool4 

switch opts.resLayer
    case 50
        skip3_input = 'res3dx'
    case 101
        skip3_input = 'res3b3x'
    case 152
        skip3_input = 'res3b7x'
end

%% build skip network
skip_inputs = {};
switch opts.resLayer
    case 50
        % 50 layer
        skip_inputs = {'res3bx', 'res3cx'};
        
    case 101
        % 101 layer
        for ll = 1 : 2
                skip_inputs{end+1} = sprintf('res3b%dx',ll);
        end
    case 152
        % 152 layer
        for ll = 1 : 6
                skip_inputs{end+1} = sprintf('res3b%dx',ll);
        end
end

skip_inputs = ['res3ax', skip_inputs,  skip3_input];

[net, classifier_out] = skipNetwork(net, skip_inputs, 512, 512, ...
    nClass, opts.newLr, 'skip3');

% Add summation layer
net.addLayer('sum3', dagnn.Sum(), ['x7', classifier_out], 'x10') ;

%% Add deconvolution layers
filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x10', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% add conv layers to mimic the crf
% net.addLayer('crf_1', ...
%      dagnn.Conv('size', [3 3 nClass nClass], 'pad', 1), ...
%      'prediction_1', 'prediction', {'crf_f1','crf_b1'});
% 
% f = net.getParamIndex('crf_f1') ;
% net.params(f).value = 1e-2*randn(3, 3, nClass, nClass, 'single');
% net.params(f).learningRate = 1;
% net.params(f).weightDecay = 1 ;
% 
% f = net.getParamIndex('crf_b1') ;
% net.params(f).value = zeros(1, 1, nClass, 'single') ;
% net.params(f).learningRate = 2 ;
% net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

function net = fcnInitializeNetwork16s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 171;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

% nh = 512;
% nClass = 150;

nh = opts.nh;
nClass = opts.nClass;

%% Remove the last layer
net.removeLayer('deconv32') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'sum_1_out', 'x36', 'deconvf_1') ;

f = net.getParamIndex('deconvf_1') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% build skip network
skip_inputs = {'x30', 'x28', 'x26', 'x24'};
% skip_inputs = {'x24'};
[net, skip_classifier_out] = skipNetwork(net, skip_inputs, 512, nh, ...
    nClass, opts.newLr, 'skip4');

% Add summation layer
net.addLayer('sum2', dagnn.Sum(), ['x36', skip_classifier_out], 'x38') ;
% net.addLayer('Weighted_sum_2', ...
%     DagGatedsum('method', 'sum'), ...
%     ['x36', skip_classifier_out], 'x38', 'WeightedSum_param2');
% f = net.getParamIndex('WeightedSum_param2') ;
% net.params(f).value = weightedSum_initialize(nClass,numel(['x36', skip_classifier_out]),1) ;
% net.params(f).learningRate = 100 ;
% net.params(f).weightDecay = 0.01;

%% Add deconvolution layers
filters = single(bilinear_u(16, nClass, nClass)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 8, ...
  'crop', 4, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x38', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;
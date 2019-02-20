function net = fcnInitializeNetwork4s(net, varargin)
opts.rnn = false;
opts.nh = 256;
opts.nClass = 150;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;
opts.newLr = opts.newLr * 0.1;

%% Remove the last layer
net.removeLayer('deconv8') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x42', 'x43', 'deconvf_3') ;

f = net.getParamIndex('deconvf_3') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% build skip network
skip_inputs = {'x12', 'x14', 'x16'};
[net, skip_classifier_out] = skipNetwork(net, skip_inputs, 256, nh, ...
    nClass, opts.newLr, 'skip2');

% Add summation layer
net.addLayer('sum4', dagnn.Sum(), ['x43', skip_classifier_out], 'x46') ;

%% Add deconvolution layers
filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv4', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x46', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;
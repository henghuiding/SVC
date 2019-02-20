function net = fcnInitializeResNetwork4s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;

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
  'x10', 'x11', 'deconvf_3') ;

f = net.getParamIndex('deconvf_3') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;



%% build skip network

% skip_inputs = {'res2ax', 'res2bx', 'res2cx'};
skip_inputs = {'res2cx'};
        
[net, classifier_out] = skipNetwork(net, skip_inputs, 256, 256, ...
    nClass, opts.newLr, 'skip2');

% Add summation layer
% net.addLayer('sum4', dagnn.Sum(), ['x11', classifier_out], 'x12') ;

% net.addLayer('sum4', dagnn.Sum(), classifier_out, 'x12') ;

%% feedback BDR1
layer_in1= 'x11'; layer_in2= classifier_out{1}; poolsize=3; layer_prefix='BDR1';
[net, layer_out] = feedback_BDR(net, layer_in1, layer_in2, poolsize, layer_prefix);
net.addLayer('sum5_1', dagnn.Sum(), {layer_out, 'x11'}, 'sum_4_out') ;

%% Add deconvolution layers
filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv4', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'sum_4_out', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

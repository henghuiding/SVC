function net = fcnInitializeNetwork8s(net, varargin)
opts.rnn = false;
opts.nh = 256;
opts.nClass = 171;
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
  'x38', 'x39', 'deconvf_2') ;

f = net.getParamIndex('deconvf_2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% build skip network
skip_inputs = {'x17'};
[net, skip_classifier_out_1] = skipNetwork(net, skip_inputs, 256, nh, ...
    nClass, opts.newLr, 'skip3_1');

skip_inputs = {'x19', 'x21', 'x23'};
% skip_inputs = {};
[net, skip_classifier_out_2] = skipNetwork(net, skip_inputs, 512, nh, ...
    nClass, opts.newLr, 'skip3_2');

% Add summation layer
net.addLayer('sum3', dagnn.Sum(), ['x39', skip_classifier_out_1, ...
    skip_classifier_out_2], 'x42') ;

%     net.addLayer('Weighted_sum_3', ...
%         DagGatedsum('method', 'sum'), ...
%         ['x39', skip_classifier_out_1,skip_classifier_out_2], 'x42', 'WeightedSum_param3');
%     f = net.getParamIndex('WeightedSum_param3') ;
%     net.params(f).value = weightedSum_initialize(nClass,numel( ['x39', skip_classifier_out_1,skip_classifier_out_2]),1) ;
%     net.params(f).learningRate = 100 ;
%     net.params(f).weightDecay = 0.01 ;

%% Add deconvolution layers
filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x42', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% add conv layers to mimic the crf

% ker_size=1;
% net.addLayer('crf_2', ...
%      dagnn.Conv('size', [ker_size ker_size nClass nClass], 'pad', floor(ker_size/2)), ...
%      'prediction_1', 'prediction', {'crf_f2','crf_b2'});
% 
% f = net.getParamIndex('crf_f2') ;
% net.params(f).value = 1e-5*randn(ker_size, ker_size, nClass, nClass, 'single');
% for i=1:nClass
%     net.params(f).value(:,:,i,i)=ones(ker_size, ker_size, 1, 1, 'single');
% %     net.params(f).value(2,2,i,i)=1;
% end
% net.params(f).learningRate = 3;
% net.params(f).weightDecay = 0.1 ;
% 
% f = net.getParamIndex('crf_b2') ;
% net.params(f).value = zeros(1, 1, nClass, 'single') ;
% net.params(f).learningRate = 6 ;
% net.params(f).weightDecay = 0.1 ;

%net.addLayer('sum_prediction', dagnn.Sum(), {'prediction_2', 'prediction_1'}, 'prediction') ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

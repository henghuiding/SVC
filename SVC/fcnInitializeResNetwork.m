function net = fcnInitializeResNetwork(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16

opts.sourceModelPath = '../imagenet/imagenet-resnet-50-dag.mat' ;
opts.rnn = false;
opts.layers = 1;
opts.kerSize = 3;
opts.nh = 512;
opts.nClass = 150;
opts.recursive = false;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;
net = dagnn.DagNN.loadobj(load(opts.sourceModelPath)) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------
% Number of classes
nClass = opts.nClass;
nh = opts.nh;

net.removeLayer('prob');
% net.removeLayer('fc365');
net.removeLayer('fc1000');
net.removeLayer('pool5');
% net.removeLayer('relu5');
% net.removeLayer('bn5');

% ker_size=[15 15 15 15];sigma = [1 1 1 1];layer_prefix='LGC'; ResLayer=[316,326,336];
% for i=1:3
%     gaus_layer = sprintf('%s__gaus_%d', layer_prefix, i);
%     gaus_param_f = sprintf('%s_gaus_f_%d', layer_prefix, i);
%     gaus_f = 0.1*ones(ker_size(i), ker_size(i), 1, 512, 'single');%1e-2*randn(ker_size(i), ker_size(i), 1, nh, 'single');%
% 
%     %% gaus layer 1 Local
% %     net.addLayer(gaus_layer, ...
% %         Gaussian('size', ker_size(i), 'sigma',sigma(i)), ...
% %         bn_out, gaus_out);
%     net.addLayer(gaus_layer, ...
%         dagnn.Conv('size', [ker_size(i) ker_size(i) 1 512], 'hasBias',0, 'pad', floor(ker_size(i)/2)), ...
%         net.layers(ResLayer(i)).inputs, net.layers(ResLayer(i)).outputs, {gaus_param_f});
%     
%     f = net.getParamIndex(gaus_param_f) ;
%     net.params(f).value = bsxfun(@times, gaus_f, fspecial('gaussian', [ker_size(i),ker_size(i)], sigma(i))) ;%gaus_f;%
%     net.params(f).value = net.params(f).value -mean(net.params(f).value(:));
%     net.params(f).learningRate = opts.newLr ;
%     net.params(f).weightDecay = 1 ;
% end
% net.removeLayer('res5a_branch2b');net.removeLayer('res5b_branch2b');net.removeLayer('res5c_branch2b');
%% adapat the network

% net.addLayer('adaptation', ...
%      dagnn.Conv('size', [1 1 2048 nh], 'pad', 0), ...
%      'res5cx', 'res6x', {'adaptation_f','adaptation_b'});
% 
% f = net.getParamIndex('adaptation_f') ;
% net.params(f).value = 1e-2*randn(1, 1, 2048, nh, 'single') ;
% net.params(f).learningRate = 1 * opts.newLr;
% net.params(f).weightDecay = 1 ;
% 
% f = net.getParamIndex('adaptation_b') ;
% net.params(f).value = zeros(1, 1, nh, 'single') ;
% net.params(f).learningRate = 2 * opts.newLr ;
% net.params(f).weightDecay = 1 ;
% 
% net.addLayer('adapation_relu', ...
%         dagnn.ReLU(),...
%         'res6x', 'res6x1');


% [net, adaptation_out] = short2_skipNetwork_ReLU(net, {'res5cx'}, 2048, nh, nClass, opts.newLr, 'adaptation');
% 
% [net, adaptation_out2] = short2_skipNetwork_ReLU(net, {'res5cx'}, 2048, 256, nClass, opts.newLr, 'adaptation2');
% ker_size=17; shape_size = 15; sigma = 10; layers = numel(ker_size);
% [net, sigmoid_outs, ~] = Peer_Conv(net, adaptation_out2, ker_size, sigma, 256, shape_size*shape_size, layers, opts.newLr, 0, 'PGC');
% [net, sigmoid_outs] = skip_location_wise_conv(net, layer_out, 256, shape_size*shape_size, 1, opts.newLr, 'shapeGates');
%% build context network
% [net, ~, cn_classifier_out] = contextNetwork(net, 'res6x1', opts.kerSize,...
%     nh, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);
% [net, ~, cn_classifier_out] = CCL(net, 'res6x1', ...
%     3, nh, nh, nClass, 6, opts.newLr, 'CCL', opts.recursive,[5 5 5 5 5 5 5], 0);

% ker_size=[7 15 19 23]; sigma = [0.5 1 5 10]; layers = numel(ker_size);
% [net, ~, cn_classifier_out] = PGC_Conv(net, 'res6x1', ker_size, sigma, nh, nClass, layers, opts.newLr, 'PGC');
%% build skip network
% skip_inputs = {};
[net, adaptation_out] = short2_skipNetwork_ReLU(net, {'res5cx'}, 2048, nh, nClass, opts.newLr, 'adaptation');

ker_size=11; sigma = 5; layers = numel(ker_size);
[net, layer_out, cn_out] = PGC_Conv(net, adaptation_out, ker_size, sigma, nh, nClass, layers, opts.newLr, 1, 'PGC');


ker_size=13; local_size=3; shape_size = ker_size-2*floor(local_size/2); layers = numel(ker_size);
[net, sigmoid_outs, ~] = Peer_Conv(net, layer_out, ker_size, 1, nh, shape_size*shape_size, layers, opts.newLr, 0, 'Paired');

% build skip network
skip_inputs = {};
cn_classifier_out={};
net.addLayer('ShpCotx', ShapeContext('padsize', floor(shape_size/2), 'kersize', shape_size), [adaptation_out, sigmoid_outs], 'ShpCotx_out');
[net, cn_classifier_out] = skipNetwork(net, [adaptation_out, 'ShpCotx_out'], nh, nh, ...
    nClass, opts.newLr, 'skip_ShpCotx');


skip_inputs = {'res5ax', 'res5bx', 'res5cx'};
[net, skip_classifier_out] = skipNetwork(net, skip_inputs, 2048, nh, ...
    nClass, opts.newLr, 'skip5');

%%
% -------------------------------------------------------------------------
%  Summing layer
% -------------------------------------------------------------------------
if numel(cn_classifier_out) > 0
    net.addLayer('sum_1_1', dagnn.Sum(), [cn_classifier_out, skip_classifier_out, cn_out], 'sum_1_out') ;
    deconv_in = 'sum_1_out';
else
    error('The depth of context network must be deeper than 1.');
end

% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------


filters = single(bilinear_u(32, nClass, nClass)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 16, ...
  'crop', 8, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
   deconv_in, 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
  WeightSegmentationLoss('loss', 'idfsoftmaxlog'), ...
  {'prediction', 'label', 'classWeight'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

if 0
  figure(100) ; clf ;
  n = numel(net.vars) ;
  for i=1:n
    vl_tightsubplot(n,i) ;
    showRF(net, 'input', net.vars(i).name) ;
    title(sprintf('%s', net.vars(i).name)) ;
    drawnow ;
  end
end





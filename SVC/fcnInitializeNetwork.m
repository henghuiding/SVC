function net = fcnInitializeNetwork(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16
% opts.sourceModelPath= '../FCNs-S3/pose/lsp-mirror-crop/net-epoch-11.mat' ;

opts.sourceModelPath = '../imagenet/imagenet-vgg-verydeep-16.mat' ;
opts.rnn = false;
opts.recursive = false;
opts.layers = 1;
opts.kerSize = 3;
opts.nh = 512;
opts.nClass = 171;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;
net = load(opts.sourceModelPath) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------
% Number of classes
% nh = 512;
% nClass = 150;

nh = opts.nh;
nClass = opts.nClass;

% FCN only
% net.layers = net.layers(1:end-2);
% net.layers{32}.pad = [3, 3, 3, 3];

%% For imagenet pretrained model
net.layers = net.layers(1:end-6);
 %Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

%% For places pretrained model
% net = dagnn.DagNN.loadobj(net) ;
% net.removeLayer('prob');
% net.removeLayer('fc8a');
% net.removeLayer('drop7');
% net.removeLayer('relu7');
% net.removeLayer('fc7');
% net.removeLayer('drop6');
% net.removeLayer('relu6');
% net.removeLayer('fc6');

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

%% build context network
[net, ~, classifier_out] = contextNetwork(net, 'x31', opts.kerSize,...
    512, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);

% [net, ~, classifier_out] = contextNetwork(net, 'x35', opts.kerSize,...
%     4096, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);

% [net, classifier_out] = ASPP(net, 'x31', 512, 1024, nClass, opts.newLr);

%%
% -------------------------------------------------------------------------
%  Summing layer
% -------------------------------------------------------------------------
if numel(classifier_out) > 0
    net.addLayer('sum_1_1', dagnn.Sum(), classifier_out, 'sum_1_out') ;
    deconv_in = 'sum_1_out';
else
    error('The depth of context network must be deeper than 1.');
end


%%
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
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

%%
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

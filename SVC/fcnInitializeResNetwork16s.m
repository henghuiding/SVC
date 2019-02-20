function net = fcnInitializeResNetwork16s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
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
  'sum_1_out', 'x3', 'deconvf_1') ;

f = net.getParamIndex('deconvf_1') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% Add direct output from pool4 
switch opts.resLayer
    case 50
        skip4_input = 'res4fx'
    case 101
        skip4_input = 'res4b22x'
    case 152
        skip4_input = 'res4b35x'
end

%% build skip network
skip_inputs = {};
switch opts.resLayer
    case 50
        % 50 layer
        skip_inputs = {'res4ex', 'res4dx', 'res4cx', 'res4bx'};
        
    case 101
        % 101 layer
        for ll = 1 : 21
            if mod(ll,2) == 0
                skip_inputs{end+1} = sprintf('res4b%dx',ll);
            end
        end
    case 152
        % 152 layer
        for ll = 1 : 34
            if mod(ll,3) == 0
                skip_inputs{end+1} = sprintf('res4b%dx',ll);
            end
        end
end

skip_inputs = ['res4ax', skip_inputs, skip4_input];
[net, classifier_out] = skipNetwork(net, skip_inputs, 1024, 512, ...
    nClass, opts.newLr, 'skip4');

% Add summation layer
if numel(skip_inputs) > 0
    net.addLayer('sum2', dagnn.Sum(), ['x3', classifier_out], 'x6');
else
    net.addLayer('sum2', dagnn.Sum(), {'x3', 'x5'}, 'x6') ;
end

%% Add deconvolution layers
filters = single(bilinear_u(16, nClass, nClass)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 8, ...
  'crop', 4, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x6', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;
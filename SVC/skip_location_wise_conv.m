function [net, shpeConv_outs] = skip_location_wise_conv(net, layer_in, ...
     nh0, nh, kersize, newLr, layer_prefix)
 
 num_skips = numel(layer_in);
 shpeConv_outs = cell(1, num_skips);
 
 for i = 1 : num_skips
    conv_layer = sprintf('%s_conv_%d', layer_prefix, i);
    S_layer = sprintf('%s_S_%d',layer_prefix, i);
    bn_layer = sprintf('%s_bn_%d',layer_prefix, i);
    conv_out = sprintf('%s_conv_out_%d', layer_prefix, i);
    S_out = sprintf('%s_S_out_%d', layer_prefix, i);
    bn_out = sprintf('%s_bn_out_%d', layer_prefix, i);
    
    conv_param_f = sprintf('%s_f_%d', layer_prefix, i);
    conv_param_b = sprintf('%s_b_%d', layer_prefix, i);
    conv_f = 1e-2*randn(kersize, kersize, nh0, nh, 'single');
    conv_b = zeros(1, 1, nh, 'single');
    
    bn_in = layer_in{i};
    
        %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_%d', layer_prefix, i);
    bn_param_b = sprintf('%s_bn_b_%d', layer_prefix, i);
    bn_param_m = sprintf('%s_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        bn_in, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh0, 1, 'single') ;
    net.params(f).learningRate = 1 * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh0, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh0, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
         %% conv layer
    net.addLayer(conv_layer, ...
        dagnn.Conv('size', [kersize kersize nh0 nh], 'pad', floor(kersize/2)), ...
        bn_out, conv_out, {conv_param_f,conv_param_b});
    
    f = net.getParamIndex(conv_param_f) ;
    net.params(f).value = conv_f ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(conv_param_b) ;
    net.params(f).value = conv_b ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
%     nh=nh0;

    
    %% S
    net.addLayer(S_layer, ...
        dagnn.SoftMax(),...
        conv_out, S_out);
       
    
    %% add an output layer 
    
    shpeConv = sprintf('%s_shpeConv_%d', layer_prefix, i);
    shpeConv_out = sprintf('%s_shpeConv_out_%d', layer_prefix, i);
    
    shpeConv_param_f = sprintf('%s_shpeConv_f_%d', layer_prefix, i);
    shpeConv_param_b = sprintf('%s_shpeConv_b_%d', layer_prefix, i);
    shpeConv_f = 1e-2*randn(1, 1, nh, nh, 'single');
    shpeConv_b = zeros(1, 1, nh, 'single');
    
    net.addLayer(shpeConv, ...
        dagnn.Conv('size', [1 1 nh nh], 'pad', 0), ...
        S_out, shpeConv_out, {shpeConv_param_f,shpeConv_param_b});
    
    f = net.getParamIndex(shpeConv_param_f) ;
    net.params(f).value = shpeConv_f;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(shpeConv_param_b) ;
    net.params(f).value = shpeConv_b;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    shpeConv_outs{i} = shpeConv_out;
 end
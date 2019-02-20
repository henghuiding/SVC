function [net, layer_outs, classifier_outs] = short2_skipNetwork_Sigmoid(net, layer_in, ...
     nh0, nh, nClass, newLr, layer_prefix)
 
 num_skips = numel(layer_in);
 classifier_outs = cell(1, num_skips);
 layer_outs = cell(1, num_skips);
 for i = 1 : num_skips
    conv_layer = sprintf('%s_conv_%d', layer_prefix, i);
    sigmoid_layer = sprintf('%s_sigmoid_%d',layer_prefix, i);
    drop_layer = sprintf('%s_drop_%d',layer_prefix, i);
    bn_layer = sprintf('%s_bn_%d',layer_prefix, i);
    conv_out = sprintf('%s_conv_out_%d', layer_prefix, i);
    sigmoid_out = sprintf('%s_sigmoid_out_%d', layer_prefix, i);
    drop_out = sprintf('%s_drop_out_%d', layer_prefix, i);
    bn_out = sprintf('%s_bn_out_%d', layer_prefix, i);
    
    conv_param_f = sprintf('%s_cw_f_%d', layer_prefix, i);
    conv_param_b = sprintf('%s_cw_b_%d', layer_prefix, i);
    conv_f = 1e-2*randn(1, 1, nh0, nh, 'single');
    conv_b = zeros(1, 1, nh, 'single');
    
    conv_in = layer_in{i};
    

    
    %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_%d', layer_prefix, i);
    bn_param_b = sprintf('%s_bn_b_%d', layer_prefix, i);
    bn_param_m = sprintf('%s_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        conv_in, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
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
        dagnn.Conv('size', [1 1 nh0 nh], 'pad', 0), ...
        bn_out, conv_out, {conv_param_f,conv_param_b});
    
    f = net.getParamIndex(conv_param_f) ;
    net.params(f).value = conv_f ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(conv_param_b) ;
    net.params(f).value = conv_b ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
%     %% ReLU
%     net.addLayer(relu_layer, ...
%         dagnn.ReLU(),...
%         bn_out, relu_out);
    net.addLayer(sigmoid_layer,dagnn.mySigmoid('slope',0.1,'threshold',0),conv_out,sigmoid_out);
       
    %% dropout
%     net.addLayer(drop_layer, ...
%         dagnn.DropOut( 'rate', 0.5),...
%         relu_out, drop_out);
    
    %% add an output layer 
    
    classifier_outs{i} = sigmoid_out;
    layer_outs{i} = conv_out;
 end
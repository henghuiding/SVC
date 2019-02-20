function [net, layer_out, classifier_outs] = contextNetwork(net, layer_in, ...
    ker_size, nh0, nh, nClass, layers, newLr, layer_prefix, recursive)
classifier_outs = [];
if layers == 0
    layer_out = layer_in;
    return;
end

% shared version for recursive net

if recursive
    cw_param_f = sprintf('%s_cw_f_shared', layer_prefix);
    cw_param_b = sprintf('%s_cw_b_shared', layer_prefix);
    cw_f_shared = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
    cw_b_shared = zeros(1, 1, nh, 'single');
    
    
    classifier_f = sprintf('%s_cw_classifier_f_shared', layer_prefix);
    classifier_b = sprintf('%s_cw_classifier_b_shared', layer_prefix);
    classifier_f_shared = 1e-2*randn(1, 1, nh, nClass, 'single');
    classifier_b_shared = zeros(1, 1, nClass, 'single');
end


for i = 1 : layers
    if i == 1,
        conv_in = layer_in;
    end
        
    conv_layer = sprintf('%s_cw_conv_%d', layer_prefix, i);
    relu_layer = sprintf('%s_cw_relu_%d',layer_prefix, i);
    drop_layer = sprintf('%s_cw_drop_%d',layer_prefix, i);
    conv_out = sprintf('%s_conv_out_%d', layer_prefix, i);
    relu_out = sprintf('%s_relu_out_%d', layer_prefix, i);
    drop_out = sprintf('%s_drop_out_%d', layer_prefix, i);
    bn_layer = sprintf('%s_cw_bn_%d', layer_prefix, i);
    bn_out = sprintf('%s_bn_out_%d', layer_prefix, i);
    
    if ~recursive
        cw_param_f = sprintf('%s_cw_f_%d', layer_prefix, i);
        cw_param_b = sprintf('%s_cw_b_%d', layer_prefix, i);
        cw_f_shared = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
        cw_b_shared = zeros(1, 1, nh, 'single');
    end
    
    %% conv layer
    net.addLayer(conv_layer, ...
        dagnn.Conv('size', [ker_size ker_size nh0 nh], 'pad', floor(ker_size/2)), ...
        conv_in, conv_out, {cw_param_f,cw_param_b});
    
    f = net.getParamIndex(cw_param_f) ;
    net.params(f).value = cw_f_shared ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b) ;
    net.params(f).value = cw_b_shared ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    
    %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_%d', layer_prefix, i);
    bn_param_b = sprintf('%s_bn_b_%d', layer_prefix, i);
    bn_param_m = sprintf('%s_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        conv_out, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% ReLU
    net.addLayer(relu_layer, ...
        dagnn.ReLU(),...
        bn_out, relu_out);
    
    conv_in = relu_out; % input for next conv layer
    nh0 = nh; 
    
    %% Drop
%     net.addLayer(drop_layer, ...
%         dagnn.DropOut('rate', 0.5), ...
%         relu_out, drop_out);
    
    %% add an output layer
    
    [net, classifier_out] = skipNetwork(net, {relu_out}, ...
     nh, nh, nClass, newLr, sprintf('skip_%s_%d',layer_prefix,i));
    
%     classifier = sprintf('%s_cw_classifier_%d', layer_prefix, i);
%     classifier_out = sprintf('%s_cw_classifier_out_%d', layer_prefix, i);
%     
%     if ~recursive
%         classifier_f = sprintf('%s_cw_classifier_f_%d', layer_prefix, i);
%         classifier_b = sprintf('%s_cw_classifier_b_%d', layer_prefix, i);
%         classifier_f_shared = 1e-2*randn(1, 1, nh, nClass, 'single');
%         classifier_b_shared = zeros(1, 1, nClass, 'single');
%     end
%     
%     
%     net.addLayer(classifier, ...
%         dagnn.Conv('size', [1 1 nh nClass], 'pad', 0), ...
%         relu_out, classifier_out, {classifier_f,classifier_b});
%     
%     f = net.getParamIndex(classifier_f) ;
%     net.params(f).value = classifier_f_shared;
%     net.params(f).learningRate = 1  * newLr;
%     net.params(f).weightDecay = 1 ;
%     
%     f = net.getParamIndex(classifier_b) ;
%     net.params(f).value = classifier_b_shared;
%     net.params(f).learningRate = 2  * newLr;
%     net.params(f).weightDecay = 1 ;
%     
%     
%     classifier_out = mat2cell(classifier_out, 1);
    classifier_outs = [ classifier_outs, classifier_out];


end
layer_out = relu_out;
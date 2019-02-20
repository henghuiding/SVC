function [net, layer_out, classifier_outs] = CCL(net, layer_in, ...
    ker_size, nh0, nh, nClass, layers, newLr, layer_prefix, recursive,dilate,residual)
classifier_outs = [];
layer_out={};
if layers == 0
    layer_out = layer_in;
    return;
end

% shared version for recursive net

if recursive
    cw_param_f1 = sprintf('%s_cw_f_shared', layer_prefix);
    cw_param_b1 = sprintf('%s_cw_b_shared', layer_prefix);
    cw_f_shared1 = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
    cw_b_shared1 = zeros(1, 1, nh, 'single');
end


for i = 1 : layers
%     if i == 1,
%         layer_in = layer_in;
%     end
    %% Batch Normalization
    bn_layer = sprintf('%s_bn_%d', layer_prefix, i);
    bn_out = sprintf('%s_bn_out_%d', layer_prefix, i);

    relu_layer = sprintf('%s_relu_%d',layer_prefix, i);
    relu_out = sprintf('%s_relu_out_%d', layer_prefix, i);
    
    bn_param_f = sprintf('%s_bn_f_%d', layer_prefix, i);
    bn_param_b = sprintf('%s_bn_b_%d', layer_prefix, i);
    bn_param_m = sprintf('%s_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        layer_in, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh0, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh0, 1, 'single') ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh0, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
        
    conv_layer1 = sprintf('%s_LC_conv_%d', layer_prefix, i);
    conv_out1 = sprintf('%s_LC_out_%d', layer_prefix, i);

    if ~recursive
        cw_param_f1 = sprintf('%s_LC_f_%d', layer_prefix, i);
        cw_param_b1 = sprintf('%s_LC_b_%d', layer_prefix, i);
        cw_f_shared1 = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
        cw_b_shared1 = zeros(1, 1, nh, 'single');
    end
    

    %% conv layer 1 Local
    net.addLayer(conv_layer1, ...
        dagnn.Conv('size', [ker_size ker_size nh0 nh], 'pad', floor(ker_size/2)), ...
        bn_out, conv_out1, {cw_param_f1, cw_param_b1});
    
    f = net.getParamIndex(cw_param_f1) ;
    net.params(f).value = cw_f_shared1 ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b1) ;
    net.params(f).value = cw_b_shared1 ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    
    conv_layer2 = sprintf('%s_CT_conv_%d', layer_prefix, i);
    conv_out2 = sprintf('%s_CT_out_%d', layer_prefix, i);
    pooling_layer2 = sprintf('%s_CT_plooing_%d', layer_prefix, i);
    pooling_out2 = sprintf('%s_CT_plooing_%d', layer_prefix, i);

    if ~recursive
        cw_param_f2 = sprintf('%s_CT_f_%d', layer_prefix, i);
        cw_param_b2 = sprintf('%s_CT_b_%d', layer_prefix, i);
        cw_f_shared2 = 1e-2*randn(3, 3, nh0, nh, 'single');
        cw_b_shared2 = zeros(1, 1, nh, 'single');
    end
    
    %% average pooling
%     net.addLayer(pooling_layer2, dagnn.Pooling('poolSize',dilate(i),'pad',floor(dilate(i)/2),'stride',1,...
%         'method','avg'), bn_out, pooling_out2 );
    %% conv layer 2 Context
    net.addLayer(conv_layer2, ...
        dagnn.Conv('size', [3 3 nh0 nh], 'pad', floor((3+(dilate(i)-1)*2)/2),'dilate',dilate(i)), ...
        bn_out , conv_out2, {cw_param_f2, cw_param_b2});
    
    f = net.getParamIndex(cw_param_f2) ;
    net.params(f).value = cw_f_shared2 ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b2) ;
    net.params(f).value = cw_b_shared2 ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    

    

%% contrast    
    contrast_layer = sprintf('%s_contrast_%d', layer_prefix, i);
    contrast_out = sprintf('%s_contrast_out_%d', layer_prefix, i);
    if residual
        net.addLayer(contrast_layer, dagnn.Sum(), {conv_out1, conv_out2, layer_in}, contrast_out) ;
    else
        net.addLayer(contrast_layer, dagnn.Minus(), {conv_out1, conv_out2}, contrast_out) ;
    end
    
        %% ReLU
    net.addLayer(relu_layer, ...
        dagnn.ReLU(),...
        contrast_out, relu_out);

    layer_in = relu_out; % input for next conv layer
    nh0 = nh;
    
    %% dropout
%     net.addLayer(drop_layer, ...
%         dagnn.DropOut( 'rate', i/layers *  0.5),...
%         relu_out, drop_out);
%      
    %% add an output layer
    
    [net, classifier_out] = skipNetwork(net, {relu_out}, nh, nh, ...
    nClass, newLr, sprintf('%s_skip_%d',layer_prefix,i));
    
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
%     net.params(f).learningRate = 1 * newLr ;
%     net.params(f).weightDecay = 1 ;
%     
%     f = net.getParamIndex(classifier_b) ;
%     net.params(f).value = classifier_b_shared;
%     net.params(f).learningRate = 2  * newLr;
%     net.params(f).weightDecay = 1 ;
    
    
%     classifier_out = mat2cell(classifier_out, 1);
    classifier_outs = [classifier_outs, classifier_out];
    layer_out = [layer_out relu_out];
end

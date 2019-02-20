function [net, layer_out, classifier_outs] = Peer_Conv(net, layer_in, ...
    ker_size, sigma, nh, nk, layers, newLr, clsf, layer_prefix)
classifier_outs = [];
layer_out={};
if layers == 0
    layer_out = layer_in;
    return;
end


%% Batch Normalization
bn_layer = sprintf('%s_bn', layer_prefix);
bn_out = sprintf('%s_bn_out', layer_prefix);


bn_param_f = sprintf('%s_bn_f', layer_prefix);
bn_param_b = sprintf('%s_bn_b', layer_prefix);
bn_param_m = sprintf('%s_bn_m', layer_prefix);

net.addLayer(bn_layer, ...
    dagnn.BatchNorm(), ...
    layer_in, bn_out, {bn_param_f, bn_param_b, bn_param_m});

f = net.getParamIndex(bn_param_f) ;
net.params(f).value = ones(nh, 1, 'single') ;
net.params(f).learningRate = 1  * newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex(bn_param_b) ;
net.params(f).value = zeros(nh, 1, 'single') ;
net.params(f).learningRate = 1 * newLr ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex(bn_param_m) ;
net.params(f).value = zeros(nh, 2, 'single') ;
net.params(f).learningRate = 0  ;
net.params(f).weightDecay = 0 ;

for i = 1 : layers
    gaus_layer = sprintf('%s_Peer_%d', layer_prefix, i);
    gaus_out = sprintf('%s_out_%d', layer_prefix, i);
    gaus_param_f = sprintf('%s_Peer_f_%d', layer_prefix, i);
    gaus_f = 1e-2*randn(ker_size(i), ker_size(i), nk, nh, 'single');%1e-2*randn(ker_size(i), ker_size(i), 1, nh, 'single');%
    gaus_f = abs(gaus_f);
    
    mask = zeros(ker_size(i),ker_size(i), nk, 'single');
    ddd=1;
    for dd=2:ker_size(i)-1
        for hh=2:ker_size(i)-1
            mask(hh-1:hh+1,dd-1:dd+1,ddd)=-1;ddd=ddd+1;
        end
    end
    mask(floor(ker_size(i)/2):floor(ker_size(i)/2)+2,floor(ker_size(i)/2):floor(ker_size(i)/2)+2,:)= ...
        mask(floor(ker_size(i)/2):floor(ker_size(i)/2)+2,floor(ker_size(i)/2):floor(ker_size(i)/2)+2,:)+1;
    mask(floor(ker_size(i)/2):floor(ker_size(i)/2)+2,floor(ker_size(i)/2):floor(ker_size(i)/2)+2,round(nk/2))= 0;
    
    %% gaus layer 1 Local
    %     net.addLayer(gaus_layer, ...
    %         Gaussian('size', ker_size(i), 'sigma',sigma(i)), ...
    %         bn_out, gaus_out);
    net.addLayer(gaus_layer, ...
        dagnn.Conv('size', [ker_size(i) ker_size(i) nh nk], 'hasBias',0, 'pad', floor(ker_size(i)/2)), ...
        bn_out, gaus_out, {gaus_param_f});
    
    f = net.getParamIndex(gaus_param_f) ;
    net.params(f).value = bsxfun(@times, gaus_f, mask) ;%gaus_f;%
    net.params(f).value = permute(net.params(f).value, [1 2 4 3]);
%     mask=abs(mask);
    save('mask','mask');
%     net.params(f).value = net.params(f).value -mean(net.params(f).value(:));
    net.params(f).learningRate = newLr ;
    net.params(f).weightDecay = 1 ;

    S_layer = sprintf('%s_S%d',layer_prefix,i);
    S_out = sprintf('%s_S_out%d', layer_prefix,i);

%     %% ReLU
%     net.addLayer(relu_layer, ...
%         dagnn.ReLU(),...
%         gaus_out, relu_out);
    net.addLayer(S_layer, ...
        Gaus(),...
        gaus_out, S_out);
    %% add an output layer
    if clsf
        [net, classifier_out] = skipNetwork(net, {relu_out}, nh, nh, ...
            nClass, newLr, sprintf('%s_skip_%d',layer_prefix,i));
        classifier_outs = [classifier_outs, classifier_out];
    end
    layer_out = [layer_out S_out];
end
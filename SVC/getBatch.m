function y = getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = 512; % maximum image size of 768 x 768 pixels without poping bug in matconvnet
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 0 ;
opts.classWeights = ones(1,21,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useGpu = false ;
opts.readFromDisk = false;
opts.rgbMean = [116.4871 113.0024 104.1206]';
opts.vgg = true;
opts.stream = 'all';
opts.dataset = 'VOC';
opts = vl_argparse(opts, varargin);
opts.rgbMean = imdb.rgbMean;


if opts.prefetch
  % to be implemented
  ims = [] ;
  labels = [] ;
  return ;
end

% space for images
ims = zeros(opts.imageSize, opts.imageSize, 3, ...
  numel(images)*opts.numAugments, 'single') ;

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize ;
ly = opts.labelOffset : opts.labelStride : opts.imageSize ;
labels = zeros(numel(ly), numel(lx), 1, numel(images)*opts.numAugments, 'single');

si = 1 ;

for i=1:numel(images)

    im = imread(imdb.images.data{images(i)});
    
%     im  = single(im) / 255;
      
    if images(i) > numel(imdb.images.labels)
        % testing data for updating BN statistics
        anno = zeros(size(im,1), size(im,2), 'uint8');
    else
        labelPath = imdb.images.labels{images(i)};
        [~,~,ext] = fileparts(labelPath);
        if strcmp(ext(2:end), 'png')
            anno = imread(imdb.images.labels{images(i)});
        elseif strcmp(ext(2:end), 'mat')
            anno = load(imdb.images.labels{images(i)});
            anno = anno.LabelMap;
        end
    end
    
    % for VOC only
    if strcmp(opts.dataset, 'VOC')
        anno = mod(anno+1, 255);
    end

    
    % acquire image
    rgb = single(im);
    if size(rgb,3) == 1
        rgb = cat(3, rgb, rgb, rgb) ;
    end
    
    % crop & flip
    h = size(rgb,1) ;
    w = size(rgb,2) ;
    for ai = 1:opts.numAugments
        sz = [opts.imageSize opts.imageSize];
        scale = max(h/sz(1), w/sz(2)) ;
        scale = scale .* (1 + (rand(1)-.5)/5) ;
        
        sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
        sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
        if rand > 0.5, sx = fliplr(sx) ; end
        
        okx = find(1 <= sx & sx <= w) ;
        oky = find(1 <= sy & sy <= h) ;
        if ~isempty(opts.rgbMean)
            opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
            ims(oky,okx,:,si) = bsxfun(@minus, rgb(sy(oky),sx(okx),:), opts.rgbMean) ;
            
%             for ch = 1 : 3
%                 ims(oky,okx,ch,si) = ims(oky,okx,ch,si) / opts.std(ch) ;
%             end
        else
            ims(oky,okx,:,si) = rgb(sy(oky),sx(okx),:) ;
        end
        
        tlabels = zeros(sz(1), sz(2), 'uint8') ;
        tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
        tlabels = single(tlabels(ly,lx)) ;
        labels(:,:,1,si) = tlabels ;
        si = si + 1 ;
    end
    
end

if opts.useGpu
  ims = gpuArray(ims) ;
end
if opts.vgg
    y = {'input', ims, 'label', labels, 'classWeight', [0, opts.classWeights]};
else
    y = {'data', ims, 'label', labels, 'classWeight', [0, opts.classWeights]  };
end

%% data augmentation

% if ~opts.readFromDisk
%     im = imdb.images.data{images};
%     anno = imdb.images.labels{images};
% else
%     im = imread(imdb.images.data{images});
%     anno = imread(imdb.images.labels{images});
% end
% 
% % acquire image
% rgb = single(im);
% if size(rgb,3) == 1
%     rgb = cat(3, rgb, rgb, rgb) ;
% end
% 
% h = size(rgb,1) ;
% w = size(rgb,2) ;
% 
% % acquire imageSize divisible by 32 
% % sz = min(opts.imageSize, ceil( [h,w] /32) * 32);
% sz = [opts.imageSize opts.imageSize];
% 
% % space for images
% ims = zeros(sz(1), sz(2), 3, 'single') ;
% 
% % space for labels
% lx = opts.labelOffset : opts.labelStride : sz(2) ;
% ly = opts.labelOffset : opts.labelStride : sz(1) ;
% labels = zeros(numel(ly), numel(lx), 'single') ;
% 
% si = 1 ;
% % for i=1:numel(images)
% 
%   % crop & flip
%   for ai = 1:opts.numAugments
% %     sz = opts.imageSize(1:2) ;
%     scale = max(h/sz(1), w/sz(2)) ;
%     scale = scale .* (1 + (rand(1)-.5)/5) ;
% 
%     sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
%     sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
%     if rand > 0.5, sx = fliplr(sx) ; end
% 
%     okx = find(1 <= sx & sx <= w) ;
%     oky = find(1 <= sy & sy <= h) ;
%     if ~isempty(opts.rgbMean)
%         opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
%         ims(oky,okx,:,si) = bsxfun(@minus, rgb(sy(oky),sx(okx),:), opts.rgbMean) ;
%     else
%       ims(oky,okx,:,si) = rgb(sy(oky),sx(okx),:) ;
%     end
% 
%     tlabels = zeros(sz(1), sz(2), 'uint8') ;
%     tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
%     tlabels = single(tlabels(ly,lx)) ;
%     labels(:,:,1,si) = tlabels ;
%     si = si + 1 ;
%   end
%   
% 
% if opts.useGpu
%   ims = gpuArray(ims) ;
% end
% % y = {'input', ims, 'label', labels};
% y = {'input', ims, 'label', labels, 'classWeight', [0, opts.classWeights] };




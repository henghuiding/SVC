function imdb = updateImdbVOC()
% Update the format of VOC imdb to make it compatible with current code
% imdb.images.data
% imdb.images.labels
% imdb.rgbMean
% imdb.classFrequency

baseDir = '/home/bshuai/datasets/VOC2012/';

% load original imdb
imdb = load(fullfile(baseDir, 'imdb-train-val-update-disk.mat'));
nImgs = numel(imdb.images.name);

for i = 1: nImgs
    imdb.images.data{i} = sprintf(imdb.paths.image, imdb.images.name{i});
    imdb.images.labels{i} = sprintf(imdb.paths.classSegmentation, imdb.images.name{i});
end

%% statistics for train images
trainIdx = imdb.images.set == 1 & imdb.images.segmentation ;
nTrainImgs = sum(trainIdx);
train_data = imdb.images.data(trainIdx);
train_label = imdb.images.labels(trainIdx);

tid = ticStatus('Calculating mean Image...',1,1);
rgbMean = zeros(1,1,3);
frequency = zeros(1,22);
nPixels = 0;
for i = 1 : nTrainImgs
    I = single(imread(train_data{i}));
    [h,w,~] = size(I);
    nPixels = nPixels + h*w;
    rgbMean = rgbMean + (sum(sum(I,1),2));
    
    label = imread(train_label{i});
    % transform label
    label = mod(label + 1, 255);
    
    frequency = frequency + hist(label(:), 0:21);
    tocStatus(tid,i/nTrainImgs);
end
rgbMean = rgbMean / nPixels;


imdb.rgbMean = rgbMean;
imdb.classFrequency = frequency;
imdb.meta.sets = {'train', 'val', 'test'};


%% save
classes = imdb.classes;
classFrequency = imdb.classFrequency;
images = imdb.images;
paths = imdb.paths;
rgbMean = imdb.rgbMean;
sets = imdb.sets;
meta = imdb.meta;
save(fullfile(baseDir, 'imdb-train-val-update-disk.mat'), ...
    'classes', 'classFrequency', 'images', 'paths', 'rgbMean', ...
    'sets', 'meta');

function h = draw_label_image(im, unary_pred, gt, palette, classes)

[im_row, im_col, ~] = size(im);

if im_row >= 350
    scale = 350 / im_row;
    im = imresize(im, scale, 'bilinear');
    unary_pred = imresize(unary_pred, scale, 'nearest');
    gt = imresize(gt, scale, 'nearest');
end

labels = unique([unary_pred, gt]);
max_length = maxLength(classes(labels+1));
side_length = 75 + max_length * 9;

% one-row layout
unary_im = palette(unary_pred+1,:);
gt_im = palette(gt+1,:);

unary_im = reshape(unary_im, [size(unary_pred), 3]) * 255;
gt_im = reshape(gt_im, [size(unary_pred), 3]) * 255;

blank_col = ones([size(im,1), 5, size(im,3)]) *255;
imMerge = [im, blank_col, unary_im, blank_col, gt_im];
tmp = ones([size(imMerge,1) side_length size(imMerge,3)]) * 255;
imMerge = [imMerge tmp];
imMerge = uint8(imMerge);

show(imMerge, 2);
hold on;

for cc = 1:length(labels)
    plot([0 0],'LineWidth', 8,'Color',palette(labels(cc)+1,:));
end
LEG = legend(classes(labels+1));hold off;drawnow;
set(gcf,'PaperPositionMode','auto');
LEG.FontSize = 13;
hold off;

function ll = maxLength(classes)
% Find the maximum length of class names
ll = 0;
for i = 1 : numel(classes)
    if (length(classes{i}) > ll )
        ll = length(classes{i});
    end
end
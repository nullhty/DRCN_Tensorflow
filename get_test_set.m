%生成图片的ycbcr通道，用于测试
clear;
up_scale = 2;
folder = 'Set14';
filepaths = dir(fullfile(folder,'*.png'));
psnr_bic = 0.0;
for i = 1:length(filepaths)
    im = imread(fullfile(folder,filepaths(i).name));
    if size(im,3)==3
        im = rgb2ycbcr(im);
        im = im(:,:,1);
    end
    im_gnd = modcrop(im, up_scale);
    im_gnd = single(im_gnd)/255;

    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
    im_b = imresize(im_l, [size(im_gnd,1) size(im_gnd,2)], 'bicubic');
    name_bic = ['LRX' num2str(up_scale) '/' filepaths(i).name '.mat'];
    save(name_bic,'im_b')
    name_gnd = ['Gnd/' filepaths(i).name '.mat'];
    save(name_gnd,'im_gnd')
    %im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
    %im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
    %disp([num2str(i) ': ' num2str(compute_psnr(im_gnd,im_b))])
    %psnr_bic = psnr_bic + compute_psnr(im_gnd,im_b);
end
%psnr_bic/length(filepaths)

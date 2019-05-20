clear
dataDir = '291';%fullfile('data', '291');
%mkdir('train');
count_train = 1;
count_label = 1;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
patch_size = 41;
stride = 41;
data = zeros(patch_size, patch_size, 1, 1);
label = zeros(patch_size, patch_size, 1, 1);
down = [1,0.9,0.8,0.7];%
for f_iter = 1:numel(f_lst)
    %     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    %img_raw = imread(f_path);
    for i = 1:length(down)
        
        
        img_raw = imread(f_path);
        img_raw = rgb2ycbcr(img_raw);
        
        %img_raw = im2double(img_raw(:,:,1));
        img_raw = img_raw(:,:,1);
		
        img_raw = imresize(img_raw,down(i),'bicubic');
        
        img_size = size(img_raw);
        width = img_size(2);
        height = img_size(1);
        
        img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
        
        img_size = size(img_raw);
        
        
        x_size = (img_size(2)-patch_size)/stride+1;
        y_size = (img_size(1)-patch_size)/stride+1;
        
        img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
        img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
        img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
        
        for x = 0:x_size-1
            for y = 0:y_size-1
                x_coord = x*stride; y_coord = y*stride;
                patch_name = sprintf('train/%d',count_label);
                
                patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
                label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                
                %%save(patch_name, 'patch');
                patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
                data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_2', patch_name), 'patch');
                %patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_3', patch_name), 'patch');
                %patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_4', patch_name), 'patch');
                
                
                patch_name = sprintf('train/%d',count_label);
                
                patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
                label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %%save(patch_name, 'patch');
                patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
                data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_2', patch_name), 'patch');
                %patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_3', patch_name), 'patch');
                %patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_4', patch_name), 'patch');
                
                patch_name = sprintf('train/%d',count_label);
                
                patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
                label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %%save(patch_name, 'patch');
                patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
                data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_2', patch_name), 'patch');
                %patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_3', patch_name), 'patch');
                %patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_4', patch_name), 'patch');
                
                
                patch_name = sprintf('train/%d',count_label);
                
                patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
                label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                %label(:, :, 1, count_label) = patch;  count_label = count_label + 1;
                
                %%save(patch_name, 'patch');
                patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
                data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_2', patch_name), 'patch');
                %patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_3', patch_name), 'patch');
                %patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
                %data(:, :, 1, count_train) = patch;  count_train = count_train + 1;
                %%save(sprintf('%s_4', patch_name), 'patch');
            end
        end
    end
    
    display(count_label);
end
order = randperm(count_label-1);
data = data(:, :, 1, order);
label = label(:, :, 1, order);


%data=permute(data,[4 3 1 2]);
%label=permute(label,[4 3 1 2]);

h5create('train.h5','/data',size(data),'Datatype','single');
h5create('train.h5','/label',size(label),'Datatype','single');

h5write('train.h5','/data',data);
h5write('train.h5','/label',label);

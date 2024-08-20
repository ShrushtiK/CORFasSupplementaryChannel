parpool('local', 6);

sigma = 2.2;
beta = 4;
inhibitionFactor = 1.8;
highthresh = 0.007;

folder_path_input = '/tmp/dataset/';
folder_path_output = '/tmp/contour/';

% Get list of types (directories at level 1)
types = dir(folder_path_input);
types = types([types.isdir]); % Keep only directories
types = types(~ismember({types.name}, {'.', '..'})); % Remove '.' and '..'

for type_idx = 1:numel(types)
    type_folder = fullfile(folder_path_input, types(type_idx).name);

    % Get list of levels (directories at level 2)
    levels = dir(type_folder);
    levels = levels([levels.isdir]); % Keep only directories
    levels = levels(~ismember({levels.name}, {'.', '..'})); % Remove '.' and '..'

    for level_idx = 1:numel(levels)
        level_folder = fullfile(type_folder, levels(level_idx).name);

        % Get list of classes (directories at level 3)
        classes = dir(level_folder);
        classes = classes([classes.isdir]); 
        classes = classes(~ismember({classes.name}, {'.', '..'})); % Remove '.' and '..'

        parfor class_idx = 1:numel(classes)
            class_folder = fullfile(level_folder, classes(class_idx).name);
            output_folder = fullfile(folder_path_output, types(type_idx).name, levels(level_idx).name, classes(class_idx).name);
            if ~exist(output_folder, 'dir')
                mkdir(output_folder);
            end

            files = dir(fullfile(class_folder, '*.JPEG'));
            local_counter = 0;
            for k = 1:length(files)
                filename = files(k).name;
                filepath = fullfile(class_folder, filename);
                original_image = imread(filepath);
                try
                    [binarymap, corfresponse] = CORFContourDetection(original_image, sigma, beta, inhibitionFactor, highthresh);
                    corfresponse_normalized = mat2gray(corfresponse);
                    cmap = im2uint8(corfresponse_normalized);
                    [~, name, ~] = fileparts(filename);
                catch
                    fprintf('Error processing file: %s\n', filename);
                    continue;
                end
                imwrite(cmap, fullfile(output_folder, [name '.png']));
                local_counter = local_counter + 1;
                if mod(local_counter, 500) == 0
                    fprintf('Processed %d images in class %s\n', local_counter, class_folder.name);
                end
            end
        end
    end
end

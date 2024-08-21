sigma = 2.2;
beta = 4;
inhibitionFactor = 1.8;
highthresh = 0.007;

folder_path_input = '/tmp/dataset/';
folder_path_output = '/tmp/contour/';

subfolders = dir(folder_path_input);
subfolders = subfolders([subfolders.isdir]); 
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

display(subfolders)

counter = 0;
for folder_idx = 1:1000
    current_folder = fullfile(folder_path_input, subfolders(folder_idx).name);
    output_folder = fullfile(folder_path_output,subfolders(folder_idx).name);
    mkdir(output_folder);
    files = dir(fullfile(current_folder, '*.JPEG'));
    for k = 1:length(files)
        filename = files(k).name;
        filepath = fullfile(current_folder, filename);
        original_image = imread(filepath);
        try
            [binarymap, corfresponse] = CORFContourDetection(original_image, sigma, beta, inhibitionFactor, highthresh);
            corfresponse_normalized = mat2gray(corfresponse);
            cmap = im2uint8(corfresponse_normalized);
            [~, name, ~] = fileparts(filename);
         catch
            display('Error processing file: %s\n', filename);
            continue
        end
            imwrite(cmap, fullfile(output_folder, [name '.png']));
            counter = counter + 1;
            if mod(counter, 500) == 0
                display(num2str(counter));
            end

    end
end

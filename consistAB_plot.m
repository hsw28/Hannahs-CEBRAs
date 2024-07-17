function consistAB_plot(combined_data, combined_std)

% Data should be in the form:
% combined_data = NaN(5, 8, 5);
% combined_data(:, 1:4, :) = test_data;
% combined_data(:, 5:8, :) = control_data;
% combined_std = NaN(5, 8, 5);
% combined_std(:, 1:4, :) = test_std;
% combined_std(:, 5:8, :) = control_std;

conditions = [2, 3, 5, 7, 10];

% Set any NaN values in combined_std to 0
combined_std(isnan(combined_std)) = 0;
combined_data(isnan(combined_data)) = 0;

% Rearrange data to alternate test and control
combined_data(:, 1:8, :) = combined_data(:, [1, 5, 2, 6, 3, 7, 4, 8], :);

% Reshape the data for plotting
data_combined = [];
for cond = 1:5
    data_cond = [combined_data(1, :, cond); combined_data(2, :, cond); combined_data(3, :, cond); combined_data(4, :, cond); combined_data(5, :, cond)];
    data_cond = reshape(data_cond', 1, []);
    data_combined = [data_combined; data_cond];
end

% Generate x-axis values with spacing
x = [];
for cond = 1:5
    x = [x, (cond-1)*10 + (1:8)]; % Adding a gap of 2 units between groups
end

% Plot the 3D bar graph
figure;
h = bar3(x, data_combined', 0.8);

% Assign colors to the bars correctly
colors2 = [0, 0, 0.8; 0.3, 0.6, 1; 0.4, 0.8, 0.4; 1, 0.7, 0.4; 0.9, 0.3, 0.3]; % Dark Blue, Light Blue, Green, Orange, Red

for k = 1:length(h)
    zdata = get(h(k), 'ZData');
    [nrows, ncols] = size(zdata);
    currcolor = colors2(k,:);
    for j = 1:size(zdata, 1) / 6
        if mod(j, 2) == 1
            % Lighter version of the color for test data
            colors((j-1)*6+1:j*6, :, :) = repmat(reshape(currcolor + 0.2*(1 - currcolor), 1, 1, 3), 6, ncols, 1);
        else
            % Darker version of the color for control data
            colors((j-1)*6+1:j*6, :, :) = repmat(reshape(currcolor * 0.7, 1, 1, 3), 6, ncols, 1);
        end
    end
    h(k).CData = colors;
end

hold off;
ylabel('Condition');
xlabel('Animal Index');
zlabel('Value');
title('3D Bar Graph of Test and Control Data per Animal and Condition with Standard Deviations');
grid on;
view(3); % Set view to 3D

end

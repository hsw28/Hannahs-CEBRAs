function f = consistAB_plot(combined_data,combined_std)

%data should be in the form:
%combined_data = NaN(5, 8, 5);
%combined_data(:, 1:4, :) = test_data;
%combined_data(:, 5:8, :) = control_data;
%combined_std = NaN(5, 8, 5);
%combined_std(:, 1:4, :) = test_std;
%combined_std(:, 5:8, :) = control_std;

conditions = [2, 3, 5, 7, 10];

% Set any NaN values in combined_std to 0
combined_std(isnan(combined_std)) = 0;
combined_data(isnan(combined_data)) = 0;

% Rearrange data to alternate test and control
combined_data(:,1:8,:) = combined_data(:,[1,5,2,6,3,7,4,8],:);

% Reshape the data for plotting
data_combined = [];
for cond = 1:5
    data_cond = [combined_data(1,:,cond); combined_data(2,:,cond); combined_data(3,:,cond); combined_data(4,:,cond); combined_data(5,:,cond)];
    data_cond = reshape(data_cond', 1, []);
    data_combined = [data_combined; data_cond];
end

% Plot the 3D bar graph
figure;
h = bar3(data_combined', 0.8);

% Assign colors to the bars correctly
for k = 1:length(h)
    zdata = get(h(k), 'ZData');
    for j = 1:size(zdata, 1) / 6
        % Set the color for the control and test bars
        if mod(j, 2) == 1
            % Red for test data
            zdata((j-1)*6+1:j*6, :) = 0.9;
            set(h(k), 'CData', zdata);
        else
            % Blue for control data
            zdata((j-1)*6+1:j*6, :) = 0.1;
            set(h(k), 'CData', zdata);
        end
    end
end

% Add error bars
hold on;
[rows, cols] = size(data_combined);
for row = 1:rows
    for col = 1:cols
        y_center = data_combined(row, col);
        std_val = combined_std(ceil(row / 5), mod(col - 1, 8) + 1, ceil(row / 1));
        x_center = row;
        z_center = col;

        line([x_center x_center], [z_center z_center], [y_center - std_val y_center + std_val], 'Color', 'k', 'LineWidth', 1.5);
    end
end

hold off;
ylabel('Condition');
xlabel('Animal Index');
zlabel('Value');
title('3D Bar Graph of Test and Control Data per Animal and Condition with Standard Deviations');
grid on;
view(3); % Set view to 3D

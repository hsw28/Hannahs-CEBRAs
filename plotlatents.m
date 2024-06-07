function f = plotlatents(data_matrix, group_column, measurement_column, shuff_column)

% Assuming 'data' is your 500x9 matrix
% measurement_column is the measurement of interest
% group_column contains the group identifiers

data = data_matrix;

% Extract measurements and groups
measurements = data(:, measurement_column);  % Change the index to select a different measurement column
measurements_shuff = data(:,shuff_column);
groups = data(:, group_column);

% Initialize arrays for means and standard errors
unique_groups = unique(groups);
means = zeros(length(unique_groups), 1);
std_errors = zeros(length(unique_groups), 1);

means_shuff = zeros(length(unique_groups), 1);
std_errors_shuff = zeros(length(unique_groups), 1);

% Calculate means and standard errors for each group
for i = 1:length(unique_groups)
    group_data = measurements(groups == unique_groups(i));
    means(i) = mean(group_data);
    std_errors(i) = std(group_data) / sqrt(length(group_data));  % Standard Error

    group_data_shuff = measurements_shuff(groups == unique_groups(i));
    means_shuff(i) = mean(group_data_shuff);
    std_errors_shuff(i) = std(group_data_shuff) / sqrt(length(group_data_shuff));  % Standard Error
end

% Plotting
errorbar(unique_groups, (means), std_errors);
hold on
errorbar(unique_groups, (means_shuff), std_errors_shuff);
xlabel('Number of Latents');
ylabel('Percent Incorrect');

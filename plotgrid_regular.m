function f = plotgrid_regular(param1, param2, param3, accuracy)
    dataTable = table(param1, param2, param3, accuracy, 'VariableNames', {'Param1', 'Param2', 'Param3', 'Accuracy'});

    % Remove duplicates while calculating mean accuracy for duplicate entries
    [uniqueRows, ia, ic] = unique(dataTable(:, {'Param1', 'Param2', 'Param3'}), 'rows');
    meanAccuracy = accumarray(ic, dataTable.Accuracy, [], @mean);

    % Combine the unique rows with their corresponding mean accuracy
    uniqueData = [table2array(uniqueRows), meanAccuracy];


    % Create meshgrid for visualization
    xlin = linspace(min(uniqueData(:,1)), max(uniqueData(:,1)), 50);
    ylin = linspace(min(uniqueData(:,2)), max(uniqueData(:,2)), 50);
    zlin = linspace(min(uniqueData(:,3)), max(uniqueData(:,3)), 50);
    [X, Y, Z] = meshgrid(xlin, ylin, zlin);

    % Interpolate data onto the meshgrid
    F = scatteredInterpolant(uniqueData(:,1), uniqueData(:,2), uniqueData(:,3), uniqueData(:,4), 'linear');
    V = F(X, Y, Z);

    % Visualization using slice
    figure;
    slice(X, Y, Z, V, [], [], zlin); % Only slice along one axis for clarity
    xlabel('Parameter 1');
    ylabel('Parameter 2');
    zlabel('Parameter 3');
    %daspect([1 1 1]); % Adjust the aspect ratio to 1:1:1 or modify according to data range
    axis tight;
    lighting phong;
    shading interp;
    view(-30, 30);
    light('Position', [0 0 1], 'Style', 'infinite');
    colorbar;
    colormap('parula');
    caxis([min(uniqueData(:,4)), max(uniqueData(:,4))]);
end

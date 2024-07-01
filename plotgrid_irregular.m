function f = plotgrid_irregular(param1, param2, param3, accuracy)

  % Remove duplicates and prepare data
  dataTable = table(param1, param2, param3, accuracy, 'VariableNames', {'Param1', 'Param2', 'Param3', 'Accuracy'});
  [~, idx] = unique(dataTable(:, {'Param1', 'Param2', 'Param3'}), 'rows', 'stable');
  uniqueData = dataTable(idx, :);

  % Ensure there are no NaNs or infinite values
dataTable = rmmissing(dataTable);  % Remove any rows with missing data

% Remove duplicates and prepare data
[~, idx] = unique(dataTable(:, {'Param1', 'Param2', 'Param3'}), 'rows', 'stable');
uniqueData = dataTable(idx, :);

% Check if the data is sufficient for triangulation
if size(uniqueData, 1) < 4
    error('Insufficient unique points for 3D triangulation.');
end

% Perform triangulation
tri = delaunayTriangulation(uniqueData.Param1, uniqueData.Param2, uniqueData.Param3);

% Check the triangulation output
if isempty(tri.ConnectivityList)
    error('Triangulation failed to produce any triangles.');
end

%disp('Number of triangles:');
%disp(size(tri.ConnectivityList, 1));

% Ensure the data points for `trisurf` are referenced correctly
if size(tri.Points, 1) ~= size(uniqueData, 1)
    error('Mismatch between triangulation points and data points.');
end

% Interpolation using scattered interpolant
F = scatteredInterpolant(uniqueData.Param1, uniqueData.Param2, uniqueData.Param3, uniqueData.Accuracy, 'natural');
V = F(uniqueData.Param1, uniqueData.Param2, uniqueData.Param3);

% Plot using trisurf with the interpolated values

h = trisurf(tri.ConnectivityList, uniqueData.Param1, uniqueData.Param2, uniqueData.Param3, V);
xlabel('Number of Iterations');
ylabel('Minimum Temperature');
zlabel('Learning Rate');
title('Rat 5');
shading interp;
%light;
lighting gouraud;
material dull;
colorbar;
colormap('parula');  % Assuming 'jet' is used, or choose another


% Adjust the color axis to emphasize lower values
% Get current data range
cmin = min(V);
cmax = max(V);

% Set caxis to expand the lower value range
%caxis([cmin, cmax + (cmax - cmin) * 0.2]);  % Extend the maximum limit
%caxis([cmin *.65, cmax]) %for pos decoding
caxis([cmin*1.1, cmax *1]) %for cond decoding
caxis([cmin*1.23, cmax *1.005]) %for cond decoding rat 0307

% Alternatively, focus more sharply on lower values
%focus_factor = 0.3;  % Adjust this factor to change the focus range
%caxis([cmin, cmin + focus_factor * (cmax - cmin)]);

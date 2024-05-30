function f = plotgrid(param1, param2, param3, accuracy)


% Create 3D scatter plot
%{
figure;
scatter3(param1, param2, param3, 100, accuracy, 'filled');
xlabel('Parameter 1');
ylabel('Parameter 2');
zlabel('Parameter 3');
title('3D Scatter Plot of Grid Search Results');
colorbar;
colormap('jet');
caxis([min(accuracy) max(accuracy)]);
grid on;
%}

% Create surface plot
% Assuming param1, param2, param3 form a grid
[X, Y, Z] = ndgrid(unique(param1), unique(param2), unique(param3));
V = griddata(param1, param2, param3, accuracy, X, Y, Z);

figure;
slice(X, Y, Z, V, [], [], unique(param3));
xlabel('Parameter 1');
ylabel('Parameter 2');
zlabel('Parameter 3');
title('Surface Plot of Grid Search Results');
colorbar;
colormap('parula');
caxis([min(accuracy) max(accuracy)]);
grid on;

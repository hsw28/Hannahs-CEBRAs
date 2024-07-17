function errorbar3(X, Y, Z, E)
    % ERRORBAR3 creates a 3D error bar plot
    % X, Y, Z are the coordinates of the data points
    % E is the error value

    hold on;

    for k = 1:numel(Z)
        x = X(k);
        y = Y(k);
        z = Z(k);
        e = E(k);
        % Plot the error bar
        plot3([x x], [y y], [z-e z+e], 'k', 'LineWidth', 1.5);
    end

    hold off;
end

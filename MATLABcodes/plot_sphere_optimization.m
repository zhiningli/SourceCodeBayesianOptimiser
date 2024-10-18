function plot_sphere_optimization()
    % Define the path to the data folder relative to the MATLAB code
    data_path = fullfile('..', 'data', 'optimization_data.csv');
    
    % Read the data from the CSV file
    data = readmatrix(data_path);
    x_points = data(:, 1);  % Extract x-coordinates
    y_points = data(:, 2);  % Extract y-coordinates
    values = data(:, 3);    % Extract corresponding function values

    % Create a grid of points for plotting the Sphere function
    x = linspace(-5.12, 5.12, 400);
    y = linspace(-5.12, 5.12, 400);
    [X, Y] = meshgrid(x, y);

    % Evaluate the Sphere function over the grid
    Z = X.^2 + Y.^2;  % Sphere function: f(x, y) = x^2 + y^2

    % Create a 3D plot of the Sphere function
    figure;
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); % Surface plot
    colormap('parula');  % Use a suitable colormap for better visualization
    hold on;

    % Overlay the evaluation points from the optimization process
    scatter3(x_points, y_points, values, 50, 'r', 'filled', 'DisplayName', 'Evaluation Points');

    % Highlight the global minimum at (0, 0)
    scatter3(0, 0, 0, 100, 'w', 'x', 'LineWidth', 2, 'DisplayName', 'Global Minimum (0, 0)');

    % Set labels and title
    title('3D Sphere Function with Evaluation Points');
    xlabel('x');
    ylabel('y');
    zlabel('Function Value');
    legend('Sphere Surface', 'Evaluation Points', 'Global Minimum (0, 0)');
    view(30, -45);  % Adjust the viewing angle for better perspective

    % Enable grid for better visualization
    grid on;

    % Save the plot as an image file
    saveas(gcf, fullfile('..', 'data', 'sphere_optimization_plot.png'));
    fprintf('3D plot saved as ''sphere_optimization_plot.png'' in the data folder.\n');
end

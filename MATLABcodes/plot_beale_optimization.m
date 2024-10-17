% MATLAB script to read optimization data and plot the Beale function with evaluation points
function plot_beale_optimization()
    % Define the path to the data folder relative to the MATLAB code
    data_path = fullfile('..', 'data', 'optimization_data.csv');
    
    % Read the data from the CSV file
    data = readmatrix(data_path);
    x_points = data(:, 1);  % Extract x-coordinates
    y_points = data(:, 2);  % Extract y-coordinates
    values = data(:, 3);    % Extract corresponding function values

    % Create a grid of points for plotting the Beale function
    x = linspace(-4.5, 4.5, 400);
    y = linspace(-4.5, 4.5, 400);
    [X, Y] = meshgrid(x, y);

    % Evaluate the Beale function over the grid
    Z = (1.5 - X + X.*Y).^2 + (2.25 - X + X.*Y.^2).^2 + (2.625 - X + X.*Y.^3).^2;
    % Create a 3D plot of the Beale function
    figure;
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); % Surface plot
    colormap('parula');  % Use a suitable colormap for better visualization
    hold on;

    % Overlay the evaluation points from the optimization process
    scatter3(x_points, y_points, values, 50, 'r', 'filled', 'DisplayName', 'Evaluation Points');

    % Highlight the global minimum at (3, 0.5)
    scatter3(3, 0.5, 0, 100, 'w', 'x', 'LineWidth', 2, 'DisplayName', 'Global Minimum (3, 0.5)');

    % Set labels and title
    title('3D Beale Function with Evaluation Points');
    xlabel('x');
    ylabel('y');
    zlabel('Function Value');
    legend('Beale Surface', 'Evaluation Points', 'Global Minimum (3, 0.5)');
    view(30, -45);  % Adjust the viewing angle for better perspective

    % Enable grid for better visualization
    grid on;

    % Save the plot as an image file
    saveas(gcf, fullfile('..', 'data', 'beale_optimization_plot.png'));
    fprintf('3D plot saved as ''beale_optimization_plot.png'' in the data folder.\n');
end

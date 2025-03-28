%plan:
%- 10 for numPoints, xRange, yRange if no input
%- points must be even
%- make random points
%- make graph with points

function [points] = randPoints(numPoints,xRange, yRange)

    if nargin < 1 %10 for numPoints, xRange, yRange if no input
        numPoints = 10;
        xRange = 10;
        yRange = 10;
    elseif mod(numPoints, 2) ~= 0 %points must be even
        error('numPoints must be even');
    end
    
    %make random points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    points = [x; y]';
    disp('Random Points');
    disp(points);

    %make graph with points
    figure;
    scatter(x, y, 100, 'filled');
    xlabel('X');
    ylabel('Y');
    title(num2str(numPoints), ' Random Points');
    xlim([0 xRange]);
    ylim([0 yRange]);
    grid on;
    hold on;
end

%Next: Implement brute force

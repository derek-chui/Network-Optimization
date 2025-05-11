%plan:
%- make random points and weights
%- sort these 12 points relative to origin (1 closest)
%- make brute force triplet groupings O(n!!!) with points
%- all grouping functions
%- show all groupings on graph
%- show results in command window
%- HELPER FUNCTIONS
%- all grouping functions
%- calc score for grouping set
%- generates all unique groupings (brute force) recursively
%- plot groupings
%- get weight for results
%- print groupings
%- compare results to brute force

function NOMA3S()
    numPoints = 12;
    xRange = 10;
    yRange = 10;
    zRange = 10;

    %make random points and weights
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    weights = rand(1, numPoints) * 10;
    points = [x; y; z]';

    %sort these 12 points relative to origin (1 closest)
    dist = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist);
    points = points(sortedIndices, :);
    weights = weights(sortedIndices);

    %make brute force triplet groupings O(n!!!) with points
    pointIndices = 1:numPoints;
    groupings = makeGroupings(pointIndices);
    maxScore = -inf;
    bestTriplets = [];
    %get total scores
    for i = 1:length(groupings)
        groupingSet = groupings{i};
        score = calcScore(groupingSet, points, weights);
        if score > maxScore
            maxScore = score;
            bestTriplets = groupingSet;
        end
    end

    %all grouping functions
    set1 = [1, 12, 2; 3, 11, 4; 5, 10, 6; 7, 9, 8];
    set2 = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];

    DNOMA = DNOMAGrouping(numPoints);

    DNLUPA = DNLUPAGrouping(numPoints);

    MUG = MUGGrouping(numPoints);

    LCG = LCGGrouping(points);
    DEC = DECGrouping(points);

    greedyTriplets = greedyGrouping(points, weights);

    %show all groupings on graph
    plotGroupings(points, bestTriplets, 'Brute Force');
    plotGroupings(points, set1, 'Set 1');
    plotGroupings(points, set2, 'Set 2');
    plotGroupings(points, DNOMA, 'DNOMA');
    plotGroupings(points, DNLUPA, 'DNLUPA');
    plotGroupings(points, MUG, 'MUG');
    plotGroupings(points, LCG, 'LCG');
    plotGroupings(points, DEC, 'DEC');
    plotGroupings(points, greedyTriplets, 'Greedy');

    %show results in command window
    results.BruteForce = struct('triplets', bestTriplets, ...
        'weights', getWeight(bestTriplets, weights), ...
        'totalScore', maxScore, 'complexity', 'O(n!!!)');

    results.set1 = struct('triplets', set1, ...
        'weights', getWeight(set1, weights), ...
        'totalScore', calcScore(set1, points, weights), 'complexity', 'O(1)');

    results.set2 = struct('triplets', set2, ...
        'weights', getWeight(set2, weights), ...
        'totalScore', calcScore(set2, points, weights), 'complexity', 'O(1)');

    results.DNOMA = struct('triplets', DNOMA, ...
        'weights', getWeight(DNOMA, weights), ...
        'totalScore', calcScore(DNOMA, points, weights), 'complexity', 'O(nlogn)');

    results.DNLUPA = struct('triplets', DNLUPA, ...
        'weights', getWeight(DNLUPA, weights), ...
        'totalScore', calcScore(DNLUPA, points, weights), 'complexity', 'O(nlogn)');

    results.MUG = struct('triplets', MUG, ...
        'weights', getWeight(MUG, weights), ...
        'totalScore', calcScore(MUG, points, weights), 'complexity', 'O(n)');

    results.LCG = struct('triplets', LCG, ...
        'weights', getWeight(LCG, weights), ...
        'totalScore', calcScore(LCG, points, weights), 'complexity', 'O(nlogn)');

    results.DEC = struct('triplets', DEC, ...
        'weights', getWeight(DEC, weights), ...
        'totalScore', calcScore(DEC, points, weights), 'complexity', 'O(nlogn)');

    results.Greedy = struct('triplets', greedyTriplets, ...
        'weights', getWeight(greedyTriplets, weights), ...
        'totalScore', calcScore(greedyTriplets, points, weights), 'complexity', 'O(n^3)');

    printTripletResults(results);
    compareBruteForce(results);
end

%HELPER FUNCTIONS

%all pairing functions
function triplets = DNOMAGrouping(N)
    group1 = 1:4;
    group2 = 5:8;
    group3 = 9:12;
    triplets = [group1(1), group2(1), group3(1);
                group1(2), group2(2), group3(2);
                group1(3), group2(3), group3(3);
                group1(4), group2(4), group3(4)];
end

function triplets = DNLUPAGrouping(N)
    % indices = 1:N;
    % triplets = reshape(indices, 3, [])';
    triplets = [1,6,12; 2,5,11; 3,8,10; 4,7,9];
end

function triplets = MUGGrouping(N)
    triplets = reshape(1:N, 3, [])';
end

function triplets = LCGGrouping(points)
    [~, order] = sort(sqrt(sum(points.^2, 2)));
    triplets = reshape(order, 3, [])';
end
function triplets = DECGrouping(points)
    strengths = sum(points, 2);
    [~, sorted] = sort(strengths, 'descend');
    triplets = reshape(sorted, 3, [])';
end

function triplets = greedyGrouping(points, weights)
    indices = 1:size(points,1);
    triplets = [];
    while length(indices) >= 3
        maxScore = -inf;
        best = [];
        for i = 1:length(indices)
            for j = i+1:length(indices)
                for k = j+1:length(indices)
                    idx = [indices(i), indices(j), indices(k)];
                    p = points(idx,:);
                    w = weights(idx);
                    distSum = norm(p(1,:) - p(2,:)) + norm(p(2,:) - p(3,:)) + norm(p(1,:) - p(3,:));
                    weightDiff = max(w) - min(w);
                    score = distSum + weightDiff;
                    if score > maxScore
                        maxScore = score;
                        best = idx;
                    end
                end
            end
        end
        triplets = [triplets; best];
        indices = setdiff(indices, best);
    end
end

%calc score for grouping set
function score = calcScore(groupingSet, points, weights)
    score = 0;
    for i = 1:size(groupingSet,1)
        idx = groupingSet(i,:);
        p = points(idx,:);
        w = weights(idx);

        distSum = norm(p(1,:) - p(2,:)) + norm(p(2,:) - p(3,:)) + norm(p(1,:) - p(3,:));
        weightDiff = max(w) - min(w);
        score = score + distSum + weightDiff;
    end
end

%generates all unique groupings (brute force) recursively
function groupings = makeGroupings(indices)
    if isempty(indices)
        groupings = {[]};
        return;
    end
    groupings = {};
    first = indices(1);
    for i = 2:length(indices)
        for j = i+1:length(indices)
            rest = indices([2:i-1, i+1:j-1, j+1:end]);
            sub = makeGroupings(rest);
            for k = 1:length(sub)
                groupings{end+1} = [first, indices(i), indices(j); sub{k}];
            end
        end
    end
end

%plot groupings
function plotGroupings(points, triplets, name)
    figure;
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled');
    hold on;
    for i = 1:size(triplets,1)
        idx = triplets(i,:);
        plot3(points(idx,1), points(idx,2), points(idx,3), 'r-', 'LineWidth', 2);
        plot3([points(idx(1),1), points(idx(3),1)], ...
              [points(idx(1),2), points(idx(3),2)], ...
              [points(idx(1),3), points(idx(3),3)], 'r-', 'LineWidth', 2);
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title([name, ' Triplets']);
    grid on;
end

%get weight for results
function weightsMatrix = getWeight(triplets, weights)
    weightsMatrix = zeros(size(triplets));
    for i = 1:size(triplets, 1)
        weightsMatrix(i, :) = weights(triplets(i, :));
    end
end

%print groupings
function printTripletResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        tripletStruct = results.(key);
        triplets = tripletStruct.triplets;
        fprintf('\n%s Grouping (%s):\n', key, tripletStruct.complexity);
        disp(triplets);
        
        fprintf('Weights for each triplet:\n');
        for j = 1:size(triplets,1)
            w = tripletStruct.weights(j,:);
            fprintf('  Triplet %d: [%.2f, %.2f, %.2f]\n', j, w(1), w(2), w(3));
        end
        
        fprintf('Total Score (Distance + Weight Difference): %.4f\n\n', tripletStruct.totalScore);
    end
end

%compare results to brute force
function compareBruteForce(results)
    bruteForceScore = results.BruteForce.totalScore;
    methods = fieldnames(results);
    disp(' ');
    for i = 1:length(methods)
        method = methods{i};
        if strcmp(method, 'BruteForce')
            continue;
        end
        delta = bruteForceScore - results.(method).totalScore;
        disp(['Brute Force v ', method, '? ', num2str(delta)]);
    end
    disp('The higher the difference, the more optimal brute force is for (distance + weight difference).');
end

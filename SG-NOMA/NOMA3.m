function NOMA3()
    numPoints = 12;
    xRange = 10;
    yRange = 10;
    zRange = 10;

    %make rand pts
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    weights = rand(1, numPoints);
    points = [x; y; z]';

    %sort pts from origin
    dist = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist);
    points  = points(sortedIndices, :);
    weights = weights(sortedIndices);

    %brute force
    pointIndices = 1:numPoints;
    groupings = makeGroupings(pointIndices);
    maxScore = -inf;
    bestTriplets = [];

    for i = 1:length(groupings)
        groupingSet = groupings{i};
        score = calcScore(groupingSet, points, weights);
        if score > maxScore
            maxScore = score;
            bestTriplets = groupingSet;
        end
    end

    %grouping funcs
    set1    = [1, 12, 2; 3, 11, 4; 5, 10, 6; 7, 9, 8];
    set2    = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];
    DNOMA   = DNOMAGrouping(numPoints);
    DNLUPA  = DNLUPAGrouping(numPoints);
    MUG     = MUGGrouping(numPoints);
    LCG     = LCGGrouping(points);
    DEC     = DECGrouping(points);
    greedyTriplets = greedyGrouping(points, weights);

    plotGroupings(points, bestTriplets, 'Brute Force', weights);
    plotGroupings(points, set1, 'Set 1', weights);
    plotGroupings(points, set2, 'Set 2', weights);
    plotGroupings(points, DNOMA, 'DNOMA', weights);
    plotGroupings(points, DNLUPA, 'DNLUPA', weights);
    plotGroupings(points, MUG, 'MUG', weights);
    plotGroupings(points, LCG, 'LCG', weights);
    plotGroupings(points, DEC, 'DEC', weights);
    plotGroupings(points, greedyTriplets, 'Greedy', weights);

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

%helpers

function triplets = DNOMAGrouping(N)
    g1 = 1:4; g2 = 5:8; g3 = 9:12;
    triplets = [g1(1), g2(1), g3(1);
                g1(2), g2(2), g3(2);
                g1(3), g2(3), g3(3);
                g1(4), g2(4), g3(4)];
end

function triplets = DNLUPAGrouping(N)
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
    eta = 3;
    alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
    P = 1; N0 = 1e-4;

    while length(indices) >= 3
        bestScore = -inf;
        bestGroup = [];
        for i = 1:length(indices)
            for j = i+1:length(indices)
                for k = j+1:length(indices)
                    idx = [indices(i), indices(j), indices(k)];
                    d = vecnorm(points(idx, :)');
                    z = raylrnd(1, [1,3]);
                    h = (1 ./ d.^(eta/2)) .* z;
                    h = sort(h, 'descend');
                    R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
                    R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
                    R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
                    U  = R1 + R2 + R3;
                    w  = mean(weights(idx));
                    score = U * w;
                    if score > bestScore
                        bestScore = score;
                        bestGroup = idx;
                    end
                end
            end
        end
        triplets = [triplets; bestGroup];
        indices = setdiff(indices, bestGroup);
    end
end

function score = calcScore(groupingSet, points, weights)
    alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
    P = 1; N0 = 1e-4; eta = 3;
    score = 0;
    for i = 1:size(groupingSet, 1)
        idx = groupingSet(i, :);
        d = vecnorm(points(idx, :)');
        z = raylrnd(1, [1,3]);
        h = (1 ./ d.^(eta/2)) .* z;
        h = sort(h, 'descend');
        R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
        R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
        R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
        U  = R1 + R2 + R3;
        w  = mean(weights(idx));
        score = score + U * w;  % WEIGHTED
    end
end

%make all unique groupings recursively (w bf)
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

function plotGroupings(points, triplets, name, weights)
    figure('Color', 'w');
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled');
    hold on;
    colors = lines(size(triplets,1));
    for i = 1:size(triplets,1)
        idx = triplets(i,:);
        c = colors(i,:);
        plot3(points(idx,1), points(idx,2), points(idx,3), '-', 'LineWidth', 2, 'Color', c);
        plot3([points(idx(1),1), points(idx(3),1)], ...
              [points(idx(1),2), points(idx(3),2)], ...
              [points(idx(1),3), points(idx(3),3)], '-', 'LineWidth', 2, 'Color', c);
    end
    for i = 1:size(points,1)
        text(points(i,1), points(i,2), points(i,3), ...
            sprintf('%.2f', weights(i)), ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'w', 'EdgeColor', 'k', ...
            'BackgroundColor', 'k', ...
            'Margin', 1, ...
            'HorizontalAlignment', 'left');
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title([name, ' Triplets (Weighted)']);
    grid on;
end

%get trip weights
function weightsMatrix = getWeight(triplets, weights)
    weightsMatrix = zeros(size(triplets));
    for i = 1:size(triplets, 1)
        weightsMatrix(i, :) = weights(triplets(i, :));
    end
end

function printTripletResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        tripletStruct = results.(key);
        triplets = tripletStruct.triplets;
        fprintf('\n%s Grouping (%s):\n', key, tripletStruct.complexity);
        disp(triplets);
        fprintf('Triplet Weights:\n');
        for j = 1:size(triplets,1)
            w = tripletStruct.weights(j,:);
            fprintf('  Triplet %d: [%.2f, %.2f, %.2f]\n', j, w(1), w(2), w(3));
        end
        fprintf('Total Score (Weighted Utility): %.4f\n\n', tripletStruct.totalScore);
    end
end

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
    disp('The higher the difference, the more optimal brute force is for (weighted utility).');
end

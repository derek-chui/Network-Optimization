function NOMA2()
    numPoints = 12;
    xRange = 10;
    yRange = 10;
    zRange = 10;
    
    %make rand pts
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    points = [x; y; z]';
    r = raylrnd(1, numPoints, 1);
    weights = rand(1, numPoints);

    %sort pts from origin
    dist = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist);
    points = points(sortedIndices, :);
    r = r(sortedIndices);
    weights = weights(sortedIndices);

    %divide close & far (for hungarian/jv)
    close = 1:(numPoints/2);
    far   = (numPoints/2 + 1):numPoints;

    %jv/hungarian cost matrix
    costMatrix = zeros(length(close), length(far));
    for i = 1:length(close)
        for j = 1:length(far)
            idxA = close(i);
            idxB = far(j);
            costMatrix(i, j) = norm(points(idxA,:) - points(idxB,:));
        end
    end

    %brute force utility with weights
    pointIndices = 1:numPoints;
    pairings   = makePairings(pointIndices);
    maxUtility = -inf;
    bestPairs  = [];
    for i = 1:length(pairings)
        pairingSet = pairings{i};
        utility = calcUtility(pairingSet, points, r, weights);
        if utility > maxUtility
            maxUtility = utility;
            bestPairs = pairingSet;
        end
    end

    %pairing funcs
    set1 = [1,12; 2,11; 3,10; 4,9; 5,8; 6,7];
    set2 = [1,7; 2,8; 3,9; 4,10; 5,11; 6,12];
    DNOMA = DNOMAPairing(numPoints);
    DNLUPA = DNLUPAPairing(numPoints);
    MUG = MUGPairing(numPoints);
    LCG = LCGPairing(points);
    DEC = DECPairing(points, r);
    hungarianPairs = hungarianPairing(costMatrix, close, far);
    jvPairs = JVPairing(costMatrix, close, far);
    greedyPairs = greedyPairing(points, r, weights);

    %show results
    results.BruteForce = struct('pairs', bestPairs, ...
        'utility', maxUtility, 'complexity', 'O(n!!)');

    results.set1 = struct('pairs', set1, ...
        'utility', calcUtility(set1, points, r, weights), 'complexity', 'O(1)');

    results.set2 = struct('pairs', set2, ...
        'utility', calcUtility(set2, points, r, weights), 'complexity', 'O(1)');

    results.DNOMA = struct('pairs', DNOMA, ...
        'utility', calcUtility(DNOMA, points, r, weights), 'complexity', 'O(nlogn)');

    results.DNLUPA = struct('pairs', DNLUPA, ...
        'utility', calcUtility(DNLUPA, points, r, weights), 'complexity', 'O(nlogn)');

    results.MUG = struct('pairs', MUG, ...
        'utility', calcUtility(MUG, points, r, weights), 'complexity', 'O(n)');

    results.LCG = struct('pairs', LCG, ...
        'utility', calcUtility(LCG, points, r, weights), 'complexity', 'O(nlogn)');

    results.DEC = struct('pairs', DEC, ...
        'utility', calcUtility(DEC, points, r, weights), 'complexity', 'O(nlogn)');

    results.Hungarian = struct('pairs', hungarianPairs, ...
        'utility', calcUtility(hungarianPairs, points, r, weights), 'complexity', 'O(n^3)');

    results.JV = struct('pairs', jvPairs, ...
        'utility', calcUtility(jvPairs, points, r, weights), 'complexity', 'O(n^3)');

    results.Greedy = struct('pairs', greedyPairs, ...
        'utility', calcUtility(greedyPairs, points, r, weights), 'complexity', 'O(n^3)');

    printPairingResults(results);
    compareBruteForce(results);

    %bf plot
    figure('Color', 'w');
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled'); hold on;
    colors = lines(size(bestPairs, 1));
    for j = 1:size(bestPairs, 1)
        idx1 = bestPairs(j, 1);
        idx2 = bestPairs(j, 2);
        c = colors(j, :);
        plot3([points(idx1,1), points(idx2,1)], ...
              [points(idx1,2), points(idx2,2)], ...
              [points(idx1,3), points(idx2,3)], ...
              '-', 'LineWidth', 2, 'Color', c);
    end
    for i = 1:numPoints
        text(points(i,1), points(i,2), points(i,3), ...
            sprintf('%.2f', weights(i)), ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'w', 'EdgeColor', 'k', ...
            'BackgroundColor', 'k', ...
            'Margin', 1, ...
            'HorizontalAlignment', 'left');
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Brute Force Pairing (Weighted)');
    grid on;
    
    %greedy plot
    figure('Color', 'w');
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled'); hold on;
    colors = lines(size(bestPairs, 1));
    for j = 1:size(bestPairs, 1)
        idx1 = bestPairs(j, 1);
        idx2 = bestPairs(j, 2);
        c = colors(j, :);
        plot3([points(idx1,1), points(idx2,1)], ...
              [points(idx1,2), points(idx2,2)], ...
              [points(idx1,3), points(idx2,3)], ...
              '-', 'LineWidth', 2, 'Color', c);
    end
    for i = 1:numPoints
        text(points(i,1), points(i,2), points(i,3), ...
            sprintf('%.2f', weights(i)), ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'w', 'EdgeColor', 'k', ...
            'BackgroundColor', 'k', ...
            'Margin', 1, ...
            'HorizontalAlignment', 'left');
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Greedy Pairing (Weighted)');
    grid on;
end

%Helpers

function Unetwork = calcUtility(pairingSet, points, r, weights)
    eta = 3;
    alpha = 0.6;
    N0 = 1e-4;
    P = 1;
    Unetwork = 0;
    for j = 1:size(pairingSet, 1)
        idx1 = pairingSet(j, 1);
        idx2 = pairingSet(j, 2);
        d1 = norm(points(idx1,:));
        d2 = norm(points(idx2,:));
        h1 = (1 / (d1^(eta/2))) * r(idx1);
        h2 = (1 / (d2^(eta/2))) * r(idx2);
        R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
        R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
        w  = mean([weights(idx1), weights(idx2)]);
        Unetwork = Unetwork + (R1 + R2) * w;
    end
end

function pairs = greedyPairing(points, r, weights)
    N = size(points, 1);
    remaining = 1:N;
    pairs = [];
    eta = 3;
    alpha = 0.6;
    P = 1;
    N0 = 1e-4;
    while numel(remaining) > 1
        bestU = -inf;
        bestPair = [];
        for i = 1:numel(remaining)
            for j = i+1:numel(remaining)
                idx1 = remaining(i); idx2 = remaining(j);
                d1 = norm(points(idx1,:));
                d2 = norm(points(idx2,:));
                h1 = (1 / (d1^(eta/2))) * r(idx1);
                h2 = (1 / (d2^(eta/2))) * r(idx2);
                R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
                R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
                w  = mean([weights(idx1), weights(idx2)]);
                U  = (R1 + R2) * w;
                if U > bestU
                    bestU = U;
                    bestPair = [idx1, idx2];
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = DNOMAPairing(N)
    g1 = 1:floor(N/4);
    g2 = (floor(N/4)+1):floor(N/2);
    g3 = (floor(N/2)+1):floor(3*N/4);
    g4 = (floor(3*N/4)+1:N);
    oddPairs = [];
    len = min(length(g1), length(g3));
    for i = 1:len
        oddPairs = [oddPairs; g1(i), g3(i)];
    end
    evenPairs = [];
    len = min(length(g2), length(g4));
    for i = 1:len
        evenPairs = [evenPairs; g2(i), g4(i)];
    end
    pairs = [oddPairs; evenPairs];
end

function pairs = DNLUPAPairing(N)
    group1 = 1:floor(N/2);
    group2 = (floor(N/2)+1):N;
    pairs = [];
    for i = 1:length(group1)
        pairs = [pairs; group1(i), group2(end-i+1)];
    end
end

function pairs = MUGPairing(N)
    pairs = [];
    for i = 1:2:N-1
        pairs = [pairs; i, i+1];
    end
    if mod(N,2) ~= 0
        pairs(end,2) = N;
    end
end

function pairs = LCGPairing(points)
    N = size(points, 1);
    [~, order] = sort(sqrt(sum(points.^2, 2)));
    pairs = [];
    for i = 1:2:N-1
        pairs = [pairs; order(i), order(i+1)];
    end
end

function pairs = DECPairing(points, r)
    eta = 3;
    N = size(points, 1);
    gains = zeros(N, 1);
    for i = 1:N
        d = norm(points(i,:));
        gains(i) = (1 / (d^(eta/2))) * r(i);
    end
    [~, strong] = sort(gains, 'descend');
    [~, weak]   = sort(gains, 'ascend');
    pairs = [];
    for i = 1:N/2
        pairs = [pairs; strong(i), weak(i)];
    end
end

function pairs = hungarianPairing(costMatrix, close, far)
    maxVal     = max(costMatrix(:));
    costMatrix = maxVal - costMatrix;
    [match, ~] = matchpairs(costMatrix, 1e6);
    pairs = [close(match(:,1))', far(match(:,2))'];
end

function pairs = JVPairing(costMatrix, close, far)
    costMatrix = -costMatrix;
    [match, ~] = matchpairs(costMatrix, -1);
    pairs = [];
    for i = 1:size(match,1)
        pairs = [pairs; close(match(i,1)), far(match(i,2))];
    end
end

function pairings = makePairings(indices)
    if isempty(indices)
        pairings = {[]}; return;
    end
    pairings = {};
    first = indices(1);
    for i = 2:length(indices)
        second    = indices(i);
        remaining = indices([2:i-1, i+1:end]);
        subPairings = makePairings(remaining);
        for j = 1:length(subPairings)
            pairing = [[first, second]; subPairings{j}];
            pairings{end+1} = pairing;
        end
    end
end

function printPairingResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        pairStruct = results.(key);
        disp([key, ' Set ', pairStruct.complexity, ':']);
        disp(pairStruct.pairs);
        disp(['Utility (weighted): ', num2str(pairStruct.utility)]);
    end
end

function compareBruteForce(results)
    bruteForceUtility = results.BruteForce.utility;
    methods = fieldnames(results);
    fprintf('\n%-20s %-20s\n', 'Algorithm', 'vs BF Utility');
    fprintf('-----------------------------------------\n');
    for i = 1:length(methods)
        method = methods{i};
        if strcmp(method, 'BruteForce')
            continue;
        end
        methodUtility = results.(method).utility;
        deltaUtility  = bruteForceUtility - methodUtility;
        fprintf('%-20s %-20.4f\n', method, deltaUtility);
    end
    fprintf('\nThe higher the number, the more effective Brute Force is compared to the other algorithm (weighted utility).\n');
end

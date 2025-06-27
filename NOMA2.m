
function NOMA2()
    numPoints = 12;
    xRange = 10;
    yRange = 10;
    zRange = 10;
    
    %make random points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    points = [x; y; z]';
    r = raylrnd(1, numPoints, 1);

    %sort these 10 points relative to origin (1 closest)
    dist = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist);
    points = points(sortedIndices, :);
    r = r(sortedIndices);

    %cost matrix for hungarian

    %divide into close and far
    close = 1:(numPoints/2);
    far = (numPoints/2 + 1):numPoints;
    %build cost matrix Cij = dist(i,j) or Ri + Rj (for now, use Euclidean dist)
    costMatrix = zeros(length(close), length(far)); %#close x #far with 0
    for i = 1:length(close) %iterate close and far users
        for j = 1:length(far)
            idxA = close(i); %get user indices
            idxB = far(j);
            %dist = cost of pairing two users
            costMatrix(i, j) = norm(points(idxA,:) - points(idxB,:));
        end
    end

    %make utility based brute force pairing O(n^(n/2)) with points
    pointIndices = 1:numPoints;
    pairings = makePairings(pointIndices);
    maxUtility = -inf;
    bestPairs = [];
    for i = 1:length(pairings)
        pairingSet = pairings{i};
        utility = calcUtility(pairingSet, points, r);
        if utility > maxUtility
            maxUtility = utility;
            bestPairs = pairingSet;
        end
    end


    %all pairing functions
    set1 = [1,12; 2,11; 3,10; 4,9; 5,8; 6,7];
    set2 = [1,7; 2,8; 3,9; 4,10; 5,11; 6,12];
    DNOMA = DNOMAPairing(numPoints);
    DNLUPA = DNLUPAPairing(numPoints);
    MUG = MUGPairing(numPoints);
    LCG = LCGPairing(points);
    DEC = DECPairing(points, r);
    hungarianPairs = hungarianPairing(costMatrix, close, far);
    jvPairs = JVPairing(costMatrix, close, far);
    greedyPairs = greedyPairing(points);
    
    %show results in command window
    results.BruteForce     = struct('pairs', bestPairs,        'utility', maxUtility,         'complexity', 'O(n!!)');
    results.set1           = struct('pairs', set1,             'utility', calcUtility(set1, points, r),             'complexity', 'O(1)');
    results.set2           = struct('pairs', set2,             'utility', calcUtility(set2, points, r),             'complexity', 'O(1)');
    results.DNOMA          = struct('pairs', DNOMA,            'utility', calcUtility(DNOMA, points, r),            'complexity', 'O(nlogn)');
    results.DNLUPA         = struct('pairs', DNLUPA,           'utility', calcUtility(DNLUPA, points, r),           'complexity', 'O(nlogn)');
    results.MUG            = struct('pairs', MUG,              'utility', calcUtility(MUG, points, r),              'complexity', 'O(n)');
    results.LCG            = struct('pairs', LCG,              'utility', calcUtility(LCG, points, r),              'complexity', 'O(nlogn)');
    results.DEC            = struct('pairs', DEC,              'utility', calcUtility(DEC, points, r),              'complexity', 'O(nlogn)');
    results.Hungarian      = struct('pairs', hungarianPairs,   'utility', calcUtility(hungarianPairs, points, r),   'complexity', 'O(n^3)');
    results.JV             = struct('pairs', jvPairs,          'utility', calcUtility(jvPairs, points, r),          'complexity', 'O(n^3)');
    results.Greedy         = struct('pairs', greedyPairs,      'utility', calcUtility(greedyPairs, points, r),      'complexity', 'O(n^3)');

    
    printPairingResults(results);
    compareBruteForce(results);

    %display graphs
    filename = 'pairing_animation.gif';
    hFig = figure;
    methods = fieldnames(results);
    for frameIdx = 1:3
        for i = 1:length(methods)
            method = methods{i};
            pairs = results.(method).pairs;
            clf(hFig);
            scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled');
            hold on;
            for j = 1:size(pairs, 1)
                idx1 = pairs(j, 1);
                idx2 = pairs(j, 2);
                plot3([points(idx1,1), points(idx2,1)], ...
                      [points(idx1,2), points(idx2,2)], ...
                      [points(idx1,3), points(idx2,3)], ...
                      'r-', 'LineWidth', 2);
            end
            xlabel('X'); ylabel('Y'); zlabel('Z');
            title([method, ' Pairing']);
            grid on;
            frame = getframe(hFig);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            if i == 1 && frameIdx == 1
                imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 1);
            else
                imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
            end
            pause(1);
        end
    end
end

%HELPER FUNCTIONS

%all pairing functions
function pairs = DNOMAPairing(N)
    %divide into 4 groups
    %FNN (First Nearest Near)
    g1 = 1:floor(N/4);
    %SNN (Second Nearest Near)
    g2 = (floor(N/4)+1):floor(N/2);
    %FFF (First Farthest Far)
    g3 = (floor(N/2)+1):floor(3*N/4);
    %SFF (Second Farthest Far)
    g4 = (floor(3*N/4)+1:N);
    %pair g1 with g3 (odd)
    oddPairs = [];
    len = min(length(g1), length(g3));
    for i = 1:len
        oddPairs = [oddPairs; g1(i), g3(i)];
    end
    %pair g2 with g4 (even)
    evenPairs = [];
    len = min(length(g2), length(g4));
    for i = 1:len
        evenPairs = [evenPairs; g2(i), g4(i)];
    end
    pairs = [oddPairs; evenPairs]; %combine
end
function pairs = DNLUPAPairing(N)
    %divide into 2 groups
    group1 = 1:floor(N/2);
    group2 = (floor(N/2)+1):N;
    %pair first in g1 with last in g2 etc
    pairs = [];
    for i = 1:length(group1)
        pairs = [pairs; group1(i), group2(end-i+1)];
    end
end
function pairs = MUGPairing(N)
    %pairing neighbours
    pairs = [];
    for i = 1:2:N-1 %1,3,5
        pairs = [pairs; i, i+1]; %pair 1 w/ 2, 3 w/ 4
    end
    %if odd, last point pairs with nearest available earlier point
    if mod(N,2) ~= 0
        pairs(end,2) = N;
    end
end
function pairs = LCGPairing(points)
    N = size(points, 1);
    [~, order] = sort(sqrt(sum(points.^2, 2))); %sort by dist to origin
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
        gains(i) = (1 / (d^(eta/2))) * r(i); % true channel gain
    end
    [~, strong] = sort(gains, 'descend');
    [~, weak] = sort(gains, 'ascend');
    pairs = [];
    for i = 1:N/2
        pairs = [pairs; strong(i), weak(i)];
    end
end
function pairs = hungarianPairing(costMatrix, close, far)
    %invert and minimize tot cost, far = small cost, close = large cost
    maxVal = max(costMatrix(:));
    costMatrix = maxVal - costMatrix;
    [match, ~] = matchpairs(costMatrix, 1e6); %make unmatch pairings expensive, so all should be matched
    pairs = [close(match(:,1))', far(match(:,2))']; %close + far arrays
end
function pairs = JVPairing(costMatrix, close, far)
    %matchpairs from optimization toolbox / custom LAP
    %max score to min negative score
    costMatrix = -costMatrix;
    [match, ~] = matchpairs(costMatrix, -1); %-1 for max
    pairs = [];
    for i = 1:size(match,1) %pair clusters, match close row + far column
        pairs = [pairs; close(match(i,1)), far(match(i,2))];
    end
end
function pairs = greedyPairing(points)
    N = size(points, 1);
    indices = 1:N;
    pairs = [];
    while length(indices) > 1
        maxDist = -inf;
        bestPair = [];
        %brute force check all remaining possible point pairs
        for i = 1:length(indices)
            for j = i+1:length(indices)
                idx1 = indices(i);
                idx2 = indices(j);
                dist = norm(points(idx1,:) - points(idx2,:));
                if dist > maxDist %update if better
                    maxDist = dist;
                    bestPair = [idx1, idx2];
                end
            end
        end
        %save best pair, remove from list
        pairs = [pairs; bestPair];
        indices(indices == bestPair(1)) = [];
        indices(indices == bestPair(2)) = [];
    end
end

%calc utility for pairing set
function Unetwork = calcUtility(pairingSet, points, r)
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
        Unetwork = Unetwork + (R1 + R2);
    end
end

%generates all unique pairings (brute force) recursively
function pairings = makePairings(indices)
    if isempty(indices) %base case
        pairings = {[]};
        return;
    end
    pairings = {};
    first = indices(1);
    for i = 2:length(indices) %loop over all possible choices for pairing with first
        second = indices(i);
        remaining = indices([2:i-1, i+1:end]); %exclude 1 2, get remaining
        subPairings = makePairings(remaining); %all possible pairings with remaining (recursive)
        for j = 1:length(subPairings) %each pair from recursion
            pairing = [[first, second]; subPairings{j}]; %get [1 3; 2 5; 4 6] like example above
            pairings{end+1} = pairing; %save this set to list of all possible pairings [Set 1], [Set 2], etc.
        end
    end
end

%print pairings
function printPairingResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        pairStruct = results.(key);
        disp([key, ' Set ', pairStruct.complexity, ':']);
        disp(pairStruct.pairs);
        disp(['Utility: ', num2str(pairStruct.utility)]);
    end
end

%compare results to brute force
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
        deltaUtility = bruteForceUtility - methodUtility;
        fprintf('%-20s %-20.4f\n', method, deltaUtility);
    end
    fprintf('\nThe higher the number, the more effective Brute Force is compared to the other algorithm.\n');
end


    rng('shuffle');

    numPoints = 12;
    if mod(numPoints,2) ~= 0
        error('numPoints must be even for NOMA2Q2 (pairing).');
    end

    xMin = -5; xMax = 5;
    yMin = -5; yMax = 5;
    zMin = -5; zMax = 5;

    x = xMin + (xMax - xMin) * rand(1, numPoints);
    y = yMin + (yMax - yMin) * rand(1, numPoints);
    z = zMin + (zMax - zMin) * rand(1, numPoints);

    weights = rand(1, numPoints);
    points  = [x; y; z]';

    fading = raylrnd_local(1, [1, numPoints]);

    dist0 = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist0, 'ascend');
    points  = points(sortedIndices, :);
    weights = weights(sortedIndices);
    fading  = fading(sortedIndices);

    eta   = 3;
    Ptx   = 1;
    N0    = 1e-4;
    alpha = 0.6;

    pointIndices = 1:numPoints;
    pairings = makePairings(pointIndices);

    maxScore = -inf; bestPairs = [];
    for i = 1:length(pairings)
        S = calcScore(pairings{i}, points, weights, fading, eta, alpha, Ptx, N0);
        if S > maxScore
            maxScore  = S;
            bestPairs = pairings{i};
        end
    end

    set1   = Set1Pairs(numPoints);
    set2   = Set2Pairs(numPoints);
    DNOMA  = DNOMAPairs(numPoints);
    DNLUPA = DNLUPAPairs(numPoints);
    MUG    = MUGPairs(numPoints);
    LCG    = LCGPairs(points);
    DEC    = DECPairs(points);

    greedyPairs = greedyPairing(points, weights, fading, eta, alpha, Ptx, N0);

    qParams = struct( ...
        'episodes',      1500, ...
        'alpha',         0.35, ...
        'gamma',         0.92, ...
        'epsilon_start', 0.90, ...
        'epsilon_end',   0.05, ...
        'epsilon_decay', 0.995, ...
        'log_every',     100, ...
        'verbose',       true ...
    );

    [qPairs, qScore] = qLearningPairing(points, weights, fading, eta, alpha, Ptx, N0, qParams);

    plotPairs(points, bestPairs,  'Brute Force', weights);
    % plotPairs(points, set1,       'Set 1',       weights);
    % plotPairs(points, set2,       'Set 2',       weights);
    % plotPairs(points, DNOMA,      'DNOMA',       weights);
    % plotPairs(points, DNLUPA,     'DNLUPA',      weights);
    % plotPairs(points, MUG,        'MUG',         weights);
    % plotPairs(points, LCG,        'LCG',         weights);
    % plotPairs(points, DEC,        'DEC',         weights);
    plotPairs(points, greedyPairs,'Greedy',     weights);
    plotPairs(points, qPairs,     'Q-Learning', weights);

    results.BruteForce = struct('pairs', bestPairs, ...
        'weights', getWeight(bestPairs, weights), ...
        'totalScore', maxScore, 'complexity', 'O((n-1)!!)' );

    results.Set1 = struct('pairs', set1, ...
        'weights', getWeight(set1, weights), ...
        'totalScore', calcScore(set1, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(1)');

    results.Set2 = struct('pairs', set2, ...
        'weights', getWeight(set2, weights), ...
        'totalScore', calcScore(set2, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(1)');

    results.DNOMA = struct('pairs', DNOMA, ...
        'weights', getWeight(DNOMA, weights), ...
        'totalScore', calcScore(DNOMA, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(n)');

    results.DNLUPA = struct('pairs', DNLUPA, ...
        'weights', getWeight(DNLUPA, weights), ...
        'totalScore', calcScore(DNLUPA, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(n)');

    results.MUG = struct('pairs', MUG, ...
        'weights', getWeight(MUG, weights), ...
        'totalScore', calcScore(MUG, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(n)');

    results.LCG = struct('pairs', LCG, ...
        'weights', getWeight(LCG, weights), ...
        'totalScore', calcScore(LCG, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(n^2)');

    results.DEC = struct('pairs', DEC, ...
        'weights', getWeight(DEC, weights), ...
        'totalScore', calcScore(DEC, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(nlogn)');

    results.Greedy = struct('pairs', greedyPairs, ...
        'weights', getWeight(greedyPairs, weights), ...
        'totalScore', calcScore(greedyPairs, points, weights, fading, eta, alpha, Ptx, N0), 'complexity', 'O(n^3)');

    results.QLearning = struct('pairs', qPairs, ...
        'weights', getWeight(qPairs, weights), ...
        'totalScore', qScore, 'complexity', 'O(E·n^2)');

    printPairResults(results);
    compareBruteForce(results);
end

function pairs = Set1Pairs(N)
    pairs = [(1:N/2)', (N:-1:N/2+1)'];
end

function pairs = Set2Pairs(N)
    pairs = reshape(1:N, 2, [])';
end

function pairs = DNOMAPairs(N)
    idx = 1:N;
    G1 = idx(1:N/4);
    G2 = idx(N/4+1:N/2);
    G3 = idx(N/2+1:3*N/4);
    G4 = idx(3*N/4+1:end);
    pairs = [G1(:), G3(:); G2(:), G4(:)];
end

function pairs = DNLUPAPairs(N)
    if N ~= 12
        pairs = Set1Pairs(N);
        return;
    end
    pairs = [1, 6;
             2, 5;
             3, 8;
             4, 7;
             9, 12;
             10, 11];
end

function pairs = MUGPairs(N)
    pairs = reshape(1:N, 2, [])';
end

function pairs = LCGPairs(points)
    N = size(points,1);
    remaining = 1:N;
    pairs = [];
    while numel(remaining) > 1
        bestDist = inf;
        bestPair = [];
        for ii = 1:numel(remaining)
            for jj = ii+1:numel(remaining)
                d = norm(points(remaining(ii),:) - points(remaining(jj),:));
                if d < bestDist
                    bestDist = d;
                    bestPair = [remaining(ii), remaining(jj)];
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = DECPairs(points)
    strengths = sum(points, 2);
    [~, sorted] = sort(strengths, 'descend');
    pairs = reshape(sorted, 2, [])';
end

function pairings = makePairings(idx)
    if isempty(idx)
        pairings = {[]};
        return;
    end
    first = idx(1);
    rest  = idx(2:end);

    pairings = {};
    for k = 1:length(rest)
        second = rest(k);
        remaining = rest;
        remaining(k) = [];
        subs = makePairings(remaining);
        for jj = 1:length(subs)
            pairings{end+1} = [[first, second]; subs{jj}];
        end
    end
end

function pairs = greedyPairing(points, weights, fading, eta, alpha, Ptx, N0)
    N = size(points,1);
    remaining = 1:N;
    pairs = [];
    while numel(remaining) > 1
        bestU = -inf;
        bestPair = [];
        for ii = 1:numel(remaining)
            for jj = ii+1:numel(remaining)
                pr = [remaining(ii), remaining(jj)];
                u = pairUtility(pr, points, weights, fading, eta, alpha, Ptx, N0);
                if u > bestU
                    bestU = u;
                    bestPair = pr;
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function score = calcScore(pairs, points, weights, fading, eta, alpha, Ptx, N0)
    score = 0;
    for i = 1:size(pairs, 1)
        score = score + pairUtility(pairs(i,:), points, weights, fading, eta, alpha, Ptx, N0);
    end
end

function plotPairs(points, pairs, name, weights)
    figW = 6;
    figH = 4;

    figure('Color','w', ...
           'Units','inches', ...
           'Position',[0.5 0.5 figW figH], ...
           'Renderer','painters');

    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled'); hold on;
    colors = lines(size(pairs,1));

    for i = 1:size(pairs,1)
        idx = pairs(i,:);
        c = colors(i,:);
        plot3(points(idx,1), points(idx,2), points(idx,3), '-', 'LineWidth', 2, 'Color', c);

        [~, posStrong] = min(vecnorm(points(idx, :)', 2));
        sIdx = idx(posStrong);
        plot3(points(sIdx,1), points(sIdx,2), points(sIdx,3), 'go', 'MarkerSize', 14, 'LineWidth', 2);
    end

    for i = 1:size(points,1)
        text(points(i,1), points(i,2), points(i,3), ...
            sprintf('%.2f', weights(i)), ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'w', 'EdgeColor', 'k', ...
            'BackgroundColor', 'k', 'Margin', 1, ...
            'HorizontalAlignment', 'left');
    end

    plot3(0, 0, 0, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

    xlim([-5 5]); ylim([-5 5]); zlim([-5 5]);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title([name, ' Pairs (Weighted)']);
    grid on; box on;

    set(gcf,'PaperUnits','inches');
    set(gcf,'PaperSize',[figW figH]);
    set(gcf,'PaperPosition',[0 0 figW figH]);
    set(gcf,'PaperPositionMode','manual');
    set(gcf,'InvertHardcopy','off');

    base = ['NOMA2Q2_' regexprep(name,'[^a-zA-Z0-9]+','_')];

    print(gcf, '-dpdf',  '-painters', [base '.pdf']);
    print(gcf, '-depsc', '-painters', '-loose', [base '.eps']);
    try
        print(gcf, '-dsvg', '-painters', [base '.svg']);
    catch
    end
end

function weightsMatrix = getWeight(pairs, weights)
    weightsMatrix = zeros(size(pairs));
    for i = 1:size(pairs, 1)
        weightsMatrix(i, :) = weights(pairs(i, :));
    end
end

function printPairResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        s = results.(key);
        pr = s.pairs;
        fprintf('\n%s Pairing (%s):\n', key, s.complexity);
        disp(pr);
        fprintf('Pair Weights:\n');
        for j = 1:size(pr,1)
            w = s.weights(j,:);
            fprintf('  Pair %d: [%.2f, %.2f]\n', j, w(1), w(2));
        end
        fprintf('Total Score (Weighted Utility): %.6f\n\n', s.totalScore);
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
    disp('The higher the difference, the more optimal brute force is (weighted utility).');
end

function [pairs, episodeBestScore] = qLearningPairing(points, weights, fading, eta, alphaNoma, Ptx, N0, p)
    if ~isfield(p,'log_every'), p.log_every = max(1, floor(p.episodes/20)); end
    if ~isfield(p,'verbose'), p.verbose = true; end

    N = size(points,1);
    Q = containers.Map('KeyType','char','ValueType','double');

    episodeBestScore = -inf;
    episodeBestPairs = [];

    eps = p.epsilon_start;

    for ep = 1:p.episodes
        remaining = true(1,N);
        current   = [];   % size 0 or 1
        cumScore  = 0;
        tracePairs = [];

        while any(remaining)
            sKey = makeStateKey_pair(remaining, current);
            actions = find(remaining);

            if rand < eps
                a = actions(randi(length(actions)));
            else
                a = argmaxQ_pair(Q, sKey, actions);
            end

            remaining(a) = false;
            current = [current, a];

            reward = 0;
            if numel(current) == 2
                reward = pairUtility(current, points, weights, fading, eta, alphaNoma, Ptx, N0);
                cumScore = cumScore + reward;
                tracePairs = [tracePairs; current];
                current = [];
            end

            sNextKey = makeStateKey_pair(remaining, current);
            nextActions = find(remaining);

            if isempty(nextActions)
                maxQnext = 0;
            else
                maxQnext = maxQ_pair(Q, sNextKey, nextActions);
            end

            qaKey = makeQAKey_pair(sKey, a);
            oldQ = 0;
            if isKey(Q, qaKey), oldQ = Q(qaKey); end
            Q(qaKey) = (1 - p.alpha)*oldQ + p.alpha*(reward + p.gamma*maxQnext);
        end

        if cumScore > episodeBestScore
            episodeBestScore = cumScore;
            episodeBestPairs = tracePairs;
        end

        eps = max(p.epsilon_end, eps * p.epsilon_decay);

        if p.verbose && (ep == 1 || ep == p.episodes || mod(ep, p.log_every) == 0)
            fprintf('Episode %4d/%4d | episode utility = %.6f | best = %.6f\n', ...
                ep, p.episodes, cumScore, episodeBestScore);
            drawnow limitrate
        end
    end

    remaining = true(1,N);
    current = [];
    pairs = [];
    while any(remaining)
        sKey = makeStateKey_pair(remaining, current);
        actions = find(remaining);
        a = argmaxQ_pair(Q, sKey, actions);

        remaining(a) = false;
        current = [current, a];

        if numel(current) == 2
            pairs = [pairs; current];
            current = [];
        end
    end

    if isempty(pairs) || size(pairs,2) ~= 2 || size(pairs,1) ~= N/2
        pairs = episodeBestPairs;
    end
end

function key = makeStateKey_pair(remaining, current)
    r = char('0' + remaining);
    if isempty(current)
        c = '[]';
    else
        c = sprintf('%02d', current(1));
    end
    key = [r,'|',c];
end

function k = makeQAKey_pair(sKey, a)
    k = [sKey,'|a:',num2str(a)];
end

function a = argmaxQ_pair(Q, sKey, actions)
    bestVal = -inf; a = actions(1);
    for i = 1:length(actions)
        k = makeQAKey_pair(sKey, actions(i));
        v = 0; if isKey(Q, k), v = Q(k); end
        if v > bestVal
            bestVal = v; a = actions(i);
        end
    end
end

function m = maxQ_pair(Q, sKey, actions)
    vals = zeros(1,length(actions));
    for i = 1:length(actions)
        k = makeQAKey_pair(sKey, actions(i));
        if isKey(Q, k), vals(i) = Q(k); end
    end
    if isempty(vals), m = 0; else, m = max(vals); end
end

function rew = pairUtility(idx, points, weights, fading, eta, alpha, P, N0)
    d = vecnorm(points(idx, :)');
    h = (1 ./ d.^(eta/2)) .* fading(idx);

    [h_sorted, ord] = sort(h, 'ascend');
    h_w = h_sorted(1);
    h_s = h_sorted(2);

    gamma_w = (alpha * P * h_w^2) / ((1 - alpha) * P * h_w^2 + N0);
    gamma_s = ((1 - alpha) * P * h_s^2) / N0;

    a1 = 0.30; a2 = 0.98; c1 = 0.25; c2 = -0.8;
    xi_w = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_w + c2)));
    xi_s = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_s + c2)));

    kk = weights(idx);
    kk = kk(ord);
    k_w = kk(1);
    k_s = kk(2);

    rew = k_w * xi_w * log2(1 + gamma_w) + k_s * xi_s * log2(1 + gamma_s);
end

function r = raylrnd_local(b, sz)
    if numel(sz) == 1, sz = [1, sz]; end
    u = max(eps, rand(sz));
    r = b .* sqrt(-2 .* log(u));
end

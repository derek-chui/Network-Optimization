function powerNoise()
    [P_vals, bruteP, greedyP, dnomaP, lcgP, jvP] = getPowerUtilityData();
    [N0_vals, bruteN, greedyN, dnomaN, lcgN, jvN] = getNoiseUtilityData();

    figure('Color', 'w', 'Position', [100 100 700 800]);

    subplot(2,1,1);
    plot(P_vals, bruteP, '-o', 'LineWidth', 2, 'DisplayName', 'Brute Force'); hold on;
    plot(P_vals, greedyP, '-s', 'LineWidth', 2, 'DisplayName', 'SG-NOMA');
    plot(P_vals, dnomaP,  '-^', 'LineWidth', 2, 'DisplayName', 'DNOMA');
    plot(P_vals, lcgP,    '-d', 'LineWidth', 2, 'DisplayName', 'LCG');
    plot(P_vals, jvP,     '-p', 'LineWidth', 2, 'DisplayName', 'JV'); hold off;
    xlabel('Total Transmission Power (P)');
    ylabel('Total Utility');
    title('Utility vs. Transmission Power P by Algorithm');
    legend('show', 'Location', 'northwest');
    grid on;

    subplot(2,1,2);
    semilogx(N0_vals, bruteN, '-o', 'LineWidth', 2, 'DisplayName', 'Brute Force'); hold on;
    semilogx(N0_vals, greedyN, '-s', 'LineWidth', 2, 'DisplayName', 'SG-NOMA');
    semilogx(N0_vals, dnomaN,  '-^', 'LineWidth', 2, 'DisplayName', 'DNOMA');
    semilogx(N0_vals, lcgN,    '-d', 'LineWidth', 2, 'DisplayName', 'LCG');
    semilogx(N0_vals, jvN,     '-p', 'LineWidth', 2, 'DisplayName', 'JV'); hold off;
    xlabel('Noise Power (N_0)', 'Interpreter','tex');
    ylabel('Total Utility');
    title('Utility vs. Noise Power N_0 by Algorithm');
    legend('show', 'Location', 'southwest');
    grid on;
end

function [P_vals, brute, greedy, dnoma, lcg, jv] = getPowerUtilityData()
    P_vals = [0.1, 0.5, 1, 2, 5];
    numPoints = 12;
    eta = 3; alpha = 0.6; N0 = 1e-4; sigma = 1;

    brute = zeros(size(P_vals));
    greedy = zeros(size(P_vals));
    dnoma = zeros(size(P_vals));
    lcg = zeros(size(P_vals));
    jv = zeros(size(P_vals));

    for k = 1:length(P_vals)
        P = P_vals(k);
        rng(42);
        [points, r] = generateScenario(numPoints, 10, 10, 10, sigma);

        pairings = makePairings(1:numPoints);
        maxU = -inf;
        for i = 1:length(pairings)
            util = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
            if util > maxU
                maxU = util;
            end
        end
        brute(k) = maxU;
        greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
        jv(k)     = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
    end
end

function [N0_vals, brute, greedy, dnoma, lcg, jv] = getNoiseUtilityData()
    N0_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
    numPoints = 12;
    eta = 3; alpha = 0.6; P = 1; sigma = 1;

    brute = zeros(size(N0_vals));
    greedy = zeros(size(N0_vals));
    dnoma = zeros(size(N0_vals));
    lcg = zeros(size(N0_vals));
    jv = zeros(size(N0_vals));

    for k = 1:length(N0_vals)
        N0 = N0_vals(k);
        rng(42);
        [points, r] = generateScenario(numPoints, 10, 10, 10, sigma);

        pairings = makePairings(1:numPoints);
        maxU = -inf;
        for i = 1:length(pairings)
            util = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
            if util > maxU
                maxU = util;
            end
        end
        brute(k) = maxU;
        greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
        jv(k)     = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
    end
end

function [points, r] = generateScenario(n, xr, yr, zr, sigma)
    x = rand(1, n) * xr;
    y = rand(1, n) * yr;
    z = rand(1, n) * zr;
    points = [x; y; z]';
    r = raylrnd(sigma, n, 1);
    dist = sqrt(sum(points.^2, 2));
    [~, idx] = sort(dist);
    points = points(idx, :);
    r = r(idx);
end

function U = calcUtility(pairs, points, r, eta, alpha, P, N0)
    U = 0;
    for j = 1:size(pairs, 1)
        i1 = pairs(j,1); i2 = pairs(j,2);
        d1 = norm(points(i1,:));
        d2 = norm(points(i2,:));
        h1 = (1 / (d1^(eta/2))) * r(i1);
        h2 = (1 / (d2^(eta/2))) * r(i2);
        R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
        R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
        U = U + R1 + R2;
    end
end

function pairings = makePairings(indices)
    if isempty(indices)
        pairings = {[]}; return;
    end
    pairings = {};
    first = indices(1);
    for i = 2:length(indices)
        second = indices(i);
        remaining = indices([2:i-1, i+1:end]);
        subPairings = makePairings(remaining);
        for j = 1:length(subPairings)
            pairings{end+1} = [[first, second]; subPairings{j}];
        end
    end
end

function pairs = greedyPairing(points, r, eta, alpha, P, N0)
    N = size(points, 1); remaining = 1:N; pairs = [];
    while numel(remaining) > 1
        bestU = -inf; bestPair = [];
        for i = 1:numel(remaining)
            for j = i+1:numel(remaining)
                pair = [remaining(i), remaining(j)];
                U = calcUtility(pair, points, r, eta, alpha, P, N0);
                if U > bestU
                    bestU = U;
                    bestPair = pair;
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = dnomaPairs(points)
    N = size(points, 1); idx = 1:N;
    G1 = idx(1:N/4); G2 = idx(N/4+1:N/2);
    G3 = idx(N/2+1:3*N/4); G4 = idx(3*N/4+1:end);
    pairs = [G1(:), G3(:); G2(:), G4(:)];
end

function pairs = lcgPairs(points)
    N = size(points,1); remaining = 1:N; pairs = [];
    while numel(remaining) > 1
        bestDist = inf; bestPair = [];
        for i = 1:numel(remaining)
            for j = i+1:numel(remaining)
                d = norm(points(remaining(i),:) - points(remaining(j),:));
                if d < bestDist
                    bestDist = d;
                    bestPair = [remaining(i), remaining(j)];
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = jvPairs(points, r, eta, alpha, P, N0)
    N = size(points, 1);
    close = 1:(N/2); far = (N/2+1):N;
    costMatrix = zeros(length(close), length(far));
    for i = 1:length(close)
        for j = 1:length(far)
            pair = [close(i), far(j)];
            costMatrix(i,j) = -calcUtility(pair, points, r, eta, alpha, P, N0);
        end
    end
    [match, ~] = matchpairs(costMatrix, -1);
    pairs = [close(match(:,1))', far(match(:,2))'];
end

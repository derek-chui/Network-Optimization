function powerNoiseEtaRayleigh()

    [P_vals,    bruteP, qlP, greedyP, dnomaP, lcgP] = getPowerUtilityData();
    [N0_vals,   bruteN, qlN, greedyN, dnomaN, lcgN] = getNoiseUtilityData();
    [eta_vals,  bruteE, qlE, greedyE, dnomaE, lcgE] = getEtaUtilityData();
    [sigma_vals,bruteS, qlS, greedyS, dnomaS, lcgS] = getRayleighUtilityData();

    figW = 7.5;
    figH = 8.5;
    figAll = figure('Color','w','Units','inches','Position',[0.5 0.5 figW figH], ...
                    'Renderer','painters');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    ax1 = nexttile;
    plotUtilityVsPower(ax1, P_vals, bruteP, qlP, greedyP, dnomaP, lcgP);
    legend(ax1,'show','Location','northwest');

    ax2 = nexttile;
    plotUtilityVsNoise(ax2, N0_vals, bruteN, qlN, greedyN, dnomaN, lcgN);
    legend(ax2,'show','Location','southwest');

    ax3 = nexttile;
    plotUtilityVsEta(ax3, eta_vals, bruteE, qlE, greedyE, dnomaE, lcgE);
    legend(ax3,'show','Location','northeast');

    ax4 = nexttile;
    plotUtilityVsSigma(ax4, sigma_vals, bruteS, qlS, greedyS, dnomaS, lcgS);
    legend(ax4,'show','Location','northwest');

    saveFigureVector(figAll, 'powerNoiseEtaRayleigh', figW, figH);

    saveTile(ax1, 'powerNoiseEtaRayleigh_P',     'northwest');
    saveTile(ax2, 'powerNoiseEtaRayleigh_N0',    'southwest');
    saveTile(ax3, 'powerNoiseEtaRayleigh_eta',   'northeast');
    saveTile(ax4, 'powerNoiseEtaRayleigh_sigma', 'northwest');

    figW2 = 7.5; figH2 = 4.0;
    figPN = figure('Color','w','Units','inches','Position',[1.0 1.0 figW2 figH2], ...
                   'Renderer','painters');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    axP = nexttile;
    plotUtilityVsPower(axP, P_vals, bruteP, qlP, greedyP, dnomaP, lcgP);
    legend(axP,'show','Location','northwest');

    axN = nexttile;
    plotUtilityVsNoise(axN, N0_vals, bruteN, qlN, greedyN, dnomaN, lcgN);
    legend(axN,'show','Location','southwest');

    saveFigureVector(figPN, 'powerNoiseEtaRayleigh_powerNoise', figW2, figH2);

    figER = figure('Color','w','Units','inches','Position',[1.2 1.2 figW2 figH2], ...
                   'Renderer','painters');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    axE = nexttile;
    plotUtilityVsEta(axE, eta_vals, bruteE, qlE, greedyE, dnomaE, lcgE);
    legend(axE,'show','Location','northeast');

    axS = nexttile;
    plotUtilityVsSigma(axS, sigma_vals, bruteS, qlS, greedyS, dnomaS, lcgS);
    legend(axS,'show','Location','northwest');

    saveFigureVector(figER, 'powerNoiseEtaRayleigh_etaRayleigh', figW2, figH2);
end


function [P_vals, brute, qlearn, greedy, dnoma, lcg] = getPowerUtilityData()
    P_vals = [0.1, 0.5, 1, 2, 5];

    numPoints = 12;
    eta   = 3;
    alpha = 0.6;
    N0    = 1e-4;
    sigma = 1;

    brute  = zeros(size(P_vals));
    qlearn = zeros(size(P_vals));
    greedy = zeros(size(P_vals));
    dnoma  = zeros(size(P_vals));
    lcg    = zeros(size(P_vals));

    qParams = defaultQLParams();

    for k = 1:length(P_vals)
        P = P_vals(k);

        rng(42);
        [points, r, weights] = generateScenario(numPoints, 10,10,10, sigma);

        pairings = makePairings(1:numPoints);

        maxU = -inf;
        for i = 1:length(pairings)
            u = calcUtility(pairings{i}, points, r, weights, eta, alpha, P, N0);
            if u > maxU, maxU = u; end
        end
        brute(k) = maxU;

        [qlPairs, ~] = qLearningPairing(points, r, weights, eta, alpha, P, N0, qParams);
        qlearn(k) = calcUtility(qlPairs, points, r, weights, eta, alpha, P, N0);

        greedy(k) = calcUtility(greedyPairing(points, r, weights, eta, alpha, P, N0), points, r, weights, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(numPoints), points, r, weights, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, weights, eta, alpha, P, N0);
    end
end

function [N0_vals, brute, qlearn, greedy, dnoma, lcg] = getNoiseUtilityData()
    N0_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];

    numPoints = 12;
    eta   = 3;
    alpha = 0.6;
    P     = 1;
    sigma = 1;

    brute  = zeros(size(N0_vals));
    qlearn = zeros(size(N0_vals));
    greedy = zeros(size(N0_vals));
    dnoma  = zeros(size(N0_vals));
    lcg    = zeros(size(N0_vals));

    qParams = defaultQLParams();

    for k = 1:length(N0_vals)
        N0 = N0_vals(k);

        rng(42);
        [points, r, weights] = generateScenario(numPoints, 10,10,10, sigma);

        pairings = makePairings(1:numPoints);

        maxU = -inf;
        for i = 1:length(pairings)
            u = calcUtility(pairings{i}, points, r, weights, eta, alpha, P, N0);
            if u > maxU, maxU = u; end
        end
        brute(k) = maxU;

        [qlPairs, ~] = qLearningPairing(points, r, weights, eta, alpha, P, N0, qParams);
        qlearn(k) = calcUtility(qlPairs, points, r, weights, eta, alpha, P, N0);

        greedy(k) = calcUtility(greedyPairing(points, r, weights, eta, alpha, P, N0), points, r, weights, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(numPoints), points, r, weights, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, weights, eta, alpha, P, N0);
    end
end

function [eta_vals, brute, qlearn, greedy, dnoma, lcg] = getEtaUtilityData()
    eta_vals = 2:6;

    numPoints = 12;
    alpha = 0.6;
    P     = 1;
    N0    = 1e-4;
    sigma = 1;

    brute  = zeros(size(eta_vals));
    qlearn = zeros(size(eta_vals));
    greedy = zeros(size(eta_vals));
    dnoma  = zeros(size(eta_vals));
    lcg    = zeros(size(eta_vals));

    qParams = defaultQLParams();

    for k = 1:length(eta_vals)
        eta = eta_vals(k);

        rng(42);
        [points, r, weights] = generateScenario(numPoints, 10,10,10, sigma);

        pairings = makePairings(1:numPoints);

        maxU = -inf;
        for i = 1:length(pairings)
            u = calcUtility(pairings{i}, points, r, weights, eta, alpha, P, N0);
            if u > maxU, maxU = u; end
        end
        brute(k) = maxU;

        [qlPairs, ~] = qLearningPairing(points, r, weights, eta, alpha, P, N0, qParams);
        qlearn(k) = calcUtility(qlPairs, points, r, weights, eta, alpha, P, N0);

        greedy(k) = calcUtility(greedyPairing(points, r, weights, eta, alpha, P, N0), points, r, weights, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(numPoints), points, r, weights, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, weights, eta, alpha, P, N0);
    end
end

function [sigma_vals, brute, qlearn, greedy, dnoma, lcg] = getRayleighUtilityData()
    sigma_vals = [0.5, 1.0, 1.5, 2.0, 2.5];

    numPoints = 12;
    eta   = 3;
    alpha = 0.6;
    P     = 1;
    N0    = 1e-4;

    brute  = zeros(size(sigma_vals));
    qlearn = zeros(size(sigma_vals));
    greedy = zeros(size(sigma_vals));
    dnoma  = zeros(size(sigma_vals));
    lcg    = zeros(size(sigma_vals));

    qParams = defaultQLParams();

    for k = 1:length(sigma_vals)
        sigma = sigma_vals(k);

        rng(42);
        [points, r, weights] = generateScenario(numPoints, 10,10,10, sigma);

        pairings = makePairings(1:numPoints);

        maxU = -inf;
        for i = 1:length(pairings)
            u = calcUtility(pairings{i}, points, r, weights, eta, alpha, P, N0);
            if u > maxU, maxU = u; end
        end
        brute(k) = maxU;

        [qlPairs, ~] = qLearningPairing(points, r, weights, eta, alpha, P, N0, qParams);
        qlearn(k) = calcUtility(qlPairs, points, r, weights, eta, alpha, P, N0);

        greedy(k) = calcUtility(greedyPairing(points, r, weights, eta, alpha, P, N0), points, r, weights, eta, alpha, P, N0);
        dnoma(k)  = calcUtility(dnomaPairs(numPoints), points, r, weights, eta, alpha, P, N0);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, weights, eta, alpha, P, N0);
    end
end

% =====================================================================
% FIXED: save each tile WITH its legend
% =====================================================================
function saveTile(ax, base, legendLoc)
    f = figure('Visible','off','Units','inches','Position',[0.5 0.5 6 4], 'Renderer','painters');

    axCopy = copyobj(ax, f);
    set(axCopy,'Units','normalized','Position',[0.13 0.14 0.84 0.79]);

    % IMPORTANT: legends are NOT copied with axes, so recreate them:
    if nargin < 3 || isempty(legendLoc)
        legend(axCopy,'show');
    else
        legend(axCopy,'show','Location',legendLoc);
    end

    set(f,'PaperUnits','inches','PaperSize',[6 4], 'PaperPosition',[0 0 6 4], 'PaperPositionMode','manual');
    set(f,'InvertHardcopy','off');

    print(f, '-dpdf',  '-painters', [base '.pdf']);
    print(f, '-depsc', '-painters', '-loose', [base '.eps']);
    try
        print(f, '-dsvg', '-painters', [base '.svg']);
    catch
    end
    close(f);
end

function plotUtilityVsPower(ax, P_vals, bruteP, qlP, greedyP, dnomaP, lcgP)
    axes(ax);
    plot(P_vals, bruteP,  '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(P_vals, qlP,     '-x', 'LineWidth', 1.8, 'DisplayName', 'Q-Learning');
    plot(P_vals, greedyP, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(P_vals, dnomaP,  '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    plot(P_vals, lcgP,    '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG'); hold off;
    grid on; box on;
    xlabel('Total Transmission Power (P)'); ylabel('Total Semantic Utility');
    title('Utility vs. Transmission Power P');
end

function plotUtilityVsNoise(ax, N0_vals, bruteN, qlN, greedyN, dnomaN, lcgN)
    axes(ax);
    semilogx(N0_vals, bruteN,  '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    semilogx(N0_vals, qlN,     '-x', 'LineWidth', 1.8, 'DisplayName', 'Q-Learning');
    semilogx(N0_vals, greedyN, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    semilogx(N0_vals, dnomaN,  '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    semilogx(N0_vals, lcgN,    '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG'); hold off;
    grid on; box on;
    xlabel('Noise Power (N_0)','Interpreter','tex'); ylabel('Total Semantic Utility');
    title('Utility vs. Noise Power N_0');
end

function plotUtilityVsEta(ax, eta_vals, bruteE, qlE, greedyE, dnomaE, lcgE)
    axes(ax);
    plot(eta_vals, bruteE,  '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(eta_vals, qlE,     '-x', 'LineWidth', 1.8, 'DisplayName', 'Q-Learning');
    plot(eta_vals, greedyE, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(eta_vals, dnomaE,  '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    plot(eta_vals, lcgE,    '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG'); hold off;
    grid on; box on;
    xlabel('Path Loss Exponent (\eta)','Interpreter','tex'); ylabel('Total Semantic Utility');
    title('Utility vs. Path Loss Exponent \eta');
end

function plotUtilityVsSigma(ax, sigma_vals, bruteS, qlS, greedyS, dnomaS, lcgS)
    axes(ax);
    plot(sigma_vals, bruteS,  '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(sigma_vals, qlS,     '-x', 'LineWidth', 1.8, 'DisplayName', 'Q-Learning');
    plot(sigma_vals, greedyS, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(sigma_vals, dnomaS,  '-^', 'LineWidth', 1.8, 'DeisplayName', 'DNOMA');
    plot(sigma_vals, lcgS,    '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG'); hold off;
    grid on; box on;
    xlabel('Rayleigh Scale Parameter (\sigma)','Interpreter','tex'); ylabel('Total Semantic Utility');
    title('Utility vs. Rayleigh Fading Parameter \sigma');
end

% =====================================================================
% Vector save helper for whole figures
% =====================================================================
function saveFigureVector(fig, base, figW, figH)
    set(fig, 'PaperUnits','inches');
    set(fig, 'PaperSize', [figW figH]);
    set(fig, 'PaperPosition', [0 0 figW figH]);
    set(fig, 'PaperPositionMode','manual');
    set(fig, 'InvertHardcopy','off');
    orient(fig,'portrait');

    print(fig, '-dpdf',  '-painters', [base '.pdf']);
    print(fig, '-depsc', '-painters', '-loose', [base '.eps']);
    try
        print(fig, '-dsvg', '-painters', [base '.svg']);
    catch
    end
end

function [points, r, weights] = generateScenario(n, xr, yr, zr, sigma)
    x = -xr/2 + xr * rand(1,n);
    y = -yr/2 + yr * rand(1,n);
    z = -zr/2 + zr * rand(1,n);
    points  = [x; y; z]';

    r = raylrnd_local(sigma, [n, 1]);

    weights = rand(n,1);

    dist = sqrt(sum(points.^2, 2));
    [~, idx] = sort(dist, 'ascend');
    points  = points(idx, :);
    r       = r(idx);
    weights = weights(idx);
end

function U = calcUtility(pairs, points, r, weights, eta, alpha, P, N0)

    if isempty(pairs)
        U = 0;
        return;
    end

    if size(pairs,2) ~= 2
        error('calcUtility: pairs must be an Nx2 matrix of user indices.');
    end

    a1 = 0.30; a2 = 0.98; c1 = 0.25; c2 = -0.8;

    U = 0;
    for jj = 1:size(pairs,1)
        i1 = pairs(jj,1); i2 = pairs(jj,2);

        d1 = norm(points(i1,:)); d2 = norm(points(i2,:));
        h1 = (1 / (d1^(eta/2))) * r(i1);
        h2 = (1 / (d2^(eta/2))) * r(i2);

        h = [h1, h2];
        k = [weights(i1), weights(i2)];
        [h_sorted, ord] = sort(h, 'ascend');
        k_sorted = k(ord);

        h_w = h_sorted(1); h_s = h_sorted(2);
        k_w = k_sorted(1); k_s = k_sorted(2);

        gamma_w = (alpha * P * h_w^2) / ((1 - alpha) * P * h_w^2 + N0);
        gamma_s = ((1 - alpha) * P * h_s^2) / N0;

        xi_w = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_w + c2)));
        xi_s = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_s + c2)));

        U = U + k_w * xi_w * log2(1 + gamma_w) + k_s * xi_s * log2(1 + gamma_s);
    end
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

function pairs = greedyPairing(points, r, weights, eta, alpha, P, N0)
    N = size(points,1);
    remaining = 1:N;
    pairs = [];

    while numel(remaining) > 1
        bestU = -inf;
        bestPair = [];
        for ii = 1:numel(remaining)
            for jj = ii+1:numel(remaining)
                pr = [remaining(ii), remaining(jj)];
                u = calcUtility(pr, points, r, weights, eta, alpha, P, N0);
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

function pairs = dnomaPairs(N)
    idx = 1:N;
    G1 = idx(1:N/4);         G2 = idx(N/4+1:N/2);
    G3 = idx(N/2+1:3*N/4);   G4 = idx(3*N/4+1:end);
    pairs = [G1(:), G3(:); G2(:), G4(:)];
end

function pairs = lcgPairs(points)
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

function q = defaultQLParams()
    q = struct();
    q.episodes       = 3000;
    q.lr             = 0.30;
    q.gamma          = 0.95;
    q.epsilon_start  = 0.90;
    q.epsilon_end    = 0.05;
    q.epsilon_decay  = 0.995;
    q.log_every      = 0;
    q.verbose        = false;
    q.seed           = 7;
end

function [pairs, episodeBestScore] = qLearningPairing(points, r, weights, eta, alpha, P, N0, q)

    N = size(points,1);

    if isfield(q,'seed') && ~isempty(q.seed)
        rng(q.seed);
    end

    Q = containers.Map('KeyType','char','ValueType','double');

    epsilon = q.epsilon_start;
    episodeBestScore = -inf;
    episodeBestPairs = [];

    for ep = 1:q.episodes
        remaining = true(1,N);
        current   = [];
        pairsEp   = [];
        cumScore  = 0;

        while any(remaining)
            sKey    = makeStateKey_pair(remaining, current);
            actions = find(remaining);

            if rand < epsilon
                a = actions(randi(numel(actions)));
            else
                a = argmaxQ_pair(Q, sKey, actions);
            end

            remaining(a) = false;
            current = [current, a];

            reward = 0;
            if numel(current) == 2
                reward = calcUtility(current, points, r, weights, eta, alpha, P, N0);
                pairsEp = [pairsEp; current];
                cumScore = cumScore + reward;
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
            Q(qaKey) = (1 - q.lr)*oldQ + q.lr*(reward + q.gamma*maxQnext);
        end

        if cumScore > episodeBestScore
            episodeBestScore = cumScore;
            episodeBestPairs = pairsEp;
        end

        epsilon = max(q.epsilon_end, epsilon * q.epsilon_decay);

        if q.verbose && q.log_every > 0 && mod(ep, q.log_every) == 0
            fprintf('[Q] ep %d/%d | last=%.4f | best=%.4f | eps=%.3f\n', ...
                ep, q.episodes, cumScore, episodeBestScore, epsilon);
        end
    end

    remaining = true(1,N);
    current   = [];
    pairs     = [];
    while any(remaining)
        sKey    = makeStateKey_pair(remaining, current);
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
    bits = char('0' + remaining);
    cur  = sort(current);
    if isempty(cur)
        curStr = '[]';
    else
        curStr = sprintf('%02d', cur(1));
    end
    key = [bits '|' curStr];
end

function key = makeQAKey_pair(stateKey, a)
    key = [stateKey '#A' sprintf('%02d', a)];
end

function val = getQ_pair(Q, stateKey, a)
    qaKey = makeQAKey_pair(stateKey, a);
    if isKey(Q, qaKey)
        val = Q(qaKey);
    else
        val = 0;
    end
end

function m = maxQ_pair(Q, stateKey, actions)
    vals = zeros(1, numel(actions));
    for i = 1:numel(actions)
        vals(i) = getQ_pair(Q, stateKey, actions(i));
    end
    m = max(vals);
end

function aStar = argmaxQ_pair(Q, stateKey, actions)
    bestVal = -inf;
    aStar = actions(1);
    for i = 1:numel(actions)
        a = actions(i);
        v = getQ_pair(Q, stateKey, a);
        if v > bestVal + 1e-12
            bestVal = v;
            aStar = a;
        elseif abs(v - bestVal) <= 1e-12
            aStar = min(aStar, a);
        end
    end
end

function r = raylrnd_local(b, sz)
    u = max(eps, rand(sz));
    r = b .* sqrt(-2 .* log(u));
end

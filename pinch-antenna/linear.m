%Linear, Joint Power, Pinch Antenna, Semantic, Q-Learning, NOMA

clear; clc; close all;

K = 4;
M = 4;
rng('shuffle');

P = K/2;

eta = 3;
Ptx = 10;
N0 = 1e-5;

tau_sic = 0.00; %optional overhead penalty per extra antenna assigned to the same pair

%semantic importance and fidelity
w_sem = rand(1, K);
semParams = struct( ...
    'a1', 0.30, ...
    'a2', 0.98, ...
    'c1', 0.25, ...
    'c2', -0.8 ...
);

%joint power alloc, scan alpha grid inside utility
USE_JOINT_ALPHA = true;

ALPHA_GRID = 0.50:0.05:0.95; %weak gets alpha_w strong gets 1-alpha_w

alpha_fixed_weak = 0.60; %used only if use joint alpha false
alpha_fixed_strong = 0.40;

%q learning settings
USE_Q_PAIRING = false; %rl chooses user grouping if true
USE_Q_ASSIGNMENT = true; %rl chooses antenna to grouping labels to maximize semantic utility if true

qParams = struct( ...
    'episodes',      50, ...
    'alpha',         0.35, ...
    'gamma',         0.92, ...
    'epsilon_start', 0.90, ...
    'epsilon_end',   0.05, ...
    'epsilon_decay', 0.995, ...
    'log_every',     5, ...
    'verbose',       true ...
);

%pinch antenna stuff
c = 3e8;
fc = 28e9;
lambda = c/fc;

L = 100;

d_wg = 3;

%geometry
BS = [0, -L];

y_margin = 0.30*L;
y_pos = linspace(BS(2)+y_margin, L-y_margin, M).';
A_xy = [zeros(M,1), y_pos];
A = [A_xy, d_wg*ones(M,1)];

U_xy = (rand(K,2)*2 - 1)*L;
U    = [U_xy, zeros(K,1)];

theta = 2*pi*rand(1,M);

D = zeros(K,M);
for m = 1:M
    diff = U - repmat(A(m,:),K,1);
    D(:,m) = sqrt(sum(diff.^2,2));
end
D = max(D, 1e-3);

Hc = zeros(K,M);
for m = 1:M
    Hc(:,m) = (sqrt(eta) ./ D(:,m)) .* exp(-1j*(2*pi/lambda)*D(:,m)) .* exp(-1j*theta(m));
end

%baseline effective channel if all antennas r active
h_all = sum(Hc,2);

%user grouping
if ~USE_Q_PAIRING
    [~, ord] = sort(abs(h_all).^2,'ascend');
    idx_sorted = ord(:);

    weak_idx   = idx_sorted(1:P);
    strong_idx = flipud(idx_sorted(P+1:end));
    pair_users = [weak_idx, strong_idx]; %Px2
else
    pair_users = qLearningPairingPairs( ...
        h_all, w_sem, Ptx, M, N0, USE_JOINT_ALPHA, ALPHA_GRID, ...
        alpha_fixed_weak, alpha_fixed_strong, semParams, qParams);
end

%semantic utility objective (antenna to group assignment)
if USE_Q_ASSIGNMENT
    [bestAssign, bestS, bestPer] = qLearningAntennaAssignment( ...
        pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, USE_JOINT_ALPHA, ALPHA_GRID, ...
        alpha_fixed_weak, alpha_fixed_strong, semParams, qParams);
else
    [bestAssign, bestS, bestPer] = searchBestAssignmentSemantic( ...
        pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, USE_JOINT_ALPHA, ALPHA_GRID, ...
        alpha_fixed_weak, alpha_fixed_strong, semParams);
end

pair_util = arrayfun(@(s) s.Gsum, bestPer);

%command window
fprintf('K = %d, M = %d, Pairs = %d\n', K, M, P);
fprintf('Semantic importance weights (w_sem):\n');
for i=1:K
    fprintf('  U%d: %.3f\n', i, w_sem(i));
end

fprintf('\nPairs:\n');
for p=1:P
    fprintf('  Pair %d: (U%d, U%d)\n', p, pair_users(p,1), pair_users(p,2));
end

fprintf('\nBest Assignment Per Antenna (label 0=unused, 1..P=pair id):\n  [ ');
fprintf('%d ', bestAssign);
fprintf(']\n');

fprintf('\nPer-pair semantic utility breakdown (includes best alpha per pair if enabled):\n');
for p=1:P
    ants = bestPer(p).ant;
    uw = bestPer(p).weakUser;
    us = bestPer(p).strongUser;

    if isempty(ants)
        antStr = '{}';
    else
        antStr = ['{ ', sprintf('A%d ', ants), '}'];
    end

    fprintf(['  Pair %d: Antennas %s | (Weak,Strong)=(U%d,U%d) | ' ...
             'alpha_w=%.2f alpha_s=%.2f | ' ...
             'Rw=%.3f Rs=%.3f | xi_w=%.3f xi_s=%.3f | ' ...
             'Gw=%.4f Gs=%.4f | Gsum=%.4f\n'], ...
        p, antStr, uw, us, ...
        bestPer(p).alpha_w, bestPer(p).alpha_s, ...
        bestPer(p).Rw, bestPer(p).Rs, ...
        bestPer(p).xi_w, bestPer(p).xi_s, ...
        bestPer(p).Gw, bestPer(p).Gs, bestPer(p).Gsum);
end
fprintf('\nTOTAL semantic utility: %.4f | Mean per pair: %.4f\n', sum(pair_util), mean(pair_util));

%plot
figure('Color','w'); hold on; axis equal;
% keep BS at the bottom and users anywhere in LxL square
xlim([-L L]);
ylim([BS(2)-0.10*L, L]);
box off;

plot(BS(1),BS(2),'ks','MarkerFaceColor','k','MarkerSize',9);
text(BS(1)+2,BS(2),'BS');

%antennas
for m=1:M
    on = bestAssign(m)~=0;
    face = on*[0 0 0] + ~on*[1 1 1];
    plot(A(m,1),A(m,2),'^','MarkerSize',9,'MarkerFaceColor',face,'Color','k');
    if on
        text(A(m,1)+2,A(m,2),sprintf('A%d\\rightarrowP%d',m,bestAssign(m)));
    else
        text(A(m,1)+2,A(m,2),sprintf('A%d',m));
    end
end

%users
plot(U(:,1),U(:,2),'o','MarkerFaceColor',[0.2 0.6 1],'Color',[0 0.2 0.6]);
for i=1:K
    text(U(i,1)+2, U(i,2), sprintf('U%d (%.2f)', i, w_sem(i)));
end

%dashed pair links
for p=1:P
    u1 = pair_users(p,1); u2 = pair_users(p,2);
    plot([U(u1,1) U(u2,1)], [U(u1,2) U(u2,2)], '--', 'Color', 0.5*[1 1 1], 'LineWidth', 1.0);
end

%antenna to user links
pairColors = lines(P);
for m = 1:M
    pID = bestAssign(m);
    if pID == 0, continue; end
    uu = pair_users(pID,:);
    colorPair = pairColors(pID,:);
    for t = 1:2
        i = uu(t);
        plot([A(m,1) U(i,1)], [A(m,2) U(i,2)], '-', 'Color', colorPair, 'LineWidth', 1.8);
    end
end

title(sprintf('Pinch Antennas (Linear Waveguide)'));
xlabel('x (m)'); ylabel('y (m)'); grid on;

%helpers
function [bestAssign, bestS, bestPer] = searchBestAssignmentSemantic( ...
    pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, useJointAlpha, alphaGrid, ...
    alpha_fixed_weak, alpha_fixed_strong, semParams)

    P = size(pair_users,1);

    if M <= 7 && P <= 3
        labels = 0:P;
        allAssign = cell(1,M);
        [allAssign{:}] = ndgrid(labels);
        mats = cellfun(@(x) x(:), allAssign, 'uni',0);
        ASSIGN = [mats{:}];

        bestS = -inf; bestAssign = []; bestPer = [];
        for r = 1:size(ASSIGN,1)
            a = ASSIGN(r,:);
            [Ssum, perPair] = eval_assignment_semantic( ...
                a, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
                useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);
            if Ssum > bestS
                bestS = Ssum; bestAssign = a; bestPer = perPair;
            end
        end
    else
        a = zeros(1,M);
        [bestS, bestPer] = eval_assignment_semantic( ...
            a, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
            useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);
        bestAssign = a;

        improved = true;
        while improved
            improved = false;
            for m = 1:M
                for lab = 0:P
                    if bestAssign(m)==lab, continue; end
                    a_try = bestAssign; a_try(m)=lab;
                    [S_try, per_try] = eval_assignment_semantic( ...
                        a_try, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
                        useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);
                    if S_try > bestS
                        bestS = S_try; bestAssign = a_try; bestPer = per_try; improved = true;
                    end
                end
            end
        end
    end
end

function [Ssum, perPair] = eval_assignment_semantic( ...
    assign, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
    useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams)

    P = size(pair_users,1);

    perPair = repmat(struct( ...
        'ant',[], ...
        'weakUser',[], ...
        'strongUser',[], ...
        'alpha_w',0, ...
        'alpha_s',0, ...
        'hw',0, ...
        'hs',0, ...
        'Rw',0, ...
        'Rs',0, ...
        'gamma_w',0, ...
        'gamma_s',0, ...
        'xi_w',0, ...
        'xi_s',0, ...
        'Gw',0, ...
        'Gs',0, ...
        'Gsum',0), P, 1);

    Ssum = 0;

    for p = 1:P
        ants = find(assign == p); %antennas assigned to this pair
        uA = pair_users(p,1);
        uB = pair_users(p,2);

        %effective channels = sum of contributions from assigned antennas
        if isempty(ants)
            hA = 0;
            hB = 0;
        else
            hA = sum(Hc(uA,ants));
            hB = sum(Hc(uB,ants));
        end

        %determine weak or strong via |h|^2
        if abs(hA)^2 <= abs(hB)^2
            uw = uA; us = uB; hw = hA; hs = hB;
        else
            uw = uB; us = uA; hw = hB; hs = hA;
        end

        %optional SIC overhead penalty as mentioned earlier
        over = max(0, 1 - tau_sic*max(0, numel(ants)-1));

        kw = w_sem(uw);
        ks = w_sem(us);

        %choose best alpha inside the utility calculation
        if useJointAlpha
            [alpha_w, alpha_s, Rw, Rs, gamma_w, gamma_s, xi_w, xi_s, Gw, Gs] = noma2_semantic_bestAlpha( ...
                hw, hs, kw, ks, Ptx, M, N0, alphaGrid, over, semParams);
        else
            alpha_w = alpha_fixed_weak;
            alpha_s = alpha_fixed_strong;
            [Rw, Rs, gamma_w, gamma_s, xi_w, xi_s, Gw, Gs] = noma2_semantic_givenAlpha( ...
                hw, hs, kw, ks, Ptx, M, N0, alpha_w, alpha_s, over, semParams);
        end

        perPair(p).ant = ants;
        perPair(p).weakUser = uw;
        perPair(p).strongUser = us;
        perPair(p).alpha_w = alpha_w;
        perPair(p).alpha_s = alpha_s;
        perPair(p).hw = hw;
        perPair(p).hs = hs;
        perPair(p).Rw = Rw;
        perPair(p).Rs = Rs;
        perPair(p).gamma_w = gamma_w;
        perPair(p).gamma_s = gamma_s;
        perPair(p).xi_w = xi_w;
        perPair(p).xi_s = xi_s;
        perPair(p).Gw = Gw;
        perPair(p).Gs = Gs;
        perPair(p).Gsum = Gw + Gs;

        Ssum = Ssum + (Gw + Gs);
    end
end

%noma rate and semantics given alpha
function [Rw, Rs, gamma_w, gamma_s, xi_w, xi_s, Gw, Gs] = noma2_semantic_givenAlpha( ...
    hw, hs, kw, ks, Ptx, M, N0, alpha_w, alpha_s, over, semParams)

    Ppa = Ptx / M; %power per pinch

    gw = abs(hw)^2;
    gs = abs(hs)^2;

    %weak message decodable at weak and strong users (min constraint)
    gamma_ww = (gw*Ppa*alpha_w) / (gw*Ppa*alpha_s + N0);
    gamma_sw = (gs*Ppa*alpha_w) / (gs*Ppa*alpha_s + N0);
    gamma_w  = min(gamma_ww, gamma_sw);

    %strong after SIC
    gamma_s = (gs*Ppa*alpha_s) / N0;

    %rates
    Rw = log2(1 + gamma_w);
    Rs = log2(1 + gamma_s);

    %semantic fidelity
    a1 = semParams.a1; a2 = semParams.a2; c1 = semParams.c1; c2 = semParams.c2;
    xi_w = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_w + c2)));
    xi_s = a1 + (a2-a1) ./ (1 + exp(-(c1*gamma_s + c2)));

    %semantic contributions
    Gw = kw * xi_w * Rw;
    Gs = ks * xi_s * Rs;

    %overhead penalty on strong contribution if used
    Rs = Rs * over;
    Gs = Gs * over;
end

%pick best alpha via grid, so the objective is Gw + Gs (semantic utility)
function [best_aw, best_as, bestRw, bestRs, best_gw, best_gs, best_xiw, best_xis, bestGw, bestGs] = ...
    noma2_semantic_bestAlpha(hw, hs, kw, ks, Ptx, M, N0, alphaGrid, over, semParams)

    bestVal = -inf;

    best_aw = alphaGrid(1);
    best_as = 1 - best_aw;

    bestRw = 0; bestRs = 0;
    best_gw = 0; best_gs = 0;
    best_xiw = 0; best_xis = 0;
    bestGw = 0; bestGs = 0;

    for aw = alphaGrid
        as = 1 - aw;

        if aw <= 0 || as <= 0
            continue;
        end

        [Rw, Rs, gamma_w, gamma_s, xi_w, xi_s, Gw, Gs] = noma2_semantic_givenAlpha( ...
            hw, hs, kw, ks, Ptx, M, N0, aw, as, over, semParams);

        val = Gw + Gs;

        if val > bestVal
            bestVal = val;

            best_aw = aw;
            best_as = as;

            bestRw = Rw; bestRs = Rs;
            best_gw = gamma_w; best_gs = gamma_s;
            best_xiw = xi_w; best_xis = xi_s;
            bestGw = Gw; bestGs = Gs;
        end
    end
end

%q learning for assigning antennas
function [bestAssign, bestScore, bestPer] = qLearningAntennaAssignment( ...
    pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
    useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams, p)

    if ~isfield(p,'log_every'), p.log_every = max(1, floor(p.episodes/20)); end
    if ~isfield(p,'verbose'), p.verbose = true; end

    P = size(pair_users,1);

    %state t (next antenna index) + partial assignment labels for antennas 1..t-1
    %action pick label in {0,1,...,P} for antenna t
    %reward 0 until terminal (t=M), total semantic utility (with best alpha per grouping if enabled)

    Q = containers.Map('KeyType','char','ValueType','double');

    bestScore  = -inf;
    bestAssign = zeros(1,M);

    eps = p.epsilon_start;

    for ep = 1:p.episodes
        a = zeros(1,M);
        episodeReward = 0;

        for t = 1:M
            sKey = makeAssignStateKey(t, a);

            actions = 0:P;

            %epsilon
            if rand < eps
                lab = actions(randi(numel(actions)));
            else
                lab = argmaxQ_small(Q, sKey, actions);
            end

            a(t) = lab;

            reward = 0;
            if t == M
                [reward, ~] = eval_assignment_semantic( ...
                    a, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
                    useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);
                episodeReward = reward;
            end

            %next state
            if t == M
                maxQnext = 0;
            else
                sNextKey = makeAssignStateKey(t+1, a);
                maxQnext = maxQ_small(Q, sNextKey, actions);
            end

            qaKey = makeQAKey_small(sKey, lab);
            oldQ  = 0;
            if isKey(Q, qaKey), oldQ = Q(qaKey); end
            Q(qaKey) = (1 - p.alpha)*oldQ + p.alpha*(reward + p.gamma*maxQnext);
        end

        if episodeReward > bestScore
            bestScore  = episodeReward;
            bestAssign = a;
        end

        eps = max(p.epsilon_end, eps * p.epsilon_decay);

        if p.verbose && (ep == 1 || ep == p.episodes || mod(ep, p.log_every) == 0)
            fprintf('Q Learning Episode %4d  /%4d | utility = %.4f | best = %.4f\n', ...
                ep, p.episodes, episodeReward, bestScore);
            drawnow limitrate
        end
    end

    %learned Q
    a = zeros(1,M);
    for t = 1:M
        sKey = makeAssignStateKey(t, a);
        actions = 0:P;
        lab = argmaxQ_small(Q, sKey, actions);
        a(t) = lab;
    end

    %pick better vs best episode
    [greedyScore, greedyPer] = eval_assignment_semantic( ...
        a, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
        useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);

    if greedyScore >= bestScore
        bestAssign = a;
        bestScore  = greedyScore;
        bestPer    = greedyPer;
    else
        [~, bestPer] = eval_assignment_semantic( ...
            bestAssign, pair_users, Hc, w_sem, Ptx, M, N0, tau_sic, ...
            useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);
    end
end

function key = makeAssignStateKey(t, a)
    if t <= 1
        prefix = '[]';
    else
        list = sprintf('%d,', a(1:t-1));
        list(end) = [];
        prefix = ['[', list, ']'];
    end
    key = sprintf('t:%d|pref:%s', t, prefix);
end

function k = makeQAKey_small(sKey, act)
    k = [sKey,'|a:',num2str(act)];
end

function a = argmaxQ_small(Q, sKey, actions)
    bestVal = -inf;
    a = actions(1);
    for i = 1:numel(actions)
        k = makeQAKey_small(sKey, actions(i));
        v = 0; if isKey(Q, k), v = Q(k); end
        if v > bestVal
            bestVal = v;
            a = actions(i);
        end
    end
end

function m = maxQ_small(Q, sKey, actions)
    if isempty(actions), m = 0; return; end
    vals = zeros(1,numel(actions));
    for i = 1:numel(actions)
        k = makeQAKey_small(sKey, actions(i));
        if isKey(Q, k), vals(i) = Q(k); end
    end
    m = max(vals);
end

%optional user grouping with joint alpha in reward
function pair_users = qLearningPairingPairs( ...
    h_all, w_sem, Ptx, M, N0, useJointAlpha, alphaGrid, ...
    alpha_fixed_weak, alpha_fixed_strong, semParams, p)

    if ~isfield(p,'log_every'), p.log_every = max(1, floor(p.episodes/20)); end
    if ~isfield(p,'verbose'), p.verbose = true; end

    K = numel(h_all);
    P = K/2;

    %state remaining users + current partial pair
    %action pick a remaining user
    %reward when pair completes, pair semantic utility using all antennas active channels
    %best alpha inside reward if enabled

    Q = containers.Map('KeyType','char','ValueType','double');

    bestScore = -inf;
    bestPairs = [];

    eps = p.epsilon_start;

    for ep = 1:p.episodes
        remaining = true(1,K);
        current = [];
        cumScore = 0;
        tracePairs = [];

        while any(remaining)
            sKey = makeStateKey_users(remaining, current);
            actions = find(remaining);

            if rand < eps
                a = actions(randi(numel(actions)));
            else
                a = argmaxQ_users(Q, sKey, actions);
            end

            remaining(a) = false;
            current = [current, a];

            reward = 0;
            if numel(current) == 2
                i = current(1); j = current(2);

                reward = pairUtility_semantic_allAnt( ...
                    h_all(i), h_all(j), w_sem(i), w_sem(j), Ptx, M, N0, ...
                    useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams);

                cumScore = cumScore + reward;
                tracePairs = [tracePairs; current];
                current = [];
            end

            sNextKey = makeStateKey_users(remaining, current);
            nextActions = find(remaining);
            if isempty(nextActions)
                maxQnext = 0;
            else
                maxQnext = maxQ_users(Q, sNextKey, nextActions);
            end

            qaKey = makeQAKey_users(sKey, a);
            oldQ  = 0;
            if isKey(Q, qaKey), oldQ = Q(qaKey); end
            Q(qaKey) = (1 - p.alpha)*oldQ + p.alpha*(reward + p.gamma*maxQnext);
        end

        if cumScore > bestScore
            bestScore = cumScore;
            bestPairs = tracePairs;
        end

        eps = max(p.epsilon_end, eps * p.epsilon_decay);

        if p.verbose && (ep == 1 || ep == p.episodes || mod(ep, p.log_every) == 0)
            fprintf('Q Grouping Episode %4d  /%4d | utility = %.4f | best = %.4f\n', ...
                ep, p.episodes, cumScore, bestScore);
            drawnow limitrate
        end
    end

    remaining = true(1,K);
    current = [];
    pairs = [];
    while any(remaining)
        sKey = makeStateKey_users(remaining, current);
        actions = find(remaining);
        a = argmaxQ_users(Q, sKey, actions);
        remaining(a) = false;
        current = [current, a];
        if numel(current) == 2
            pairs = [pairs; current];
            current = [];
        end
    end

    if isempty(pairs) || size(pairs,1) ~= P
        pairs = bestPairs;
    end

    pair_users = pairs;
end

function key = makeStateKey_users(remaining, current)
    r = char('0' + remaining); %1 for remaining 0 for used
    if isempty(current)
        c = '[]';
    else
        tmp = sort(current);
        list = sprintf('%d,', tmp);
        list(end) = [];
        c = ['[', list, ']'];
    end
    key = [r,'|',c];
end

function k = makeQAKey_users(sKey, a)
    k = [sKey,'|a:',num2str(a)];
end

function a = argmaxQ_users(Q, sKey, actions)
    bestVal = -inf; a = actions(1);
    for i = 1:numel(actions)
        k = makeQAKey_users(sKey, actions(i));
        v = 0; if isKey(Q,k), v = Q(k); end
        if v > bestVal
            bestVal = v; a = actions(i);
        end
    end
end

function m = maxQ_users(Q, sKey, actions)
    if isempty(actions), m = 0; return; end
    vals = zeros(1,numel(actions));
    for i=1:numel(actions)
        k = makeQAKey_users(sKey, actions(i));
        if isKey(Q,k), vals(i) = Q(k); end
    end
    m = max(vals);
end

function rew = pairUtility_semantic_allAnt( ...
    h1, h2, w1, w2, Ptx, M, N0, ...
    useJointAlpha, alphaGrid, alpha_fixed_weak, alpha_fixed_strong, semParams)

    %decide weak or strong by |h|^2
    if abs(h1)^2 <= abs(h2)^2
        hw = h1; hs = h2; kw = w1; ks = w2;
    else
        hw = h2; hs = h1; kw = w2; ks = w1;
    end

    over = 1.0; %no antenna count overhead in this grouping reward

    if useJointAlpha
        [~, ~, ~, ~, ~, ~, ~, ~, Gw, Gs] = noma2_semantic_bestAlpha( ...
            hw, hs, kw, ks, Ptx, M, N0, alphaGrid, over, semParams);
    else
        [~, ~, ~, ~, ~, ~, Gw, Gs] = noma2_semantic_givenAlpha( ...
            hw, hs, kw, ks, Ptx, M, N0, alpha_fixed_weak, alpha_fixed_strong, over, semParams);
    end

    rew = Gw + Gs;
end

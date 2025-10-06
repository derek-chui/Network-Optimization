clear; clc; close all;

K = 4; %even users
M = 4; %antennas
rng('shuffle');

eta = 3;
Ptx = 10;
N0 = 1e-5;
alpha1 = 0.6;
alpha2 = 0.4;
sigma = 1;
L = 100;

G = 6.0*ones(1,M);
r0 = 40;
gamma = 1.5;

tau_sic = 0.04; %per extra antenna used inside a pair

BS = [0,0];

R = 70; th = linspace(0,2*pi,M+1); th(end)=[];
A = [R*cos(th(:)), R*sin(th(:))];

rayleigh = @(s) s*sqrt(-2*log(max(1e-12,rand)));
dist = @(X,Y) sqrt(sum((X - Y).^2, 2));
coupling = @(d) 1 ./ (1 + (d./r0).^gamma);

noma_pair = @(hw,hs) deal( ...
    log2(1 + (alpha1*Ptx*hw.^2) ./ (alpha2*Ptx*hw.^2 + N0)), ...
    log2(1 + (alpha2*Ptx*hs.^2) ./ N0) );

U = (rand(K,2)*2 - 1)*L;

dBU = dist(U, repmat(BS,K,1));
z = arrayfun(@(~) rayleigh(sigma), 1:K).';
h0 = z ./ (max(1e-3,dBU).^(eta/2)); %Kx1 direct gains

C = zeros(K,M);
for m = 1:M
    C(:,m) = coupling(dist(U, repmat(A(m,:),K,1))); %KxM user antenna coupling
end

[~, ord] = sort(abs(h0),'ascend');
h_sorted = h0(ord); idx_sorted = ord(:);
P = K/2;
weak_idx  = idx_sorted(1:P);
strong_idx= flipud(idx_sorted(P+1:end));

pair_users = [weak_idx, strong_idx]; %Px2 (u_w, u_s)

userColors = lines(K);

function W = simplex_grid(S, step)
    if S==1, W = 1; return; end
    vals = 0:step:1;
    if isempty(vals), W=1; return; end
    W = [];
    if S==2
        for a=vals
            b = 1-a;
            if b>=-1e-12, W=[W; a b]; end
        end
        return
    end
    function rec(prefix, remain, slots)
        if slots==1
            W = [W; prefix, remain];
            return
        end
        for v = 0:step:remain
            rec([prefix, v], remain - v, slots-1);
        end
    end
    rec([],1,S);
end

function [Ssum, perPairBest] = eval_assignment(assign, pair_users, h0, C, G, noma_pair, tau_sic, Ptx, N0, alpha1, alpha2)
    Pairs = size(pair_users,1);
    %prealloc struct array with consistent fields
    perPairBest = repmat(struct('ant',[],'w',[],'Rw',0,'Rs',0,'sum',0,'hw',0,'hs',0), Pairs,1);
    Ssum = 0;
    for p = 1:Pairs
        ants = find(assign==p); %antennas assigned exclusively to this pair
        uw = pair_users(p,1); us = pair_users(p,2);

        if isempty(ants)
            hw = h0(uw); hs = h0(us);
            [Rw,Rs] = noma_pair(hw,hs);
            perPairBest(p).ant = [];
            perPairBest(p).w   = [];
            perPairBest(p).hw  = hw;
            perPairBest(p).hs  = hs;
            perPairBest(p).Rw  = Rw;
            perPairBest(p).Rs  = Rs;
            perPairBest(p).sum = Rw+Rs;
            Ssum = Ssum + (Rw+Rs);
            continue
        end

        %init best candidate
        best = struct('ant',[],'w',[],'Rw',0,'Rs',0,'sum',0,'hw',0,'hs',0);
        Smax = -inf;

        step = 0.25; %power split resolution
        W = simplex_grid(numel(ants), step);
        if isempty(W), W = 1; end
        for r = 1:size(W,1)
            w = W(r,:);
            gw = 1 + sum( G(ants).*w .* C(uw,ants) );
            gs = 1 + sum( G(ants).*w .* C(us,ants) );
            hw = h0(uw)*gw; hs = h0(us)*gs;
            if abs(hw) > abs(hs) %ensure weak/strong
                tmp = hw; hw = hs; hs = tmp;
            end
            [Rw_raw, Rs_raw] = noma_pair(hw,hs);
            over = max(0, 1 - tau_sic*max(0, numel(ants)-1));
            Rw = Rw_raw;
            Rs = Rs_raw*over;
            Sval = Rw + Rs;
            if Sval > Smax
                Smax = Sval;
                best.ant = ants;
                best.w = w;
                best.hw = hw;
                best.hs = hs;
                best.Rw = Rw;
                best.Rs = Rs;
                best.sum = Sval;
            end
        end
        perPairBest(p) = best;
        Ssum = Ssum + Smax;
    end
end


if M <= 7 && P <= 3
    labels = 0:P; %0=unused, 1..P pair id
    allAssign = cell(1,M);
    [allAssign{:}] = ndgrid(labels);
    mats = cellfun(@(x) x(:), allAssign, 'uni',0);
    ASSIGN = [mats{:}]; %(#comb)xM
    bestS = -inf; bestAssign = []; bestPer = [];
    for r = 1:size(ASSIGN,1)
        a = ASSIGN(r,:);
        [Ssum, perPairBest] = eval_assignment(a, pair_users, h0, C, G, noma_pair, tau_sic, Ptx, N0, alpha1, alpha2);
        if Ssum > bestS
            bestS = Ssum; bestAssign = a; bestPer = perPairBest;
        end
    end
else
    a = zeros(1,M);
    improved = true;
    [bestS, bestPer] = eval_assignment(a, pair_users, h0, C, G, noma_pair, tau_sic, Ptx, N0, alpha1, alpha2);
    bestAssign = a;
    while improved
        improved = false;
        for m = 1:M
            for lab = 0:P
                if bestAssign(m)==lab, continue; end
                a_try = bestAssign; a_try(m)=lab;
                [S_try, per_try] = eval_assignment(a_try, pair_users, h0, C, G, noma_pair, tau_sic, Ptx, N0, alpha1, alpha2);
                if S_try > bestS
                    bestS = S_try; bestAssign = a_try; bestPer = per_try; improved = true;
                end
            end
        end
    end
end

pair_sum = arrayfun(@(s) s.sum, bestPer);
Rw_list  = arrayfun(@(s) s.Rw,  bestPer);
Rs_list  = arrayfun(@(s) s.Rs,  bestPer);

fprintf('K = %d, M = %d\n', K, M);
fprintf('Pairs:\n');
for p=1:P
    fprintf('  Pair %d: (U%d, U%d)\n', p, pair_users(p,1), pair_users(p,2));
end
fprintf('Best Assignment Per Antenna: [%s]\n', sprintf('%d', bestAssign));
for p=1:P
    ants = bestPer(p).ant;
    if isempty(ants)
        fprintf('  Pair %d: antennas {}, Sum=%.3f (Rw=%.3f, Rs=%.3f)\n', p, bestPer(p).sum, bestPer(p).Rw, bestPer(p).Rs);
    else
        ww = bestPer(p).w(:).';
        s = sprintf('A%d: %.2f, ', [ants; ww]);
        s(end-1:end)=[];
        fprintf('  Pair %d: {%s Weight}, Total Pair Utility = %.3f (Rw = %.3f, Rs = %.3f)\n', p, s, bestPer(p).sum, bestPer(p).Rw, bestPer(p).Rs);
    end
end
fprintf('Total Sum Rate: %.3f b/s/Hz, Mean Per Pair: %.3f\n', sum(pair_sum), mean(pair_sum));

figure('Color','w'); hold on; axis equal;
xlim([-L L]); ylim([-L L]);
xline(0,'k-','LineWidth',1); yline(0,'k-','LineWidth',1); box off;

plot(BS(1),BS(2),'ks','MarkerFaceColor','k','MarkerSize',9); text(BS(1)+2,BS(2),'BS');
for m=1:M
    on = bestAssign(m)~=0;
    face = on*[0 0 0] + ~on*[1 1 1];
    plot(A(m,1),A(m,2),'^','MarkerSize',9,'MarkerFaceColor',face,'Color','k');
    if on
        text(A(m,1)+2,A(m,2),sprintf('A%d→P%d',m,bestAssign(m)));
    else
        text(A(m,1)+2,A(m,2),sprintf('A%d',m));
    end
end

plot(U(:,1),U(:,2),'o','MarkerFaceColor',[0.2 0.6 1],'Color',[0 0.2 0.6]);
for i=1:K
    text(U(i,1)+2,U(i,2),sprintf('U%d',i));
end


for p=1:P
    u1 = pair_users(p,1); u2 = pair_users(p,2);
    plot([U(u1,1) U(u2,1)], [U(u1,2) U(u2,2)], '--', 'Color', 0.5*[1 1 1], 'LineWidth', 1.0);
end

pairColors = lines(P);

for m = 1:M
    pID = bestAssign(m);
    if pID == 0, continue; end
    uu = pair_users(pID,:);
    colorPair = pairColors(pID,:);
    for t = 1:2
        i = uu(t);
        plot([A(m,1) U(i,1)], [A(m,2) U(i,2)], '-', ...
             'Color', colorPair, 'LineWidth', 1.8);
    end
end

title(sprintf('Exclusive Antenna→Pair Assignment | Total Sum = %.3f b/s/Hz', sum(pair_sum)));
xlabel('x (m)'); ylabel('y (m)'); grid on;

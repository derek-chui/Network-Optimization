clear; clc; close all;

K = 4; %users must be even
M = 4; %antennas
rng('shuffle');

P = K/2;

eta = 3; %pathloss
Ptx = 10; %signal power P
N0 = 1e-5; %noise power or sigma^2
alpha1 = 0.6; %weak power
alpha2 = 0.4; %strong

tau_sic = 0.00; %>0 to penalize strong rate if many antennas

%for exp(-j*2pi/lambda*d))
c = 3e8;
fc = 28e9;
lambda = c/fc;

%area for user placement
L = 100;

%waveguide height to avoid singularity
d_wg = 3;

%base station and antennas
BS = [0,0];

R = 70;
th = linspace(0,2*pi,M+1); th(end)=[];
A_xy = [R*cos(th(:)), R*sin(th(:))];
A = [A_xy, d_wg*ones(M,1)]; %M*3 antenna positions

%users
U_xy = (rand(K,2)*2 - 1)*L;
U = [U_xy, zeros(K,1)]; %Kx3 user positions (z=0)

%channel where h_i = sum_n sqrt(eta)*e^{-j(2pi/lambda)d}/d * e^{-j theta_n}
theta = 2*pi*rand(1,M); %phase e^{-j theta_n}

%dist matrix KxM
D = zeros(K,M);
for m = 1:M
    diff = U - repmat(A(m,:),K,1);
    D(:,m) = sqrt(sum(diff.^2,2));
end
D = max(D, 1e-3); %prevent diving by 0

%contribution matrix Hc(i,m) = contribution from antenna m to user i
Hc = zeros(K,M);
for m = 1:M
    %PART 1 CHANNEL
    Hc(:,m) = (sqrt(eta) ./ D(:,m)) .* exp(-1j*(2*pi/lambda)*D(:,m)) .* exp(-1j*theta(m));
end

%baseline channel if all antennas active (pairing order)
h_all = sum(Hc,2);
[~, ord] = sort(abs(h_all).^2,'ascend');
idx_sorted = ord(:);

weak_idx  = idx_sorted(1:P);
strong_idx= flipud(idx_sorted(P+1:end));
pair_users = [weak_idx, strong_idx]; % Px2

%assignment search (0=unused, 1..P pair id0)
function [Ssum, perPairBest] = eval_assignment(assign, pair_users, Hc, Ptx, M, N0, alpha1, alpha2, tau_sic)
    Pairs = size(pair_users,1);
    perPairBest = repmat(struct( ...
        'ant',[], ...
        'weakUser',[], ...
        'strongUser',[], ...
        'hw',0, ...
        'hs',0, ...
        'Rw',0, ...
        'Rs',0, ...
        'sum',0), Pairs,1);

    Ssum = 0;

    for p = 1:Pairs
        ants = find(assign==p); %antennas assigned to this pair
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

        %determine weak/strong via |h|^2
        if abs(hA)^2 <= abs(hB)^2
            uw = uA; us = uB; hw = hA; hs = hB;
        else
            uw = uB; us = uA; hw = hB; hs = hA;
        end

        %2 user group noma rates:
        %Rw = min{Rw decoded at weak, Rw decoded at strong}; Rs after SIC at strong
        [Rw, Rs] = noma2_paper(hw, hs, Ptx, M, N0, alpha1, alpha2);

        %penalty overhead
        over = max(0, 1 - tau_sic*max(0, numel(ants)-1));
        Rs = Rs * over;

        perPairBest(p).ant = ants;
        perPairBest(p).weakUser = uw;
        perPairBest(p).strongUser = us;
        perPairBest(p).hw = hw;
        perPairBest(p).hs = hs;
        perPairBest(p).Rw = Rw;
        perPairBest(p).Rs = Rs;
        perPairBest(p).sum = Rw + Rs;

        Ssum = Ssum + (Rw + Rs);
    end
end

function [Rw, Rs] = noma2_paper(hw, hs, Ptx, M, N0, alpha1, alpha2)
    %hw hs channel coefficients
    %power evenly distributed P/M
    Ppa = Ptx / M;

    gw = abs(hw)^2;
    gs = abs(hs)^2;

    %SIC
    Rw_ww = log2(1 + (gw*Ppa*alpha1) / (gw*Ppa*alpha2 + N0));
    Rw_sw = log2(1 + (gs*Ppa*alpha1) / (gs*Ppa*alpha2 + N0));
    Rw = min(Rw_ww, Rw_sw);

    %strong user after SIC

    %PART 2 RATES
    Rs = log2(1 + (gs*Ppa*alpha2) / N0);
end

%search best assignment
if M <= 7 && P <= 3
    labels = 0:P; %0 is unused, 1...P
    allAssign = cell(1,M);
    [allAssign{:}] = ndgrid(labels);
    mats = cellfun(@(x) x(:), allAssign, 'uni',0);
    ASSIGN = [mats{:}];

    bestS = -inf; bestAssign = []; bestPer = [];
    for r = 1:size(ASSIGN,1)
        a = ASSIGN(r,:);
        [Ssum, perPairBest] = eval_assignment(a, pair_users, Hc, Ptx, M, N0, alpha1, alpha2, tau_sic);
        if Ssum > bestS
            bestS = Ssum; bestAssign = a; bestPer = perPairBest;
        end
    end
else
    %hillclimb fallback
    a = zeros(1,M);
    [bestS, bestPer] = eval_assignment(a, pair_users, Hc, Ptx, M, N0, alpha1, alpha2, tau_sic);
    bestAssign = a;

    improved = true;
    while improved
        improved = false;
        for m = 1:M
            for lab = 0:P
                if bestAssign(m)==lab, continue; end
                a_try = bestAssign; a_try(m)=lab;
                [S_try, per_try] = eval_assignment(a_try, pair_users, Hc, Ptx, M, N0, alpha1, alpha2, tau_sic);
                if S_try > bestS
                    bestS = S_try; bestAssign = a_try; bestPer = per_try; improved = true;
                end
            end
        end
    end
end

pair_sum = arrayfun(@(s) s.sum, bestPer);

%print results
fprintf('K = %d, M = %d\n', K, M);
fprintf('Pairs:\n');
for p=1:P
    fprintf('  Pair %d: (U%d, U%d)\n', p, pair_users(p,1), pair_users(p,2));
end
fprintf('Best Assignment Per Antenna: [%s]\n', sprintf('%d', bestAssign));

for p=1:P
    ants = bestPer(p).ant;
    uw = bestPer(p).weakUser;
    us = bestPer(p).strongUser;
    if isempty(ants)
        fprintf('  Pair %d: Antennas {} | (Weak User, Strong User) = (U%d,U%d) | Group Sum Rate = %.3f (Weak User Rate = %.3f, Strong User Rate = %.3f)\n', ...
            p, uw, us, bestPer(p).sum, bestPer(p).Rw, bestPer(p).Rs);
    else
        s = sprintf('A%d, ', ants);
        s(end-1:end)=[];
        fprintf('  Pair %d: Antennas {%s} | (Weak User, Strong User) = (U%d,U%d) | Antennas Used = %d | Group Sum Rate = %.3f (Weak User Rate = %.3f, Strong User Rate = %.3f)\n', ...
            p, s, uw, us, numel(ants), bestPer(p).sum, bestPer(p).Rw, bestPer(p).Rs);
    end
end
fprintf('Total Sum Rate: %.3f b/s/Hz, Mean Per Pair: %.3f\n', sum(pair_sum), mean(pair_sum));

%plot
figure('Color','w'); hold on; axis equal;
xlim([-L L]); ylim([-L L]);
xline(0,'k-','LineWidth',1); yline(0,'k-','LineWidth',1); box off;

plot(BS(1),BS(2),'ks','MarkerFaceColor','k','MarkerSize',9); text(BS(1)+2,BS(2),'BS');

%antennas
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

%users
plot(U(:,1),U(:,2),'o','MarkerFaceColor',[0.2 0.6 1],'Color',[0 0.2 0.6]);
for i=1:K
    text(U(i,1)+2,U(i,2),sprintf('U%d',i));
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
        plot([A(m,1) U(i,1)], [A(m,2) U(i,2)], '-', ...
             'Color', colorPair, 'LineWidth', 1.8);
    end
end

title(sprintf('NOMA Pinch Antennas: Total Sum = %.3f b/s/Hz', sum(pair_sum)));
xlabel('x (m)'); ylabel('y (m)'); grid on;

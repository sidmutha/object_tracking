function [labels] = spectralClustering(W, k)
%SPECTRALCLUSTERING Performs spectral clustering on a graph
% Parameters
%   W - adjacency matrix of the graph
%   k - number of clusters
% Returns
%   labels - cluster labels for each of the graph nodes
    D = diag(sum(W));
    [V, ~] = symLaplacianEig(W, D);
    labels = kmeans(V, k);
end

function [V, S] = unnormLaplacianEig(W, D)
    L = D-W;
    [V, S] = eigs(L, 'sm');
%     [Vt, St] = eig(L);
%     [~, idx] = sort(diag(St));
%     idx = idx(1:k);
%     V = Vt(:, idx);
%     S = St(idx, idx);
end

function [V, S] = symLaplacianEig(W, D)
    L = D-W;
    [Vt, St] = eigs(L, D, 'sm');
    vSt = diag(St);
    idx = vSt < 0.2;
%     [Vt, St] = eig(L, D);
%     [~, idx] = sort(diag(St));
%     idx = idx(1:k);
    V = Vt(:, idx);
    S = St(idx, idx);
end

function [V, S] = rwLaplacianEig(W, D)
    L = D-W;
    HD = D^(-0.5);
    Lsym = HD*L*HD;
    [V, S] = eigs(Lsym, 'sm');
%     [Vt, St] = eig(Lsym);
%     [~, idx] = sort(diag(St));
%     idx = idx(1:k);
%     V = Vt(:, idx);
%     S = St(idx, idx);
end
alphaRange = 0.1:0.1:0.9;
sigmaRange = 1:5:31;
lambdaRange = round(logspace(0,2,10));
thetaRange = 1.5:0.5:2.5;
video = 'cars4';

acc = zeros(length(sigmaRange), length(lambdaRange), length(alphaRange), length(thetaRange));

for i = 1:length(sigmaRange)
    s = sigmaRange(i);
    for j = 1:length(lambdaRange)
        l = lambdaRange(j);
        for k = 1:length(alphaRange)
            a = alphaRange(k);
            for m = 1:length(thetaRange)
                t = thetaRange(m);
                acc(i, j, k, m) = sparseToDense(video, a, s, l, t, 0);
            end
        end
    end
end

[bestacc, bestidx] = max(acc(:));
display(bestacc);
[i, j, k, m] = ind2sub(size(acc), bestidx);
s = sigmaRange(i);
l = lambdaRange(j);
a = alphaRange(k);
t = thetaRange(m);
fprintf('sigma = %f', s);
fprintf('lambda = %f', l);
fprintf('alpha = %f', a);
fprintf('theta = %f', t);
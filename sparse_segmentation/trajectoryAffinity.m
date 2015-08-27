function [w] = trajectoryAffinity(floCell, trajectory1, trajectory2, lambda)
    w = exp(-lambda*trajectoryDistance(floCell, trajectory1, trajectory2));
end
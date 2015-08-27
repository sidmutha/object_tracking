function [s] = minLocalFlowVarianceSum(floCell, trajectory1, trajectory2, frameNo, l)
    s = min(localFlowVariance_sum(floCell, trajectory1, frameNo, l), localFlowVariance_sum(floCell, trajectory2, frameNo, l));
end
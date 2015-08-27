function [M] = affinityMatrix(floCell, trajectories)
    M = ones(length(trajectories));
    for i = 1:length(trajectories)-1
        for j = i+1:length(trajectories) % symmetric
            a = trajectoryAffinity(floCell, trajectories{i}, trajectories{j}, 0.1);
            M(i, j) = a;
            M(j, i) = a;
        end
    end
end
function [s, l] = localFlowVariance_sum(floCell, trajectory, frameNo, l)
    index = frameNo - trajectory.startFrame + 1; % index of frame from start
    
    if frameNo + l > trajectory.endFrame
        l = trajectory.endFrame - frameNo;
    end
    
    s = 0;
    for i = 1:l
        z = trajectory.points(index + i, :);
        s = s + localFlowVariance(floCell, z(1), z(2), 2, frameNo);
    end
end
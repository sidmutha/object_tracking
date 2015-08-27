function [d] = trajectoryFrameDist(floCell, trajectory1, trajectory2, frameNo)
%     dsp = meanSpatialDistBetweenTrajectories(trajectory1, trajectory2);
    lfv = minLocalFlowVarianceSum(floCell, trajectory1, trajectory2, frameNo, 5); % l = 5
    
    f1 = frameNo - trajectory1.startFrame + 1;
    f2 = frameNo - trajectory2.startFrame + 1;
    
    pt1_5 = trajectory1.points(min(f1+5, trajectory1.numPoints), 1:2);
    pt1_0 = trajectory1.points(1, 1:2);
    
    pt2_5 = trajectory2.points(min(f2+5, trajectory2.numPoints), 1:2);
    pt2_0 = trajectory2.points(1, 1:2);
    
    vel1 = (pt1_5 - pt1_0);
    vel2 = (pt2_5 - pt2_0);
    
    vel_dist = sum((vel1 - vel2).^2);
    
    d = vel_dist / (5*lfv);
%     d = dsp * d;
end
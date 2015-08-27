function[d] = meanSpatialDist(trajectory1, trajectory2)
    s = max(trajectory1.startFrame, trajectory2.startFrame);
    e = min(trajectory1.endFrame, trajectory2.endFrame);

    s1 = abs(s - trajectory1.startFrame) + 1;
    e1 = abs(e - trajectory1.startFrame) + 1;

    s2 = abs(s - trajectory2.startFrame) + 1;
    e2 = abs(e - trajectory2.startFrame) + 1;

    p1 = trajectory1.points(s1:e1, 1:2);
    p2 = trajectory2.points(s2:e2, 1:2);

    diff = p1 - p2;
    diff2 = diff.^2;
    sdiff = sum(diff2, 2);
    d = mean(sqrt(sdiff));
end
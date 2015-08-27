function [d] = trajectoryDistance(floCell, trajectory1, trajectory2)
    if ((trajectory1.startFrame > trajectory2.endFrame) || (trajectory2.startFrame > trajectory1.endFrame))
        d = inf;
    else
        s = max(trajectory1.startFrame, trajectory2.startFrame);
        e = min(trajectory1.endFrame, trajectory2.endFrame);

        d_ = zeros(e-s+1, 1);
        i = 1;
        %size(d_)

        dsp = meanSpatialDistBetweenTrajectories(trajectory1, trajectory2);
        for f = s:e
            d_(i) = trajectoryFrameDist(floCell, trajectory1, trajectory2, f+1);
            i = i + 1;
        end
        d_ = dsp.*d_;
        d = max(d_);
    end
end
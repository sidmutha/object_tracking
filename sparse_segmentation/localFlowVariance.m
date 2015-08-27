function [v] = localFlowVariance(floCell, x, y, w, frameNo)
    % w = window size
    %frameNo
    floFrame = floCell{frameNo};
    x1 = x - w;
    x2 = x + w;
    y1 = y - w;
    y2 = y + w;
    
    x1 = ceil(max(x1, 1));
    y1 = ceil(max(y1, 1));
        
    m_x = size(floFrame, 2);    
    x2 = ceil(min(x2, m_x));
    
    m_y = size(floFrame, 1);    
    y2 = ceil(min(y2, m_y));
    
    submat = floFrame(y1:y2, x1:x2, :);
    s = sqrt(submat(:, :, 1).^2 + submat(:, :, 2).^2);
    %r = reshape(s, [numel(s) ,1]);
    v = var(s(:));
end
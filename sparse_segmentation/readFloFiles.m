function [floCell] = readFloFiles(dirname, prefix, frange)
    floCell = cell(length(frange), 1);
    j = 1;
    for i = frange
        floCell{j} = readFlowFile([dirname, '/', prefix, sprintf('%03d', i), '.flo']);
        j = j + 1;
    end
end
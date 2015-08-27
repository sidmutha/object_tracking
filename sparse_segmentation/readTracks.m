function [trajectories] = readTracks(fname)
   f = fopen(fname);
   [~] = str2num(fgetl(f)); % get max no of frames
   aCount = str2num(fgetl(f)); % get number of trajectories
   trajectories = cell(aCount, 1);
   for i = 1:aCount
       l = fgetl(f);
       sp = str2num(l);
       mLabel = sp(1);
       aSize = sp(2);
       %mLabel
       pointsArr = zeros(aSize, 3);
       for j = 1:aSize
           ss = str2num(fgetl(f));
           pointsArr(j, 1) = ss(1);
           pointsArr(j, 2) = ss(2);
           pointsArr(j, 3) = ss(3);
       end
       s = struct('mLabel', mLabel, 'points', pointsArr, 'startFrame', pointsArr(1, 3), 'endFrame', pointsArr(1, 3) + aSize - 1, 'numPoints', aSize);
       trajectories{i} = s;
   end
   
   fclose(f);
end

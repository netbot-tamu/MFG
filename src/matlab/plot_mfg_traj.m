%% visualize camera poses/trajectory
clear; clc; close all

% fname indicates the pose file
fname = '/home/lu/mfg/build/camPose.txt';
data = load(fname);

campos = [0; 0; 0];
angleaxis = [0 1 0 0];
for i = 2:size(data,1)
    Ri = reshape(data(i,3:11),3,3)';
    angleaxis(end+1,:) = vrrotmat2vec(Ri);
    ti = data(i,12:14)';
    campos(:,end+1) = -inv(Ri)*ti;
    itv = data(i,2)-data(i-1,2);
    bs = norm(campos(:,end)-campos(:,end-1));
end

figure,plot3(campos(1,:),campos(2,:),campos(3,:),'-')
axis equal
view([0 -1 0])
xlabel('X'); ylabel('Y'); zlabel('Z');
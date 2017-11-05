function [mv_data, framePts, frameInd, frameType] = load_mv(filename,T, play) 
%% Load MVs ...
fid = fopen(filename) ;

%% Get # Rows and # Cols ..
% Skip PTS and Ind ..
precision = 'int' ; 
fread(fid,1,precision)
fread(fid,1,precision)

xdim = fread(fid, 1, precision)         
ydim = fread(fid, 1, precision)
fclose(fid) ;

% Initialize image sequence ..
mv_data = zeros(1, 1, 1) ; frameType = zeros(1) ;
framePts = zeros(1) ; frameInd = zeros(1) ;

fid = fopen(filename) ;
k = 1 ;
while(true)

fprintf('Loading frame %d ..\n', k) ;
pts = fread(fid,1,'int') ; 

% Load PTS and check for EOF ..
if isempty(pts)
	break
end

framePts(k) = pts ; 
frameInd(k) = fread(fid,1,'int') ;
% Skip X/Y Dims ..
fread(fid,1,'int') ;fread(fid,1,'int') ;
frameType(k) = fread(fid,1,'char') ;

% Read data ..
for i = 1:ydim
    for j = 1:xdim
	mv_data(i, j, k) = fread(fid,1,'signed char') ;
    end
end

k = k + 1 ;
end

% keyboard

% Close file
fclose(fid) ;

if (play)
implay((abs(mv_data(1:end/2,:,:))+abs(mv_data((end/2)+1:end,:,:)))./T)
end


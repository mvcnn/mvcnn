function [rgb] = play_rgb(filename, x, y)

fid=fopen(filename,'rb') ;
tmp = fread(fid, 'uint8') ;

data = vec2mat(tmp, x) ;
numFrame = size(data, 1)/(y*3) ;
rgb = zeros(y, x, 3, numFrame) ;

for i = 1:numFrame
    
    
    r_sind =((i-1)*3*y)+1 ; r_eind = r_sind + y - 1 ;
    g_sind = r_eind + 1 ; g_eind = g_sind + y - 1 ;
    b_sind = g_eind + 1 ; b_eind = b_sind + y - 1 ;
    
    rgb(:, :, 2, i) = data(r_sind:r_eind,:) ;
    rgb(:, :, 3, i) = data(g_sind:g_eind,:) ;
    rgb(:, :, 1, i) = data(b_sind:b_eind,:) ;
    
end


mov = immovie(rgb./255) ; implay(mov) ;
% implay(squeeze(rgb(:,:,1,:))./255, 25) ;
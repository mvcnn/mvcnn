function [epe] = calculate_epe(avmv, jmmv, play)

[av_mv_data] = load_mv(avmv, 10, play) ;
[jm_mv_data] = load_mv(jmmv, 10, play) ;

numFrame = min(size(jm_mv_data,3), size(av_mv_data,3)) ;

jmdx = jm_mv_data(1:end/2, :, 1:numFrame) ;
jmdy = jm_mv_data((end/2)+1:end, :, 1:numFrame) ;

avdx = av_mv_data(1:end/2, :, 1:numFrame) ;
avdy = av_mv_data((end/2)+1:end, :, 1:numFrame) ;    
   
sqdiff_dx = avdx - jmdx ;
sqdiff_dx = sqdiff_dx.^2 ;
    
sqdiff_dy = avdy - jmdy ;
sqdiff_dy = sqdiff_dy.^2 ;
    
sumsqdiff = sqdiff_dx + sqdiff_dy ;
    
epe = mean(mean(mean(sumsqdiff))) ;
end
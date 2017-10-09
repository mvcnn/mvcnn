#include "all_defines.h"

AVL_H264::AVL_H264()
{ 
        mvout = fopen(OUT_PATH,"wb") ;
        rgbout = fopen(RGB_PATH,"wb") ;
	pts = -1 ; frame.index = 0 ; 
}

void AVL_H264::initialize()
{
	av_register_all() ;
	avframe = av_frame_alloc() ;
	avcontext = avformat_alloc_context() ;
	avformat_open_input(&avcontext, VIDEO_PATH, NULL, NULL) ;
	avformat_find_stream_info(avcontext, NULL) ;

	avstream_index = -1;
	for(int i = 0; i < avcontext->nb_streams; i++)
	{
		AVCodecContext *enc = avcontext->streams[i]->codec;
		if( enc->codec_type == AVMEDIA_TYPE_VIDEO && avstream_index < 0 )
		{
			AVCodec *pCodec = 
				avcodec_find_decoder(enc->codec_id); 
			AVDictionary *opts = NULL ;
			av_dict_set(&opts, "flags2", "+export_mvs", 0);

			avcodec_open2(enc, pCodec, &opts) ;
			avstream_index = i ; avstream = avcontext->streams[i] ;
			// frame_width = enc->width ; frame_height = enc->height ;
			frame.width = enc->width ; frame.height = enc->height ;
			break ; // Break once stream is set
		}
	}

}

bool AVL_H264::decode_packets()
{
	static bool doproc = false;
	static AVPacket pkt, pkt_ ;

	int ret ;
	while(true)
	{
		if(doproc)
		{
			AVPacket *_pkt = &pkt_ ;
			int stat ;int got_frame = -1 ;

			av_frame_unref(avframe) ;
			stat = avcodec_decode_video2(avstream->codec, 
						avframe, &got_frame, _pkt) ;

			if (stat < 0) { av_free_packet(&pkt) ;
					doproc = false  ; }
			else { 
			stat = FFMIN(stat, _pkt->size) ; 
			_pkt->data += stat ; _pkt->size -= stat ;

			if(got_frame > 0) return true ;
			else { av_free_packet(&pkt) ;
			       doproc = false  ; } 
			     }
		}
		
		ret = av_read_frame(avcontext, &pkt) ;
		// Break if end reached --
		if(ret != 0) break ;

		doproc = true ; pkt_ = pkt ;
		// Ignore packets from other streams 
		if(pkt.stream_index != avstream_index)
		{
			doproc = false ; av_free_packet(&pkt) ;
			continue ; }
	}

	// return process_frame(&pkt) ;
       return false ; 
}

bool AVL_H264::get_motion_vectors()
{
	fprintf(stdout, "here %d %d\n", pts, frame.pts) ;
	if(!decode_packets()) {fclose(mvout) ; fclose(rgbout) ; return false ;}
	

	if      (avframe->pkt_pts != AV_NOPTS_VALUE) 
		pts = avframe->pkt_pts ;
	else if (avframe->pkt_dts != AV_NOPTS_VALUE) 
		pts = avframe->pkt_dts ;
        else 	pts = pts + 1 ;

	fprintf(stdout, "here2 %d %d\n", pts, frame.pts) ;
	if (pts <= frame.pts && frame.pts != -1) return true ;

	type = av_get_picture_type_char(avframe->pict_type) ;

	bool skip = false ; 
        if (av_frame_get_side_data(avframe, AV_FRAME_DATA_MOTION_VECTORS) == NULL) 
	   skip = true ;

        avmv = vector<AVMotionVector>();
	if(!skip)
	{
		AVFrameSideData* side_data = 
			av_frame_get_side_data(avframe, AV_FRAME_DATA_MOTION_VECTORS);
		AVMotionVector* avmv_ = (AVMotionVector*) side_data->data ;

		int count = side_data->size / sizeof(AVMotionVector) ;
		avmv = vector<AVMotionVector>(avmv_, avmv_ + count) ;
	}

        frame.pts = pts ; frame.index = frame.index + 1 ;
	frame.type = type ; 

	fprintf(stdout, "Writing to: %s\n", OUT_PATH) ;

	frame.setup(avframe, avmv) ;
        if(!avmv.empty()) { frame.print(mvout, rgbout) ; } ;
	return true;
}

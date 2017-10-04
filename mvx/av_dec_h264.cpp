#include "all_defines.h"

void AVL_H264::initialize_av()
{
	av_register_all() ;
	av_frame = av_frame_alloc() ;
	av_context = avformat_alloc_context() ;
	avformat_open_input(&av_context, VIDEO_PATH, NULL, NULL) ;
	avformat_find_stream_info(av_context, NULL) ;

	av_stream_index = -1;
	for(int i = 0; i < av_context->nb_streams; i++)
	{
		AVCodecContext *enc = av_context->streams[i]->codec;
		if( enc->codec_type == AVMEDIA_TYPE_VIDEO && av_stream_index < 0 )
		{
			AVCodec *pCodec = 
				avcodec_find_decoder(enc->codec_id); 
			AVDictionary *opts = NULL ;
			av_dict_set(&opts, "flags2", "+export_mvs", 0);

			avcodec_open2(enc, pCodec, &opts) ;
			av_stream_index = i ; av_stream = av_context->streams[i] ;
			frame_width = enc->width ; frame_height = enc->height ;

			break ; // Break once stream is set
		}
	}

}

bool AVL_H264::decode_packet(FILE* mout)
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

			av_frame_unref(av_frame) ;
			stat = avcodec_decode_video2(av_stream->codec, 
						av_frame, &got_frame, _pkt) ;

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
		
		ret = av_read_frame(av_context, &pkt) ;
		// Break if end reached --
		if(ret != 0) break ;

		doproc = true ; pkt_ = pkt ;
		// Ignore packets from other streams 
		if(pkt.stream_index != av_stream_index)
		{
			doproc = false ; av_free_packet(&pkt) ;
			continue ; }
	}

	// return process_frame(&pkt) ;
       return false ; 
}

bool AVL_H264::get_motion_vectors(int64_t& pts, char& pictType, vector<AVMotionVector>& motion_vectors, AVFrame& avframe,  FILE* mout)
{
	if(!read_packets(mout)) return false ;
	
	pictType = av_get_picture_type_char(av_frame->pict_type) ;

	if (av_frame->pkt_pts != AV_NOPTS_VALUE) 
		pts = av_frame->pkt_pts ;
        else 	pts = pts + 1 ;

	avframe = *av_frame ;
	bool noMotionVectors = false ; 
        if (av_frame_get_side_data(av_frame, AV_FRAME_DATA_MOTION_VECTORS) == NULL) 
	   noMotionVectors = true ;

	motion_vectors = vector<AVMotionVector>();
	if(!noMotionVectors)
	{
		// Reference: doc/examples/extract_mvs.c
		AVFrameSideData* side_data = 
			av_frame_get_side_data(av_frame, AV_FRAME_DATA_MOTION_VECTORS);
		AVMotionVector* av_mvs = (AVMotionVector*) side_data->data ;

		int av_mvs_count = side_data->size / sizeof(AVMotionVector) ;
		motion_vectors = vector<AVMotionVector>(av_mvs, av_mvs + av_mvs_count) ;
	}

	return true;
}

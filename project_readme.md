
steps to run :  
    (env:feat-up (anaconda))
    1) run FeatUp/dinov2_featup.py to get the featup embedded for all the videos in your path / your video  
    if get error run : rm -rf /home/dor.danino/.cache/torch/hub/mhamilton723_FeatUp_main

    (env:dino-tracker):
    2) if want use dino raw Run preprocessing/save_dino_embed_video.py to get dinov2 embedded for all the videos in davis480
    if get catch error , run :  rm -rf /home/dor.danino/.cache/torch/hub/facebookresearch_dinov2_main 

    3) Run visualization/visualize_raw_heatmaps.py to get trajectories and occlusion for all the videos in davis480, need running 2 time , each time with one of the next args:
        change : 
        args.dino_embed_video_path = os.path.join(folder_path, "dino_embeddings/dino_embed_video.pt") # for dinov2 
        args.output_path = os.path.join(output_video_folder, "dinov2")
        args.trajectory_output_path = os.path.join(output_video_folder, "dinov2_grid_trajectory")
        args.occlusion_output_path = os.path.join(output_video_folder, "dinov2_grid_occlusion")
        to :
        args.dino_embed_video_path = os.path.join(output_video_folder, "dino_featup.pt")
        args.output_path = os.path.join(output_video_folder, "dinov2_featup")
        args.trajectory_output_path = os.path.join(output_video_folder, "dinov2_featup_grid_trajectory")
        args.occlusion_output_path = os.path.join(output_video_folder, "dinov2_featup_grid_occlusion")

        if you want get heatmap for some point change --point from benchmark_grid to your point , for example [495, 150, 0]
    
    4) for heatmap videos run visualization/visualize_raw_heatmaps.py with save_video = 1 , choose in the main which video to work on

    5) for figure with visualization run:visualization/visualize_all.py make sure update all the paths in the end of the script 

    6) run visualization/visualize_pred_vs_gt.py to get pred_vs_gt , make sure update all the paths in the end of the script 

    7) run  eval/eval_benchmark.py. https://technion.zoom.us/j/94016502847?pwd=WrnN97abPzTJHN0t1sSGbnxPp76ohv.1

    * to run dino fit3d model use conda activate fit3d and run FiT3D/dinov2_fit3d.py 
    * to run dino loftup modek use conda loftup and run loftup/dino_loftup.py

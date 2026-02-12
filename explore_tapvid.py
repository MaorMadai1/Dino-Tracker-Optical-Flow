import pickle
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description="Explore TapVid benchmark data")
parser.add_argument("--video-id", type=int, default=None, help="Video ID to explore (default: show all videos)")
parser.add_argument("--benchmark-path", type=str, default="tapvid/tapvid_davis_data_strided.pkl", help="Path to benchmark pickle file")
parser.add_argument("--csv-output", type=str, default="tapvid_benchmark_points.csv", help="Output CSV file path")
args = parser.parse_args()

# Load the benchmark data
benchmark_data = pickle.load(open(args.benchmark_path, "rb"))

print(f"Total videos in benchmark: {len(benchmark_data['videos'])}")
print("=" * 60)

# Prepare CSV file
csv_file = open(args.csv_output, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['video_idx', 'query_frame', 'point_idx', 'query_x', 'query_y', 
                     'track_start_x', 'track_start_y', 'track_end_x', 'track_end_y',
                     'visible_frames', 'total_frames'])

# Filter videos if video_id is specified
if args.video_id is not None:
    videos_to_show = [vc for vc in benchmark_data["videos"] if vc["video_idx"] == args.video_id]
    if not videos_to_show:
        print(f"Video ID {args.video_id} not found!")
        print(f"Available video IDs: {[vc['video_idx'] for vc in benchmark_data['videos']]}")
        exit(1)
else:
    videos_to_show = benchmark_data["videos"]

# Iterate through selected videos
for video_config in videos_to_show:
    video_idx = video_config["video_idx"]
    w, h = video_config["w"], video_config["h"]
    
    print(f"\n{'='*60}")
    print(f"VIDEO {video_idx}")
    print(f"{'='*60}")
    print(f"  Resolution: {w} x {h}")
    
    query_points = video_config["query_points"]
    target_points = video_config["target_points"]
    occluded = video_config["occluded"]
    
    print(f"  Query frames: {sorted(query_points.keys())}")
    
    # Only process frame 0
    if 0 not in query_points:
        print(f"  WARNING: Frame 0 not found in query_points for video {video_idx}")
        continue
    
    total_points = 0
    for frame_idx in [0]:
        pts = query_points[frame_idx]
        tracks = target_points[frame_idx]
        occ = occluded[frame_idx]
        
        num_points = len(pts)
        total_points += num_points
        num_frames = tracks.shape[1] if isinstance(tracks, np.ndarray) else len(tracks[0])
        
        print(f"\n  Frame {frame_idx}:")
        print(f"    Num query points: {num_points}")
        print(f"    Track length (frames): {num_frames}")
        
        # Show all query points
        for i, pt in enumerate(pts):
            print(f"    Point {i}: (x={pt[0]:.1f}, y={pt[1]:.1f})")
            
            # Show trajectory for this point (first and last positions)
            if isinstance(tracks, np.ndarray):
                track = tracks[i]  # Shape: (T, 2)
                occ_track = occ[i]  # Shape: (T,)
            else:
                # Handle list format
                track = np.array(tracks[i]) if not isinstance(tracks[i], np.ndarray) else tracks[i]
                occ_track = np.array(occ[i]) if not isinstance(occ[i], np.ndarray) else occ[i]
            
            first_pos = track[0]
            last_pos = track[-1]
            num_visible = np.sum(~occ_track.astype(bool))
            print(f"      Trajectory: start=({first_pos[0]:.1f}, {first_pos[1]:.1f}) -> end=({last_pos[0]:.1f}, {last_pos[1]:.1f})")
            print(f"      Visible frames: {num_visible}/{len(occ_track)}")
            
            # Write to CSV
            csv_writer.writerow([
                video_idx, frame_idx, i, pt[0], pt[1],
                first_pos[0], first_pos[1], last_pos[0], last_pos[1],
                num_visible, len(occ_track)
            ])
    
    print(f"\n  TOTAL query points in video {video_idx}: {total_points}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for vc in benchmark_data["videos"]:
    total = len(vc["query_points"].get(0, [])) if 0 in vc["query_points"] else 0
    print(f"  Video {vc['video_idx']:2d}: {total:4d} query points from frame 0")

# Close CSV file
csv_file.close()
print(f"\nCSV file saved to: {args.csv_output}")
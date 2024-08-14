from datetime import datetime
import os
import shutil
import yaml
from image_utils import RTSPVideoRecorder, get_most_recent_file
from generate_masks_mobilenet import create_masks
from overlay_memmaps import overlay_videos



def reset_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, remove it and all its contents
        shutil.rmtree(directory_path)
    
    # Recreate the directory
    os.makedirs(directory_path)



if __name__ == "__main__":
    # Read the config file.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RTSP_URL = config["rtsp_url"]
    HEIGHT = config["height"]
    WIDTH = config["width"]
    DURATION = config["duration"]
    FPS = config["fps"]

    TMP_VIDEO_PATH = "_new_video.mp4"
    NEW_MASK_MEMMAP_PATH = "_new_video.dat"
    COMPOSITES_DIR = "composites"

    # Clear composites from the composites directory because they might not match
    # the position of the camera now.
    reset_directory(COMPOSITES_DIR)

    # Record a new video.
    print(f"--- Starting up... Recording some new video: {TMP_VIDEO_PATH}")
    recorder = RTSPVideoRecorder(rtsp_url=RTSP_URL,
                                 output_file=TMP_VIDEO_PATH,
                                 duration=DURATION,
                                 fps=FPS)
    recorder.record()

    # Convert the video to a masked memmap file.
    print(f"--- Converting the video to a masked memmap: {NEW_MASK_MEMMAP_PATH}")
    create_masks(path_to_video_file=TMP_VIDEO_PATH,
                 output_frame_memmaps=NEW_MASK_MEMMAP_PATH,
                 output_frame_mask_memmaps="_new_video_mask.dat")
    
    # Copy mask to the composites directory.
    composite_memmap_copy_path = os.path.join(COMPOSITES_DIR, NEW_MASK_MEMMAP_PATH)
    print(f"--- Copying {NEW_MASK_MEMMAP_PATH} to {composite_memmap_copy_path}")
    shutil.copy(NEW_MASK_MEMMAP_PATH, composite_memmap_copy_path)


    while True:
        # Get the time for file naming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        print("1) Recording video")
        # Record 5 seconds of video
        recorder = RTSPVideoRecorder(rtsp_url=RTSP_URL,
                                     output_file=TMP_VIDEO_PATH,
                                     duration=DURATION,
                                     fps=FPS)
        recorder.record()

        # Create masks for the video
        
        print(f"2) Creating mask: {NEW_MASK_MEMMAP_PATH}")
        create_masks(path_to_video_file=TMP_VIDEO_PATH,
                     output_frame_memmaps=NEW_MASK_MEMMAP_PATH,
                     output_frame_mask_memmaps="_new_video_mask.dat")

        # Overlay with the previously created composite video
        most_recent_composite = get_most_recent_file(COMPOSITES_DIR)
        output_memmap_path = os.path.join(COMPOSITES_DIR, current_time) + ".dat"

        print(f"3) Creating {output_memmap_path} from {most_recent_composite} and {NEW_MASK_MEMMAP_PATH}")
        overlay_videos(background_video_memmap=most_recent_composite,
                       foreground_video_memmap=NEW_MASK_MEMMAP_PATH,
                       output_video_memmap=output_memmap_path,
                       height=HEIGHT,
                       width=WIDTH)
        
        print("--------------------------------------------------------------")

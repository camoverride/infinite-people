# Infinite People

## Setup Wifi Camera


## Test

There must be a `new_video.dat` and at least one `*.dat` file in `composites/`

Generate new composite masks:
- `python generate_and_overlay.py`

Stream the video with the composites looping:
- `python stream_combine.py`


## TODO:

- no need for mask.dat files
- try to optimize mask overlay with tensorflow for Coral
- video files need to be created before `python stream_combine.py` is run
- clear out extra composite files
- do not hard-code image height/width - calculate it once

- get to run on desktop
- experiment with running on Pi with coral

- OpenCV with NEON and VFPv4: Ensure your OpenCV build is optimized for ARM processors, including NEON and VFPv4 extensions, which can accelerate image processing on the Raspberry Pi ???

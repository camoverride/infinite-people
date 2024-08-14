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
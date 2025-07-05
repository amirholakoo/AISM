

`http://github.com/bluenviron/mediamtx`

## Go to Release page and look for arm64 version

`https://github.com/bluenviron/mediamtx/releases`

## Copy link

`mkdir mediamtx`

`cd mediamtx`

`wget https://github.com/bluenviron/mediamtx/releases/download/v1.12.3/mediamtx_v1.12.3_linux_arm64.tar.gz`

`tar -xvzf mediamtx_v1.12.3_linux_arm64.tar.gz`

`nano mediamtx.yml`

## Paste bash at the end (make sure # comment out all_others: at the end)

```
  cam1:
    runOnInit: bash -c 'rpicam-vid -t 0 --camera 0 --nopreview --codec yuv420 --width 1280 --height 720 --inline --listen -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1280x720 -i /dev/stdin -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH'
    runOnInitRestart: yes
```
Save and Exit: `Ctrl + X, Y, Enter`

## Run Media Server:
`./mediamtx`

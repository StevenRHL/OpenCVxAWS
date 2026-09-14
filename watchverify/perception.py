"""Source-timestamped video and pretrained MediaPipe pose extraction."""
from pathlib import Path
from fractions import Fraction
import hashlib
import os
import tempfile
import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Which landmark pairs are drawn as a limb. Shared so a live preview and a recorded export
# show the same skeleton: an overlay that differs between the two would make the preview
# misleading about what the analysis actually saw.
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]
POSE_COLOUR = (102, 224, 192)


def draw_skeleton(image, tracks, confidence=.5, colour=POSE_COLOUR):
    """Draw tracked bodies onto `image` in place. `tracks` is [(track_id, pose)]."""
    for track_id, pose in tracks:
        for a, b in BONES:
            if min(pose[a, 3], pose[b, 3]) >= confidence:
                cv2.line(image, tuple(pose[a, :2].astype(int)), tuple(pose[b, :2].astype(int)),
                         colour, 2)
        visible = pose[pose[:, 3] >= confidence]
        if len(visible):
            cv2.putText(image, f'Person {track_id}', tuple(visible[0, :2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1, cv2.LINE_AA)
    return image

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def video_info(path):
    with av.open(str(path)) as container:
        if not container.streams.video: raise ValueError('This file has no video stream.')
        s=container.streams.video[0]
        duration=float(s.duration*s.time_base) if s.duration is not None else float(container.duration or 0)/1e6
        rate=float(s.average_rate or 0)
        if not rate or rate>240: raise ValueError('The video frame rate is unavailable or unsupported.')
        return {'width':s.width,'height':s.height,'duration_s':duration,'fps':rate,'frames':s.frames,'codec':s.codec_context.name,'has_audio':bool(container.streams.audio),'time_base':str(s.time_base)}

def frames(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        first=None; previous=-1.
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None: raise ValueError('Video has missing source timestamps. Convert to a timestamped MP4 first.')
            absolute=float(frame.pts*frame.time_base)
            if first is None:first=absolute
            t=absolute-first
            if t<=previous: raise ValueError('Non-increasing video timestamps; convert the file before analysis.')
            previous=t
            yield index,t,frame.to_ndarray(format='bgr24')

class PoseEstimator:
    # Surfaced so callers can record them in a cache identity; changing any of these
    # changes the extracted features.
    DETECTION_CONFIDENCE=.5
    PRESENCE_CONFIDENCE=.5
    TRACKING_CONFIDENCE=.5
    def __init__(self,variant='full',max_people=4,width=640,
                 detection_confidence=DETECTION_CONFIDENCE,
                 presence_confidence=PRESENCE_CONFIDENCE,
                 tracking_confidence=TRACKING_CONFIDENCE):
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'watchverify-mpl'))
        import mediapipe as mp
        self.mp=mp; self.width=width; self.variant=variant; self.max_people=max_people
        self.detection_confidence=detection_confidence
        self.presence_confidence=presence_confidence
        self.tracking_confidence=tracking_confidence
        asset=ROOT/'models'/f'pose_landmarker_{variant}.task'
        if not asset.exists():raise FileNotFoundError(f'The pretrained pose model {asset.name} is not installed. Run "Setup WatchVerify.command", or fetch it directly with: .venv/bin/python scripts/acquire_pose_assets.py')
        options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=max_people,min_pose_detection_confidence=detection_confidence,min_pose_presence_confidence=presence_confidence,min_tracking_confidence=tracking_confidence,output_segmentation_masks=False)
        self.model=mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.asset_sha256=sha256(asset)
        self.previous=-1
    def detect(self,bgr,t):
        h,w=bgr.shape[:2]
        scale=min(1,self.width/w)
        if scale<1:bgr=cv2.resize(bgr,(int(w*scale),int(h*scale)))
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        ms=max(self.previous+1,int(round(t*1000))); self.previous=ms
        result=self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),ms)
        return [np.array([[lm.x*w,lm.y*h,lm.z,min(lm.visibility,lm.presence)] for lm in person],dtype=np.float64) for person in result.pose_landmarks]
    def close(self):self.model.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

TIME_BASE=Fraction(1,90000)

class VideoExport:
    """Encodes at a 90 kHz timebase so sub-frame-interval source spacing survives.

    `rate` is only a framerate hint. The codec context timebase must be set
    explicitly: PyAV otherwise derives it from `rate`, which quantises any two
    frames closer than 1/rate onto the same pts and makes the muxer reject the
    stream. Real source recordings do carry such spacing (UR Fall adl-04..adl-10
    are sampled 15 ms apart).
    """
    def __init__(self,path,width,height,rate):
        self.container=av.open(str(path),mode='w')
        self.stream=self.container.add_stream('libx264',rate=Fraction(rate).limit_denominator(1001))
        self.stream.width=width//2*2;self.stream.height=height//2*2
        self.stream.pix_fmt='yuv420p';self.stream.options={'crf':'23','preset':'veryfast'}
        self.stream.time_base=TIME_BASE
        self.stream.codec_context.time_base=TIME_BASE
        self._last_pts=None
    def write(self,bgr,t):
        if bgr.shape[1]!=self.stream.width or bgr.shape[0]!=self.stream.height:
            bgr=cv2.resize(bgr,(self.stream.width,self.stream.height))
        pts=round(t/TIME_BASE)
        if self._last_pts is not None and pts<=self._last_pts:
            raise ValueError(f'Output timestamp {t:.6f}s is not after the previous frame '
                             f'({self._last_pts*float(TIME_BASE):.6f}s) at a '
                             f'{float(1/TIME_BASE):.0f} Hz timebase.')
        self._last_pts=pts
        frame=av.VideoFrame.from_ndarray(bgr,format='bgr24');frame.pts=pts;frame.time_base=TIME_BASE
        for packet in self.stream.encode(frame):self.container.mux(packet)
    def close(self):
        for packet in self.stream.encode():self.container.mux(packet)
        self.container.close()

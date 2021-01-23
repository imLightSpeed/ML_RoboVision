import argparse
import time

import cv2
from cv2 import TrackerMIL_create
import imutils
from imutils.video import VideoStream, FPS

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="mil", help="object tracker type")
args = vars(ap.parse_args())

trackersCV = {
    # "csrt": cv2.TrackerCSRT_create,
    # "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    "mil": TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}

tracker = trackersCV[args["tracker"]]()
initBB = None
if not args.get('video', False):
    print("starting video stream")
    vs = VideoStream(src=0).start()
    time.sleep(1)
else:
    vs = cv2.VideoCapture(args["video"])
fps = None
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fps.update()
            fps.stop()
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()

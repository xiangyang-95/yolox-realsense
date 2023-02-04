import cv2
import sys
import time
import numpy as np
import pyrealsense2 as rs
from openvino.inference_engine import IECore

from utils.postprocess import *
from utils.classes import COCO_CLASSES

# Model config
MODEL_PERCISION = 'FP16'
MODEL_PATH = f'./model/public/yolox-tiny/{MODEL_PERCISION}/yolox-tiny.xml'
MODEL_DEVICE = 'GPU'
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.1

# Preprocessing pipeline for YOLOX
def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


# Read network
ie = IECore()
print(f'Reading model from {MODEL_PATH}')
net = ie.read_network(model=MODEL_PATH)
if len(net.input_info) != 1:
    print(f'Sample supports only single input topologies')
    sys.exit(1)
if len(net.outputs) != 1:
    print(f'Sample supports only single output topologies')
    sys.exit(1)

print(f'Configuring input and output blobs')
input_blob = next(iter(net.input_info.keys()))
out_blob = next(iter(net.outputs))

print(f'Setting input and output precision to {MODEL_PERCISION}')
net.input_info[input_blob].precision = 'FP32'
net.outputs[out_blob].precision = MODEL_PERCISION

num_of_classes = max(net.outputs[out_blob].shape)

print(f'Loading the model to the plugin')
exec_net = ie.load_network(network=net, device_name=MODEL_DEVICE)
_, _, h, w = net.input_info[input_blob].input_data.shape

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print(f'Starting inference in synchronous mode')

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # Inference
        preproc_start = time.time()
        image, ratio = preproc(color_image, (h, w))
        image = np.expand_dims(image, axis=0)
        preproc_stop = time.time()
        # print(f'Preprocess time: {preproc_stop - preproc_start:.2f} secs')

        inference_start = time.time()
        res = exec_net.infer(inputs={input_blob: image})
        inference_end = time.time()
        # print(f'Inference time: {inference_end - inference_start:.2f} secs')

        postproc_start = time.time()
        res = res[out_blob]
        predictions = demo_postprocess(res, (h, w))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=NMS_THRESHOLD, score_thr=0.1)
        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
            color_image = vis(color_image, final_boxes, final_scores, final_cls_inds,
                            conf=SCORE_THRESHOLD, class_names=COCO_CLASSES)
        postproc_end = time.time()
        # print(f'Postprocessing time: {postproc_end - postproc_start:.2f} secs')

        processing_stat = {
            'preproc_time': f'{preproc_stop - preproc_start:.4f} secs',
            'inference_time': f'{inference_end - inference_start:.4f} secs',
            'postproc_time': f'{postproc_end - postproc_start:.4f} secs',
        }
        color_image = show_statistics(color_image, processing_stat)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(
                depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

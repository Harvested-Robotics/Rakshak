import cv2
import numpy as np
from arena_api import enums
from arena_api.buffer import BufferFactory
from arena_api.system import system
from ultralytics import YOLO

import math
import threading
import time

import torch
import pyfirmata

# torch.cuda.set_device(0) # Set to your desired GPU number


window_width = 800
window_height = 600

# object classes
classNames = ["laser"]

model = YOLO('/home/harvestedlabs/Downloads/tracking-algorithm-main/Tracking/YOLO/newerbest.pt')

def select_device_from_user_input():
    device_infos = system.device_infos
    if len(device_infos) == 0:
        print("No camera connected\nPress enter to search again")
        input()
    print("Devices found:")
    selected_index = 0
    for i in range(len(device_infos)):
        if device_infos[i]['serial'] == "222600043":
            selected_index = i

    selected_model = device_infos[selected_index]['model']
    print(f"\nCreate device: {selected_model}...")
    device = system.create_device(device_infos=device_infos[selected_index])[0]

    return device


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_height_width(topLeft, topRight, bottomRight, bottomLeft):
    height = distance(topLeft, bottomLeft)
    width = distance(topLeft, topRight)
    return width, height


def aruco_display(image):
    results = model(image, conf=0.50)  # predict on an image
    # Extract bounding boxes and keypoints if available
    markers = []
    if results:
        # Assuming results is a list of detection results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            # Check if keypoints are present in the results
            if hasattr(result, 'kp'):
                keypoints = result.kp.cpu().numpy()
                # Draw keypoints on the image
                for kpt in keypoints:
                    x, y = map(int, kpt)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Draw bounding boxes on the image
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                markers.append(((x1+x2)/2, (y1+y2)/2))

    return image, markers


def nearPoint(p1, p2):
    if p2 is None:
        return False
    logic = distance(p1, p2) < 10
    return logic

def minCostConnectPoints(points, lp):
    path = []
    done_set = set()
    while len(done_set) != len(points):
        min_d, u = float('inf'), 0
        for i, p in enumerate(points):
            if p in done_set:
                continue
            dist = abs(p[0] - lp[0]) + abs(p[1] - lp[1])
            if dist < min_d:
                u = i
                min_d = dist
        lp = points[u]
        done_set.add(lp)
        path.append(u)
    return path


def get_image(device):
    image_buffer = device.get_buffer()  # optional args
    nparray = np.ctypeslib.as_array(image_buffer.pdata, shape=(image_buffer.height, image_buffer.width, int(
        image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width,
                                                   int(image_buffer.bits_per_pixel / 8))

    display_img = cv2.cvtColor(nparray, cv2.COLOR_BayerBG2BGR)
    device.requeue_buffer(image_buffer)
    return display_img


gamma = 0.1
gamma_table = np.array(
    [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)


def apply_gamma_correction(frame, gamma):
    return cv2.LUT(frame, gamma_table)


def detect_laser(frame):
    target_blue = np.array([0, 0, 255], dtype=np.uint8)
    # Convert BGR to LAB
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    L, A, B = cv2.split(lab_frame)

    # Find the coordinates of the brightest point
    brightest_coords = np.unravel_index(np.argmax(L), L.shape)
    x, y = brightest_coords[1], brightest_coords[0]

    # Draw a circle around the brightest point
    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
    return x, y


def move_galvo(lx, ly, cX, cY, xDir, yDir, xStep, yStep, dir=1):
    if abs(lx - cX) > 2:
        if lx < cX:
            xDir.write(dir)
            dirVal = 0
        else:
            xDir.write(1 - dir)
            dirVal = 1

        step_count = 2
        if abs(lx - cX) < 20:
            step_count = 1
        if abs(lx - cX) < 50:
            step_count = 2
        elif abs(lx - cX) < 100:
            step_count = 5
        else:
            step_count = 10
        for _ in range(step_count):
            xStep.write(1)
            time.sleep(1 / 10000.0)
            xStep.write(0)

    if abs(ly - cY) > 2:
        if ly < cY:
            yDir.write(dir)
            dirVal = 0
        else:
            yDir.write(1 - dir)
            dirVal = 1
        step_count = 2
        if abs(ly - cY) < 20:
            step_count = 1
        if abs(ly - cY) < 50:
            step_count = 2
        elif abs(ly - cY) < 100:
            step_count = 5
        else:
            step_count = 10
        for _ in range(step_count):
            yStep.write(1)
            time.sleep(1 / 10000.0)
            yStep.write(0)


def target(device, markers, xDir, xStep, yDir, yStep, laser, index=0):
    frame = get_image(device)
    prev_frame = frame
    # midpoint = len(frame[0]) // 1
    # if index == 0:
    #     frame = frame[:, :midpoint]
    # else:
    #     frame = frame[:, midpoint:]
    laser_pos = detect_laser(apply_gamma_correction(frame, 2.0))
    # if index != 0:
    #     laser_pos = (laser_pos[0] + midpoint, laser_pos[1])
    idxes = minCostConnectPoints(markers, laser_pos)
    markers = [markers[i] for i in idxes]
    laser.write(0.1)
    for marker in markers:
        cX, cY = marker
        while not nearPoint(marker, laser_pos):
            laser.write(0.3)
            frame = get_image(device)
            if is_moving(frame, prev_frame):
                return
            prev_frame = frame
            # if index == 0:
            #     frame = frame[:, :len(frame[0]) // 1]
            # else:
            #     frame = frame[:, len(frame[0]) // 1:]
            laser_pos = detect_laser(apply_gamma_correction(frame, 2.0))
            # if index != 0:
            #     laser_pos = (laser_pos[0] + midpoint, laser_pos[1])
            if laser_pos is None:
                frame = get_image(device)
                # if index == 0:
                #     frame = frame[:, :len(frame[0]) // 1]
                # else:
                #     frame = frame[:, len(frame[0]) // 1:]
                laser_pos = detect_laser(
                    apply_gamma_correction(frame, 2.0))
                # if index != 0:
                #     laser_pos = (laser_pos[0] + midpoint, laser_pos[1])
                if laser_pos is not None:
                    break
                continue
            lx, ly = laser_pos

            move_galvo(lx, ly, cX, cY, xDir, yDir, xStep, yStep, dir=0)
            # move_galvo(lx, ly, cX, cY, xDir2, yDir2, xStep2, yStep2, dir=1)

            frame = get_image(device)
            # if index == 0:
            #     frame = frame[:, :len(frame[0]) // 1]
            # else:
            #     frame = frame[:, len(frame[0]) // 1:]
            laser_pos = detect_laser(apply_gamma_correction(frame, 2.0))
            # if index != 0:
            #     laser_pos = (laser_pos[0] + midpoint, laser_pos[1])

            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Image", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        print(f"Laser on at {marker}")
        laser.write(1)
        time.sleep(2)
        print("Laser off")


def is_moving(frame1, frame2):
    diff = np.sqrt(np.mean(np.square(frame1 - frame2)))
    return diff > 100


if __name__ == "__main__":
    # board1 = pyfirmata.Arduino("/dev/ttyUSB0")
    board2 = pyfirmata.Arduino("/dev/ttyUSB0")

    # Galvo 1
    # xDir1 = board1.digital[4]
    # xStep1 = board1.digital[3]
    # yDir1 = board1.digital[6]
    # yStep1 = board1.digital[5]
    # laser1 = board1.get_pin('d:9:p')

    # Galvo 2
    xDir2 = board2.digital[4]
    xStep2 = board2.digital[3]
    yDir2 = board2.digital[6]
    yStep2 = board2.digital[5]
    laser2 = board2.get_pin('d:9:p')

    # laser1.write(0.1)
    laser2.write(0.1)

    done_markers = set()
    done = False
    device = select_device_from_user_input()

    device.tl_stream_nodemap.get_node(
        'StreamBufferHandlingMode').value = 'NewestOnly'
    device.tl_stream_nodemap.get_node('StreamPacketResendEnable').value = True
    device.tl_stream_nodemap.get_node(
        'StreamAutoNegotiatePacketSize').value = True

    isp_bayer_pattern = device.nodemap.get_node('IspBayerPattern').value
    is_color_camera = False

    device.nodemap.get_node('Width').value = 3072
    device.nodemap.get_node('Height').value = 2048

    if isp_bayer_pattern != 'NONE':
        is_color_camera = True

    if is_color_camera == True:
        device.nodemap.get_node('PixelFormat').value = "BayerRG8"
    else:
        device.nodemap.get_node('PixelFormat').value = "Mono8"

    device.nodemap.get_node('DeviceStreamChannelPacketSize').value = 1500
    device.nodemap.get_node('AcquisitionMode').value = "Continuous"
    device.nodemap.get_node('AcquisitionFrameRateEnable').value = True
    device.nodemap.get_node('AcquisitionFrameRate').value = 16.8
    device.nodemap.get_node('AcquisitionFrameRateEnable').value = True

    device.nodemap['ColorTransformationEnable'].value = False
    device.nodemap['BalanceWhiteEnable'].value = True
    device.nodemap['GammaEnable'].value = True

    device.nodemap['Gamma'].value = 0.30

    device.nodemap['ColorTransformationEnable'].value = True
    device.nodemap['BalanceWhiteEnable'].value = True

    device.nodemap['TriggerSelector'].value = "AcquisitionStart"

    key = -1
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    device.start_stream()

    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_counter = 0

    doneset = set()
    prev_frame = get_image(device)
    time.sleep(1)
    while True:
        frame = get_image(device)
        while is_moving(prev_frame, frame):
            time.sleep(1)
            prev_frame = frame
            frame = get_image(device)

        arucoimage, markers = aruco_display(frame)
        midpoint = len(frame[0]) // 2
        assign_markers = {0: [], 1: []}
        for marker in markers:
            if (round(marker[0] / 100), round(marker[1] / 100)) in doneset:
                continue
            assign_markers[0].append(marker)

        marker_total_count = len(assign_markers[0]) + len(assign_markers[1])
        print(f"Total Markers: {marker_total_count}")
        if marker_total_count > 0:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"./detected_logs/{timestr}.jpg", arucoimage)

        t1 = threading.Thread(target(device, assign_markers[0], xDir2, xStep2, yDir2, yStep2, laser2, index=0))

        t1.start()
        laser2.write(0.1)

        # frame2 = get_image(device)

        # while t1.is_alive():
        #     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        #     frame2 = get_image(device)
        #     cv2.imshow(f"Image", frame2)

        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         break

        t1.join()
        [doneset.add((round(m[0] / 100), round(m[1] / 100))) for m in markers]
        print('done')
        frame = get_image(device)
        while is_moving(prev_frame, frame):
            time.sleep(1)
            prev_frame = frame
            frame = get_image(device)
            doneset = set()
        # time.sleep(0.6)

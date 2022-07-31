import cv2
import numpy as np
import mediapipe as mp
from collections import deque

#  for handling color points
blue = [deque(maxlen=1024)]
green = [deque(maxlen=1024)]
red = [deque(maxlen=1024)]
yellow = [deque(maxlen=1024)]

# array pointers
b_index = 0
g_index = 0
r_index = 0
y_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # in rgb format
index = 0  # pointer for colors array

# white canvas setup
canvas = np.zeros((471, 636, 3)) + 255  # size is 471X636, 255 for white background
canvas = cv2.rectangle(canvas, (40, 1), (140, 65), (0, 0, 0), 2)
canvas = cv2.rectangle(canvas, (160, 1), (255, 65), (255, 0, 0), 2)  # blue box
canvas = cv2.rectangle(canvas, (275, 1), (370, 65), (0, 255, 0), 2)  # green box
canvas = cv2.rectangle(canvas, (390, 1), (485, 65), (0, 0, 255), 2)  # red box
canvas = cv2.rectangle(canvas, (505, 1), (600, 65), (0, 255, 255), 2)  # yellow box

# label the canvas boxes
cv2.putText(canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Air Canvas', cv2.WINDOW_AUTOSIZE)

# initialising mediapipe
pipeHand = mp.solutions.hands
hand = pipeHand.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils  # to draw landmarks on detected hand

# initialise webcam
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
while ret:
    ret, frame = cam.read()  # read each frame
    x, y, c = frame.shape

    # flip the frame vertically
    frame = cv2.flip(frame, 1)
    # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # hsv format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # rgb format used for processing hands

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    result = hand.process(frame_rgb)  # landmark prediction result
    if result.multi_hand_landmarks:
        landmarks = []
        for handmark in result.multi_hand_landmarks:
            for l in handmark.landmark:
                print(l.x)
                print(l.y)
                # after adjusting to frame size
                lx = int(l.x * 640)
                ly = int(l.y * 480)

                landmarks.append([lx, ly])

            # drawing landmarks on frames
            mp_draw.draw_landmarks(frame, handmark, pipeHand.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)
        print(fore_finger[1] - thumb[1])
        if thumb[1] - fore_finger[1] < 30:  # do not draw when forefinger next to thumb
            blue.append(deque(maxlen=512))
            b_index += 1
            green.append(deque(maxlen=512))
            g_index += 1
            red.append(deque(maxlen=512))
            r_index += 1
            yellow.append(deque(maxlen=512))
            y_index += 1
        elif fore_finger[1] <= 65:  # finger pointed upwards
            if 40 <= fore_finger[0] <= 140:  # clear button
                """
                do not draw for any color, clear the deque and set indexes to 0
                """

                blue = [deque(maxlen=512)]
                green = [deque(maxlen=512)]
                red = [deque(maxlen=512)]
                yellow = [deque(maxlen=512)]

                b_index = 0
                g_index = 0
                r_index = 0
                y_index = 0

                canvas[67:, :, :] = 255
            elif 160 <= fore_finger[0] <= 255:  # blue color
                index = 0
            elif 275 <= fore_finger[0] <= 370:  # green color
                index = 1
            elif 390 <= fore_finger[0] <= 485:  # red color
                index = 2
            elif 505 <= fore_finger[0] <= 600:  # yellow color
                index = 3
        else:  # drawing condition
            if index == 0:
                blue[b_index].appendleft(fore_finger)
            elif index == 1:
                green[g_index].appendleft(fore_finger)
            elif index == 2:
                red[r_index].appendleft(fore_finger)
            elif index == 3:
                yellow[y_index].appendleft(fore_finger)
    else:
        blue.append(deque(maxlen=512))
        b_index += 1
        green.append(deque(maxlen=512))
        g_index += 1
        red.append(deque(maxlen=512))
        r_index += 1
        yellow.append(deque(maxlen=512))
        y_index += 1

    points = [blue, green, red, yellow]  # draw lines of all colors on the canvas and frame

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                cv2.line(canvas, points[i][j][k-1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()  # close the webcam
cv2.destroyAllWindows()  # close all active windows

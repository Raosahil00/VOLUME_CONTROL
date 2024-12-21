import cv2
import mediapipe as mp
import numpy as np
import os
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to set the volume (Windows example using NirCmd)
def set_volume(volume):
    # Volume should be between 0.0 and 1.0
    volume = max(0.0, min(volume, 1.0))
    print("Setting volume to:", int(volume * 100))  # Debugging line
    # Use a command to set the volume (this may vary based on your OS)
    os.system(f"nircmd setvolume {int(volume * 100)}")
    return int(volume * 100)  # Return the volume percentage

# Load volume icons
volume_up_icon = cv2.imread('volume_up.png', cv2.IMREAD_UNCHANGED)
volume_down_icon = cv2.imread('volume_down.png', cv2.IMREAD_UNCHANGED)

# Start video capture using the default camera
cap = cv2.VideoCapture(0)  # Changed back to 0 for default laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to reduce the size (e.g., 1280x720)
    frame = cv2.resize(frame, (1280, 720))  # Adjust the size as needed

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Initialize volume variable
    current_volume = 0
    icon_to_display = None

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the coordinates
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Draw a line between thumb and index finger
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)

            # Calculate the distance between thumb and index finger
            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            print("Distance:", distance)  # Debugging line

            # Map the distance to volume level (adjust the divisor based on your needs)
            volume = 1 - min(max(distance / 200, 0), 1)  # Adjust the divisor based on your needs
            current_volume = set_volume(volume)  # Set the volume and get the percentage

            # Determine which icon to display
            if distance < 50:  # Adjust this threshold as needed
                icon_to_display = volume_down_icon
            elif distance > 100:  # Adjust this threshold as needed
                icon_to_display = volume_up_icon

    # Display the current volume on the frame
    cv2.putText(frame, f'Volume: {current_volume}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw the volume bar
    bar_width = 300
    bar_height = 20
    bar_x = 10
    bar_y = 60
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)  # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * (current_volume / 100)), bar_y + bar_height), (0, 255, 0), -1)  # Volume level

    # Overlay the volume icon if it exists
    if icon_to_display is not None:
        icon_height, icon_width = icon_to_display.shape[:2]
        frame[10:10 + icon_height, 10:10 + icon_width] = icon_to_display

    # Display the frame
    cv2.imshow('Volume Control Using Fingers', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# %% [markdown]
# # This is a test file to see if opencv is working and cars are being detected or not

# %%
# https://www.geeksforgeeks.org/set-opencv-anaconda-environment/
# https://medium.com/analytics-vidhya/vehicle-car-detection-in-real-time-and-recorded-videos-in-python-windows-and-macos-c5548b243b18
# https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/
# https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master

# %%
import cv2

# %%
capture_car = cv2.VideoCapture(
    r"C:\Users\RUDRO\OneDrive - The University of Kansas\RUDRO\Study Documents\Fall-2022\EECS 581\Capstoneproject\EECS581-Project\Y2Mate.is - 4K Video of Highway Traffic!-KBsqQez-O4w-480p-1655187609817.mp4")

car_cascade = cv2.CascadeClassifier(
    r"C:\Users\RUDRO\OneDrive - The University of Kansas\RUDRO\Study Documents\Fall-2022\EECS 581\Capstoneproject\EECS581-Project\xml files\cars.xml")
# print(car_cascade)

# %%
# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = capture_car.read()
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    # To draw a rectangle in each cars
    count_cars = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frames, 'Car', (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)
        # Display frames in a window
        cv2.imshow('Car Detection', frames)
        count_cars += 1
    # Wait for Enter key to stop
    print(f"Cars found: {count_cars}")
    if cv2.waitKey(33) == 13:
        break

# %%
capture_car.release()
cv2.destroyAllWindows()

# %%

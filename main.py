import cv2
import time

from collections import namedtuple
from configobj import ConfigObj

"""
Slouchy uses your webcam to check if you're slouching and alert you if you are. 
This project is still in active development and not feature complete.

The main module will return a namedtuple called Maybe.
This is designed to emulate the behavior you see with Maybe/Either in functional languages.

success (Bool)  Indicates whether or not main was able to detect a single face 
                and perform the necessary calculations.

result (Bool/Str) If success is true then result will 
                  indicate slouching (True = Yes, False = No)
                  If success is false it will be a string indicating what went wrong.
"""

# Trying some pseudo-functional programming here.
# Making use of namedtuples throughout this program to simulate a Maybe/Either.
# success is always True or False, and indicates if the requested operation succeeded.
# if success result is the calculation results
# otherwise result is an error message
Maybe = namedtuple('Maybe', ['success','result'])

config                    = ConfigObj('slouchy.ini')
y_ratio_reference         = float(config['MAIN']['y_ratio_reference'])
# allowed_variance        = float(config['MAIN']['allowed_variance'])
cascade_path              = str(config['MAIN']['cascade_path'])
camera_delay              = int(config['MAIN']['camera_delay'])
allowed_forward_variance  = float(config['MAIN']['allowed_forward_variance'])
allowed_backward_variance = float(config['MAIN']['allowed_backward_variance'])

#video_device can be an int or a string, so try int, and if not assume string
try:
  video_device = int(config['MAIN']['video_device'])
except ValueError:
  video_device = str(config['MAIN']['video_device'])

cap      = cv2.VideoCapture(video_device)
camera_x = float(cap.get(3))
camera_y = float(cap.get(4))
cap.release()

# Make a square image so the x and y axes are 1:1 during face detection.
crop_side_by_pixels = (camera_x - camera_y) / 2

# These are used for array slicing, so they must be integers.
crop_x_left  = int(crop_side_by_pixels)
crop_x_right = int(camera_x - crop_side_by_pixels)

print("crop_x_left =", '{:.2f}'.format(crop_x_left))
print("crop_x_right =", '{:.2f}'.format(crop_x_right))

# Calculate y_ratio MaybeFace -> MaybeYRatioSquared
def calculate_y_ratio(MaybeFace):
  if MaybeFace.success:
    (x, y, w, h) = MaybeFace.result
  else:
    return MaybeFace

  # print("x =", '{:d}'.format(x))
  # print("h =", '{:d}'.format(h))
  print("y =", '{:d}'.format(y))
  print("w =", '{:d}'.format(w))
  print("camera_x =", '{:.4f}'.format(camera_x))
  print("camera_y =", '{:.4f}'.format(camera_y))

  # w_ratio = (camera_x - w) / float(camera_x)
  # y_ratio = y / float(camera_y)

  # Adjust y so it and w are 1:1 with each other.
  # y_adjusted = y * (camera_x / camera_y)

  # w gets larger as face gets closer.
  # y gets smaller as face gets closer.
  # Compensating for this.
  # w_adjusted = camera_x - w

  # yw_ratio   = y / float(w)

  # print("y_adjusted =", '{:.0f}'.format(y_adjusted))
  # print("w_adjusted =", '{:.0f}'.format(w_adjusted))
  # print("yw_ratio", '{:.4f}'.format(yw_ratio))

  # TODO: See if this calculation can be improved
  # c_squared = y**2 + (camera_x - w)**2
  
  # y gets smaller as face gets closer.
  # w gets larger as face gets closer.
  # Compensate for this
  c_squared = y**2 + (camera_x - w)**2

  # w_y_ratio = w_ratio / y_ratio
  # print("w/y =", '{:.4f}'.format(w_y_ratio))

  return Maybe(True, c_squared)
  # return Maybe(True, w_y_ratio)
  # return Maybe(True, yw_ratio)

def get_face_width(MaybeFace):
  if MaybeFace.success:
    (x, y, w, h) = MaybeFace.result
  else:
    return MaybeFace

  return Maybe(True, w)  

# Take a picture with the camera. 
# Ideally this is where we always transition from the impure to "pure" calculations.
# video_device -> MaybeImage
def take_picture(video_device):

  cap = cv2.VideoCapture(video_device)
  cap.open(video_device)

  # Added since some cameras need time to warm up before they can take pictures.
  if camera_delay > 0:
    time.sleep(camera_delay)

  ret, image = cap.read()

  if not ret:
    return Maybe(False, "Could not open camera. Please make sure video_device is set correctly.")

  cap.release()

  # Make the image square.
  cropped_image = image[0:camera_y, crop_x_left:crop_x_right]

  return Maybe(True, cropped_image)  

# Detect face in an image. Only ever one face, other numbers are an error.
# MaybeImage -> MaybeFace
def detect_face(MaybeImage):
  if MaybeImage.success:
    image = MaybeImage.result
  else:
    return MaybeImage

  # Make image grayscale for processing
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Create the haar cascade object
  faceCascade = cv2.CascadeClassifier(cascade_path)

  # Detect faces in the image
  # faces will be an iterable object
  faces = faceCascade.detectMultiScale(
      image=gray_image,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(40, 40),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )

  # We assume the largest face (at index zero) is the face we're interested in
  try:
    face = faces[0]
    return Maybe(True, face)
  except IndexError:
    return Maybe(False, "No faces detected. This may be due to low / uneven lighting, or your particular model of camera. Check slouchy.ini for possible fixes.")

# Detect if person is slouching 
# MaybeFace -> MaybeSlouching
def detect_slouching(MaybeFace):

  if MaybeFace.success:
    MaybeYRatioSquared = calculate_y_ratio(MaybeFace)
  else:
    return MaybeFace

  if MaybeYRatioSquared.success:
    current = MaybeYRatioSquared.result
  else:
    return MaybeYRatioSquared

  # print("y^2 + w^2 =", '{:f}'.format(c_current))
  # print("Current posture / w_y_ratio_reference:", '{:f}'
  #       .format(float(c_current) / w_y_ratio_reference))
  # print("Current posture * allowed_variance:", '{:f}'
  #   .format(float(c_current * allowed_variance)))

  # y_min = y_ratio_reference * (1.0 - allowed_variance)
  # y_max = y_ratio_reference * (1.0 + allowed_variance)

  # Allowed variances. 
  # y_ratio increases as you slouch forward, decreases as you slouch backwards
  backward_variance = y_ratio_reference * (1.0 - allowed_backward_variance)
  forward_variance  = y_ratio_reference * (1.0 + allowed_forward_variance)

  print("backward_variance", '{:.4f}'.format(backward_variance))
  print("current", '{:.4f}'.format(current))
  print("forward_variance", '{:.4f}'.format(forward_variance))

  # if y_current <= y_max:
  # if y_current >= y_min:
  if backward_variance <= current <= forward_variance:
    slouching = False
  else:
    slouching = True

  print("Slouching:", slouching)
  return Maybe(True, slouching)

def main():
  maybe_image = take_picture(video_device)
  maybe_face = detect_face(maybe_image)
  maybe_slouching = detect_slouching(maybe_face)

  return maybe_slouching

if __name__ == '__main__':
  main()  
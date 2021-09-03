import cv2 

capture = cv2.VideoCapture( 0 ) 

rect, frame = capture.read( )

print( frame.shape )
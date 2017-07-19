import gige
import sys
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


d = gige.setup()
if d is None:
	print "Nothin to do."
	sys.exit()
n_frames = 1000
while True:
	data = gige.getFrame(d, 3)
	rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
	rgb[1::2, 0::2, 0] = data[1::2, 0::2] # Red
	rgb[0::2, 0::2, 1] = data[0::2, 0::2] # Green
	rgb[1::2, 1::2, 1] = data[1::2, 1::2] # Green
	rgb[0::2, 1::2, 2] = data[0::2, 1::2] # Blue

	cv2.imshow("Data", rgb)
	if cv2.waitKey(1) == 27: 
		break  # esc to quit
cv2.destroyAllWindows()
sys.exit()
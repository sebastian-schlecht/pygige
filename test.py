import gige
import sys
import cv2
d = gige.setup()
while d:
	data = gige.getFrame(d, 3)
	cv2.imshow("Data", data[::4,::4])
	if cv2.waitKey(1) == 27: 
		break  # esc to quit
cv2.destroyAllWindows()
print "Exiting"
sys.exit()
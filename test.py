import gige
import sys

d = gige.setup()
if d:
	data = gige.getFrame(d)
	print data
else:
	print "meh"

sys.exit()
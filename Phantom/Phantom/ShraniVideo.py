import cv2 as cv
import os.path

class ShraniVideo:
	
	def __init__(self, Ime = "Brez Imena", FPS = 30):
		self.pot = os.getcwd()
		st = 0
		while os.path.isfile(self.pot + "\\" + Ime + (str(st) if st else "") + ".avi"):
			st += 1
		self.Ime = Ime + (str(st) if st else "") + ".avi"
		self.FPS = FPS
		self.out = None
	
	def DodajFrame(self, frame):
		if self.out is None:
			self.out = cv.VideoWriter(self.Ime, cv.VideoWriter_fourcc('M','J','P','G'), self.FPS, (frame.shape[1], frame.shape[0]))
		self.out.write(frame)

	def Koncal(self):
		if self.out is not None:
			self.out.release()
			self.out = None
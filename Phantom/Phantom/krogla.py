import cv2 as cv
import numpy as np

# Ali rise maske, tocke, itd.?
debug = False

def vrniKroglo(slika, elipsa, izpisujOpozorila = False):
	"""
	slika - rabi sliko na kateri isce
	elipsa - kje naj isce
	izpisujOpozorila - ali izpisuje opozorila v cmd
	Vrne tocko kje se nahaja krogla
	"""

	slika = slika.copy()

	if debug: slikaDebug = slika.copy()

	# Maskiram po plosci
	maska = cv.ellipse(np.zeros(slika.shape, dtype = np.uint8), elipsa[0], elipsa[1], elipsa[2], 0, 360, (1, 1, 1), -1)
	slika = np.where(maska == (0, 0, 0), (255, 255, 255), slika).astype(np.uint8)

	# Pretvorim v hsv prostor
	slika = cv.cvtColor(slika, cv.COLOR_BGR2HSV)

	# Maskiram po barvi zogice
	slika = cv.inRange(slika, (0, 0, 0), (180, 60, 110))
	if debug: cv.imshow("krogla maska1", slika)

	# Odstrani majhne tocke
	slika = cv.medianBlur(slika, 3)

	# Zapolni zogo
	kernel = np.ones((3, 3), np.uint8)
	slika = cv.dilate(slika, kernel)
	slika = cv.erode(slika, kernel)

	# Odstrani majhne tocke
	slika = cv.medianBlur(slika, 5)

	if debug: cv.imshow("krogla maska2", slika)

	# Najde obrobe
	obroba = cv.findContours(slika, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	
	# Ce ni nasel obrob
	if obroba is None:
		if izpisujOpozorila: print("Ne najdem robov")
		return None, None

	# Razvrstim obrobe po povrsini
	obrobe = []
	for o in obroba[0]:
		obrobe.append({"o": o, "povrsina": cv.contourArea(o)})
	obrobe = sorted(obrobe, key = lambda o: o["povrsina"], reverse = True)

	for o in obrobe:
		# Povrsina obrobe more vstrezat
		""" Te meje bi lahko določil z razmerjem vrhov.. če so roboti dvignjeni so blizje kameri in zato je zogica vecja """
		if o["povrsina"] < 270 or o["povrsina"] > 970:
			continue

		# Ce imam manj kot 5 tock ne morem dolocit elipse
		if o["o"].shape[0] < 5:
			continue

		# Dolocim elipso na dane tocke obrobe
		tocka, (MA, ma), kot = cv.fitEllipseDirect(o["o"])
		(MA, ma) = (MA / 2, ma / 2)

		if debug: cv.drawContours(slikaDebug, o["o"], -1, (0, 100, 255))
		if debug: cv.ellipse(slikaDebug, (int(tocka[0]), int(tocka[1])), (int(MA), int(ma)), int(kot), 0, 360, (255, 255, 255))
		if debug: cv.imshow("krogla slika", slikaDebug)

		# Ali je priblizno krog
		avg = np.average((MA, ma))
		vecji = np.max((MA, ma))
		manjsi = np.max((MA, ma))
		if manjsi  / vecji < 0.5:
			continue

		# Ali je primerno velika
		razmerje = np.max(elipsa[1]) / np.max((MA, ma))
		razmerjeRef = 30 / 2.3
		toleranca = 0.2
		if not (razmerje > razmerjeRef * (1 - toleranca) and razmerje < razmerjeRef * (1 + toleranca)):
			continue

		# Vrne zaokrozeno
		return (int(tocka[0]), int(tocka[1])), int(avg)

	return None, None


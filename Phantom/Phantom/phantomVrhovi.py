import cv2 as cv
import numpy as np
import operator
from itertools import combinations

# Ali rise maske, tocke, itd.?
debug = False

# Hrani zadnjo sredino plosce
sredinaPrej = None

def dolociPrimerneTocke(tocke, sredina):
	""" Tocke, ki so skupaj zdruzi v tisto, ki je najblizje sredini """
	
	# Ce ni vsaj treh tock
	if len(tocke) <= 3:
		return tocke

	# Naredi vse mozne kombinacije tock
	kombinacije = list(combinations(tocke, 3))
			
	# Izracunam napake za vsako kombinacijo
	napake = []
	for k in kombinacije:
		lk = list(k)
		n = napakaTrikotnika(lk) + napakaVelikosti(lk) + napakaSredine(lk, sredina)
		napake.append({"n": n, "k": k})

	# Razvrsti po napakah
	napake = sorted(napake, key = lambda n: n["n"])

	# Vrnem najboljso kombinacijo
	return napake[0]["k"]

def napakaSredine(tocka, sredina):
	""" Doloci napako, ki pride iz razlike med parametrom sredina in sredino tock """
	sredinaTock = np.average(tocka, axis = 0)
	razlika = np.linalg.norm(sredinaTock - sredina)
	napaka = 2.0e-05 * razlika ** 2 + 0.001 * razlika + 1.38777878078145e-17
	return napaka

def napakaVelikosti(tocke):
	""" Doloci napako glede na velikost (povrsino) trikotnika ki ga tvorijo tocke """
	povrsina = np.linalg.norm(np.cross(tocke[0] - tocke[1], tocke[1] - tocke[2])) / 2
	napaka = 1.39048172382203e-09 * povrsina ** 2 - 0.000103686517878069 * povrsina + 1.85257312695063
	return napaka if napaka > 0 else 0

def napakaTrikotnika(tocke):
	""" Izracuna kako dobro se tocke prilegajo trkotniku """
	v1 = np.sum((tocke[0] - tocke[1]) ** 2)
	v2 = np.sum((tocke[1] - tocke[2]) ** 2)
	v3 = np.sum((tocke[2] - tocke[0]) ** 2)

	v = np.array((v1, v2, v3))
	std = np.std(v)
	avg = np.average(v)

	if avg < 0.1 and avg > -0.1: return np.inf
	return std / avg

def tvorijoTrikotnik(tocke):
	""" Vrne True ce tocke tvorijo trikotnik (s toleranco) """
	return napakaTrikotnika(tocke) < 0.25

def najdiVrhe(slika, izpisujOpozorila = False):
	"""
	Slika je slika na kateri isce
	Ce je izpisuj obvestila True bo v konzolo printal zakaj ni nasel vrhov
	Vrne tocke vrhovov robotov ali None
	"""

	slika = slika.copy()
	
	# Filtrira barvo
	hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
	maska = cv.inRange(hsv, (5, 140, 90), (17, 255, 180))
	if debug: cv.imshow("vrhovi maska1", maska)
	
	# Odstranim sum
	maska = cv.medianBlur(maska, 5)	

	# Zapiranje
	kernel = np.ones((5, 5), np.uint8)
	maska = cv.dilate(maska, kernel, iterations = 3)
	maska = cv.erode(maska, kernel, iterations = 3)
	if debug: cv.imshow("vrhovi maska2", maska)

	# Za iskanje tock najblizje sredini
	global sredinaPrej
	sredinaSlike = None
	if sredinaPrej is None:
		sredinaSlike = np.flip(np.floor(np.array(maska.shape) / 2))
	else:
		sredinaSlike = sredinaPrej
	if debug: cv.circle(slika, tuple(sredinaSlike.astype(np.int)), 5, (255, 255, 0), -1)

	# Najde obrobe
	obroba = cv.findContours(maska, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	# Ce ni nasel obrob
	if obroba is None:
		if izpisujOpozorila: print("Ne najdem robov")
		return None

	# Najblizja tocka vsake obrobe sredini slike
	vrhiRobotov = [] 

	for o in obroba[0]:
		# Ce obroba ni prave dimenzije
		povrsina = cv.contourArea(o)
		if povrsina < 70 or povrsina > 1400:
			continue

		# Iz vsake obrobe najde tocko najblizje sredini slike
		najblizja = np.array((np.inf, None))
		for point in o:
			p = point[0]
			raz = np.sqrt(np.sum((sredinaSlike - p) ** 2))
			if raz < najblizja[0]:
				najblizja = np.array((raz, p))

		# Shrani najblizjo tocko vsake obrobe
		vrhiRobotov.append(najblizja[1])

	if debug: 
		for v in vrhiRobotov:
			cv.circle(slika, tuple(v), 5, (255, 255, 255), -1)

	le = len(vrhiRobotov)
	# Ce ni nasel treh vrhov
	if le < 3:
		if izpisujOpozorila: print("Nisem nasel treh tock")
		return None

	# Ce najde vec kot 3 tocke poskusi dolociti prave
	vrhiRobotov = dolociPrimerneTocke(vrhiRobotov, sredinaSlike)	

	if debug: 
		for v in vrhiRobotov:
			cv.circle(slika, tuple(v), 3, (255, 0, 255), -1)
		cv.imshow("vrhovi", slika)

	# Ce so dokaj smiselno postavleni
	if not tvorijoTrikotnik(vrhiRobotov):
		if izpisujOpozorila: print("Ne tvorijo trikotnika")
		return None

	# Shrani sredino plosce
	sredinaPrej = np.average(vrhiRobotov, axis = 0)

	# Vrne vrhove
	return np.array(vrhiRobotov)

def kotMedVektorji(v1, v2):
	"""
	Izracuna kot med dvema vektorjema
	Vrne radiane med -pi, +pi
	"""
	return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

def elipsa(tocke, slika, izpisujOpozorila = False):
	"""
	Tocke je vektor iz treh tock za elipso
	Vrne parametre elipse
	"""
	slika = slika.copy()

	c = np.average(tocke, axis = 0)

	nove = list(tocke)
	for t in tocke:
		nove.append(list(c - np.array(t) + c))
		#cv.drawMarker(slika, tuple(np.array((c - np.array(t) + c), dtype = np.int)), (255, 200, 200), cv.MARKER_TILTED_CROSS, 15)
	(x, y), (MA, ma), kot = cv.fitEllipseDirect(np.array(nove).astype(np.int32))
	#MA, ma = int(MA * 1.05), int(ma * 1.05) # Da malo poveča elipso
	#E1, E2, E3 = (int(x), int(y)), (int(MA / 2), int(ma / 2)), int(kot)
	E1, E2, E3 = np.array([x, y]), np.array([MA / 2, ma / 2]), np.array(kot)

	# region Boljše prileganje 2
	hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)

	nove = []
	preveriNajvec = 20 # Koliko tock naj najvec preveri predno obupa
	kotZacetni = kotMedVektorji(np.array([1, 0]), tocke[0] - E1)
	kotPremika = 2 * np.pi / preveriNajvec

	# Zmesa seznam (zmesa vrsti red po katerem bo iskal robove)
	vsi = np.arange(preveriNajvec)
	np.random.shuffle(vsi)

	for i in vsi:
		# Kot pod katerem trenutno iscem
		kot = kotPremika * i + kotZacetni
		# Enotski vektor v kateri smeri trenutno iscem
		v = np.array([np.cos(kot), np.sin(kot)]) 

		# Preveri linijo v katero kaze vektor v
		for s in range(int(np.min(E2) * 0.9), int(np.max(E2) * 1.1), 1):
			# Pixel, ki ga pregleduje
			pix = np.round(v * s + E1).astype(np.uint)

			## Ce je pixel preblizu tocke vrha odnehaj
			#for t in tocke:
			#	if np.linalg.norm(pix - t) < 10:
			#		break
			#else:

			# Ce je pixel izven meja slike odnehaj
			if pix[0] < 0 or pix[0] > hsv.shape[1] - 1 or pix[1] < 0 or pix[1] > hsv.shape[0] - 1:
				break

			if debug: cv.drawMarker(slika, tuple(pix), (0, 100, 50), cv.MARKER_CROSS, 1)
			
			# Preveri ali je pixel dovolj temen
			if hsv[pix[1], pix[0], 2] < 50:
				if debug: cv.drawMarker(slika, tuple(pix), (0, 255, 150), cv.MARKER_SQUARE, 10)
				nove.append(pix)
				break

		# Ce imam vsaj toliko tock lahko neha z iskanjem
		if len(nove) > 12:
			if debug: cv.imshow("Vrhovi prileganje", slika)
			(x, y), (MA, ma), kot = cv.fitEllipseDirect(np.array(nove, dtype = np.int))
			E1, E2, E3 = (int(x), int(y)), (int(MA / 2), int(ma / 2)), int(kot)
			return E1, E2, E3

	if debug: cv.imshow("Vrhovi prileganje", slika)
	if izpisujOpozorila: print("Slabo prileganje")
	return (int(E1[0]), int(E1[1])), (int(E2[0]), int(E2[1])), int(E3)

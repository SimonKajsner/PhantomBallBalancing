import cv2 as cv
import numpy as np

debug = True

cv.namedWindow("Relativna poz")
cv.moveWindow("Relativna poz", 1240, 0)

zacetniPremerKroga = 250
premerKroga = zacetniPremerKroga

def relativnaPozicijaKrogle(krogla, elipsa, referencaP, TrajektorijaMode = False):
	"""
	Izracuna relativno pozicijo krogle na elipsi
	krogla - tocka pozicija krogle
	elipsa - elipsa plosce
	referencaP - v katero tocko se zelim premaknit
	TrajektorijaMode - ce je True se bo s casom notranji krog zmanjseval
	"""
	global premerKroga

	krogla = np.array(krogla)
	ploscaCenter, (MA, ma), kot = elipsa
	referenca = np.array(ploscaCenter)
	referencaLokalna = np.array(referencaP)

	# Za koliko poveca sliko
	obroba = 10

	# Maskiram ven samo plosco
	d = np.max([MA, ma])
	maska = np.zeros([(d + obroba) * 2, (d + obroba) * 2], dtype = np.uint8)
	cv.ellipse(maska, tuple(np.floor(np.flip(np.array(maska.shape) / 2)).astype(np.int32)), (MA, ma), kot, 0, 360, 255, -1)

	# Pozicija glede na plosco
	vk = krogla - referenca + (np.array(maska.shape) / 2).astype(np.int32)
	vrl = referencaLokalna - referenca + (np.array(maska.shape) / 2).astype(np.int32)

	# Maskiram notranji krog
	if TrajektorijaMode:
		maskaNotKroga = np.zeros([(d + obroba) * 2, (d + obroba) * 2], dtype = np.uint8)
		if premerKroga > 70:
			premerKroga -= 1
		cv.circle(maskaNotKroga, tuple(vrl), int(premerKroga), 1, -1)
		maska *= maskaNotKroga

	# vektor krogla referenca
	vkr = krogla - referencaLokalna
	normVkr = np.linalg.norm(vkr)
	if normVkr:
		vkrNorm = vkr / normVkr

		# Najde robno tocko v smeri vkrNorm
		i = 0
		robnaTocka = vk
		while maska[robnaTocka[1], robnaTocka[0]]:
			i += 1
			robnaTocka = np.round(vk + vkrNorm * i).astype(np.int32)
		robnaTocka = np.round(vk + vkrNorm * i - 1).astype(np.int32)

		if debug:
			cv.circle(maska, tuple(vk), 2, 100, -1)
			cv.circle(maska, tuple(robnaTocka), 5, 100)
			cv.line(maska, tuple(vk), tuple(vrl), 50)
			cv.imshow("Relativna poz", maska)

		# Izracuna relativno med vk in robno
		relativno = (vk - vrl) / np.linalg.norm(robnaTocka - vrl)
		return relativno * np.array([-1, 1])

	return np.zeros(2)

def resetirajPremerKroga():
	"""
	Povzroci, da se nitranji krog zacne zmanjsevati zacetniPremerKroga
	Klici kadar najdes novo trajektorijo
	"""
	global premerKroga, zacetniPremerKroga
	premerKroga = zacetniPremerKroga

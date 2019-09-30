import sys
import os
import cv2 as cv
import numpy as np
from FPS import FPS
import phantomVrhovi
from ShraniVideo import ShraniVideo
import krogla
import Simulink
from RelativnaPozicija import relativnaPozicijaKrogle
from RelativnaPozicija import resetirajPremerKroga
from itertools import combinations
from trajektorijaNaPlosci import narisanaTrajektorija, VrniNaslednjo
import time
from tkinter import *

debug = False

# za PID
alpha = 0.15
P = 18.0
I = 0.2
D = 3.9

# region GUI
# Ob spremembi sliderjev
def show_values(arg):
	global P, I, D, alpha
	P = w1.get()
	I = w2.get()
	D = w3.get()
	alpha = w4.get()

# Referencna tocka
refTocka = None

# V katerem nacinu vodenja si
nacinVodenja = "center"

# Za dolocitev poljubne ref tocke
zadnjaRefTockaZaPremik = None

# Sezam tock za trajektorijo
seznaTockTrajektorije = None

# Ob kliku na gumb trajektorija
def button_klik_traj():
	global nacinVodenja, seznaTockTrajektorije
	seznaTockTrajektorije = narisanaTrajektorija(slikaOrg, [ploscaCenter, (MA, ma), kot], 2)
	resetirajPremerKroga()
	nacinVodenja = "trajektorija"

# Ob kliku na gumb center
def button_klik_center():
	global refTocka, nacinVodenja
	refTocka = None
	nacinVodenja = "center"

# Ob kliku na gumb poljubna tocka
def button_klik_poljubnoSredisce():
	global refTocka, nacinVodenja, zadnjaRefTockaZaPremik
	nacinVodenja = "poljubnaTocka"

# Graficni vmesnik
masterGui = Tk()

# Velikosti graficnega vmesnika
masterGui.geometry('%dx%d+%d+%d' % (600, 550, 0, 0))

# Sliderji za PID parametre (P, I, D, Alpha)
w1 = Scale(masterGui, from_=0, to=25, resolution=0.1, command=show_values, orient=HORIZONTAL, width=40, length=1000)
w1.pack()
w1.set(P)
w2 = Scale(masterGui, from_=0, to=5, resolution=0.01, command=show_values, orient=HORIZONTAL, width=40, length=1000)
w2.pack()
w2.set(I)
w3 = Scale(masterGui, from_=0, to=15, resolution=0.01, command=show_values, orient=HORIZONTAL, width=40, length=1000)
w3.pack()
w3.set(D)
w4 = Scale(masterGui, from_=0, to=1, resolution=0.01, command=show_values, orient=HORIZONTAL, width=40, length=1000)
w4.pack()
w4.set(alpha)

# Gumbi
w5 = Button(masterGui, text = "Trajektorija", command = button_klik_traj)
w7 = Button(masterGui, text = "Center", command = button_klik_center)
w8 = Button(masterGui, text = "Poljubna tocka", command = button_klik_poljubnoSredisce)
w5.pack()
w7.pack()
w8.pack()

# CheckBoxi za rosanje stvari na sliko
w9Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje sledi zogice", variable = w9Var).pack()
w10Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje trajektorije", variable = w10Var).pack()
w11Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje referencne tocke", variable = w11Var).pack()
w12Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje zogice", variable = w12Var).pack()
w13Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje vrhov", variable = w13Var).pack()
w14Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje elipse plosce", variable = w14Var).pack()
w15Var = BooleanVar()
Checkbutton(masterGui, text = "Risanje centra plosce", variable = w15Var).pack()
# endregion

# Okno za risanje slike
cvWindowName = "Slika"
cv.namedWindow(cvWindowName)
# Lokacija okna
cv.moveWindow(cvWindowName, 600, 0)

# Globalne za funkcijo PID_regulator
posPrejZoga = np.array((0,0), dtype=np.float)
prejnagibRad = 0.0
prejU = np.array((0,0), dtype=np.float)
cas = -1
error = np.array((0,0), dtype=np.float)
prejError = np.array((0,0), dtype=np.float)
prejPosZogaPiksli = np.array((0,0))

def PID_regulator(posZoga, posRef, posZogaPiksli):
	"""
	PID regulator za vodenje zogice
	posZoga - trenutna pozicija relativno na ref tocko
	posRef - referencna tocka relativno (vedno [0, 0])
	posZogaPiksli - trenutna pozicija v pixlih
	"""
	global prejError, error, cas, prejU, prejnagibRad, alpha, posPrejZoga, P, I, D, prejPosZogaPiksli
	
	# Prvi vstop v regulator
	if cas == -1:
		# 1 / 30 je pricakovani FPS, ni tako pomembno ker se izvede samo v prvem trenutku
		cas = time.time() - 1 / 30
	
	posZogaPiksli = np.array(posZogaPiksli)
	
	# Preteceni cas od zadnjega klica regulatorja
	a = time.time() - cas
	
	# PID regulator
	U = P * (posRef - posZoga) + I * error + D * ((posPrejZoga - posZoga) / a)
	
	# Shrani pozicijo za nasledjo sliko
	posPrejZoga = posZoga
	
	# Skaliranje
	U = U / 100
	
	# Kdaj sem nazadnje preracunal hirost
	cas = time.time()
	
	# Omejitev maksimalnega odziva 
	lingLang = np.linalg.norm(U)
	if lingLang > 1:
		nagib = 1
	else:
		nagib = lingLang
	
	# Nelinearizacija odziva
	nagibRad = np.sin(nagib)

	# Izracun filtriranja dinamike in shranitev prejsnjih vrednosti
	nagibRad = prejnagibRad * alpha + nagibRad * (1 - alpha)
	prejnagibRad = nagibRad
	
	# Upostevanje filtracije dinamike
	U = prejU * alpha + U * (1 - alpha)
	prejU = U
	
	# Preverimo ali zogica stoji in blizino zeljene tocke
	trenutniError = posRef - posZoga
	if (np.linalg.norm(posZogaPiksli - prejPosZogaPiksli)) < 2 and np.linalg.norm(trenutniError) > 0.04:
		error += (posRef - posZoga)
	else: 
		error = np.array((0, 0), dtype = np.float)

	# Varovalo za pobeg vrednosti error, kar uporabis v I clenu
	np.where(np.abs(error) > 15., np.sign(error) * 15., error)

	# Shranitev prejsnjih vrednosti
	prejError = posRef - posZoga
	prejPosZogaPiksli = posZogaPiksli
	return U, nagibRad

def onMouse(event, x, y, flags, param):
	""" Se klice ob dogodkih miske nad oknom slike """
	global zadnjaRefTockaZaPremik
	
	if debug:
		# Izpisuje parametre pixla pod misko
		b, g, r = param[y, x]
		h, s, v = cv.cvtColor(param, cv.COLOR_BGR2HSV)[y, x]
		print("{x: %3d, y: %3d} {b: %3d, g: %3d, r: %3d} {h: %3d, s: %3d, v: %3d}" % (x, y, b, g, r, h, s, v))
		
	# Ce si kliknil na okno slike si izbral novo ref tocko
	if event == cv.EVENT_LBUTTONUP:
		zadnjaRefTockaZaPremik = [x, y]
	return

# Initializacija kamere
cap = cv.VideoCapture(1)
# Nastavi fokus na neskoncno
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

# Za posnet video
snemaj = False
if snemaj: video = ShraniVideo("Test video")

# Za FPS
FPS = FPS()
#FPS.NastaviZeljeniFPS(30)

# Za risanje sledi krogle
seznamTock = []
# Dolzina sledi krogle
stTock = 10

# Glavna zanka
while(cap.isOpened()):

	# GUI
	try:
		# posodobi gui
		masterGui.update()
	except:
		# Ce zapres gui koncaj program
		break

	# Zajemi sliko kamere
	_, slikaOrg = cap.read()
	# Na slika rises stvari, slikaOrg uporabljas za detekcijo
	slika = np.copy(slikaOrg)

	# Ce smo v nacinu za rajektorijo
	if nacinVodenja == "trajektorija":

		# Dobi naslednjo ref tocko
		naslednjaIzTrajektorije = VrniNaslednjo()
		if naslednjaIzTrajektorije is not None:
			refTocka = np.flip(naslednjaIzTrajektorije)

		# Risanje trajektorije
		if w10Var.get():
			slika[seznaTockTrajektorije[:,0],seznaTockTrajektorije[:,1],:] = [25, 230, 214]

	# Ce smo v nacinu poljubne tocke
	elif nacinVodenja == "poljubnaTocka":
		# Ce tocke slucajno se nisi kliknil (izbral) tocke, jo daj na trenutno ref tocko
		if zadnjaRefTockaZaPremik is None:
			zadnjaRefTockaZaPremik = refTocka
		# ref tocka je zadnja kliknjena tocka na sliki
		refTocka = zadnjaRefTockaZaPremik
	
	# Najde vrhove robotov
	vrhovi = phantomVrhovi.najdiVrhe(slikaOrg, True)
	
	# Ce najde vrhove
	if vrhovi is not None:

		# Najde elipso plosce
		ploscaCenter, (MA, ma), kot = phantomVrhovi.elipsa(vrhovi, slikaOrg, True)

		# Rise elipso plosce
		if w14Var.get():
			cv.ellipse(slika, ploscaCenter, (MA, ma), kot, 0, 360, (255, 255, 255))

		# Rise center plosce
		if w15Var.get():
			cv.drawMarker(slika, ploscaCenter, (0, 255, 255), cv.MARKER_TILTED_CROSS, 13, 2)

		# Ce smo v nacinu vodenja v center dolocimo ref v centru
		if nacinVodenja == "center":
			refTocka = ploscaCenter

		# Rise vrhove robotov
		if w13Var.get():
			for v in vrhovi.astype(np.uint):
				cv.drawMarker(slika, tuple(v), (255, 255, 0), cv.MARKER_TILTED_CROSS, 15, 2)

		# Najdi zogico
		kroglaTocka, kroglaPolmer = krogla.vrniKroglo(slikaOrg, [ploscaCenter, (MA, ma), kot])
		
		# Ce najde zogico
		if kroglaTocka is not None:
			# Narise kroglo
			if w12Var.get():
				cv.circle(slika, kroglaTocka, kroglaPolmer, (255, 0, 255), 2)

			# Izracun normirane relativne pozicije krogle na plosci
			pozicijaProcent = relativnaPozicijaKrogle(kroglaTocka, (ploscaCenter, (MA, ma), kot), refTocka, nacinVodenja == "trajektorija")

			# Nelinearizacija
			#pp = np.abs(pozicijaProcent)
			##ppa = 1.0 * pp ** 2 + 2.70616862252382e-16 * pp - 9.02056207507939e-17
			#ppa = -0.0005566703 + 0.4013623*pp + 2.085112*pp**2 - 1.490928*pp**3
			#pozicijaProcent = np.multiply( np.sign(pozicijaProcent), ppa )

			# Pid regulator
			napaka, nagib = PID_regulator(pozicijaProcent, np.array([0, 0]), kroglaTocka)

			# Poslji na simulink
			Simulink.poslji(napaka[0], napaka[1], nagib)
			

		# Narisi premaknjeno ref tocko
		if refTocka is not None and w11Var.get():
			cv.drawMarker(slika, tuple(refTocka), (200, 100, 255), cv.MARKER_TILTED_CROSS, 10, 2)

		# Risanje sledi krogle
		if w9Var.get():
			# Dodaj tocke v seznam
			seznamTock.append(kroglaTocka)
			
			# Risi samo ce imas vsaj 2 tocke
			if len(seznamTock) >= 2:
				
				for i in range(len(seznamTock) - 1):
					# Ce vmes ni nasel zogice je None in to preskoci
					if seznamTock[i + 1] is None or seznamTock[i] is None:
						continue
					# Narisi crto od ene tocke do druge
					cv.line(slika, seznamTock[i], seznamTock[i+1], (204,102,0))
					
				# Pobrisi ce je v seznamu vec tock kot stTock
				while len(seznamTock) >= stTock:
					seznamTock.pop(0)

	# Napisi text na sliko
	if nacinVodenja == "trajektorija":
		 cv.putText(slika, "Vodenje po trajektoriji", (10, 40), 4, 0.5, (255, 255, 255))
	elif nacinVodenja == "center":
		 cv.putText(slika, "Vodenje v sredisce", (10, 40), 4, 0.5, (255, 255, 255))
	elif nacinVodenja == "poljubnaTocka":
		 cv.putText(slika, "Vodenje v poljubno tocko", (10, 40), 4, 0.5, (255, 255, 255))
	
	# Izpisi FPS
	cv.putText(slika, f"FPS: %.2f" % FPS.VrniFps(), (5, slika.shape[0] - 15), 4, 0.5, (255, 255, 255))
	
	# Prikazi sliko v oknu
	cv.imshow(cvWindowName, slika)
	
	# Klici onMouse ob dogodkih miske na oknu
	cv.setMouseCallback(cvWindowName, onMouse, slikaOrg)

	# Za posnet video
	if snemaj: video.DodajFrame(slikaOrg)

	# Za cv.imshow() da vidis kaj dela
	cv.waitKey(1)

	# Za FPS
	FPS.Klici(Izpisi = False)

# Za posnet video
if snemaj: video.Koncal()

# Zapri vsa okna
cv.destroyAllWindows()

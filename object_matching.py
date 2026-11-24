#!/usr/bin/env python3

"""

LAB3: Deskryptory

Wersja ćwiczeniowa



Cele:

1. Wyznaczyć deskryptory dla obrazu wzorcowego.

2. Znaleźć obiekt na innym obrazie przez porównanie z wzorcem.

3. Śledzić obiekt w kolejnych klatkach filmu.



Ważna uwaga:

- W filmie obiekt może nie występować w każdej klatce.

- Jeżeli obiekt nie został wiarygodnie wykryty, nie należy go zaznaczać.



Instrukcja pracy:

- Uzupełniaj kolejne bloki TODO po kolei.

- Najpierw uruchom część dla pojedynczego obrazu.

- Dopiero po poprawnym działaniu przejdź do części wideo.

- Możesz użyć metody ORB, BRISK lub SIFT.

"""



import argparse

import sys

from typing import Optional, Tuple



import cv2

import numpy as np







def zmniejsz_do_okna(obraz: np.ndarray, max_szerokosc: int = 1200, max_wysokosc: int = 800) -> np.ndarray:

    h, w = obraz.shape[:2]



    skala_w = max_szerokosc / w

    skala_h = max_wysokosc / h

    skala = min(skala_w, skala_h, 1.0)



    nowa_szerokosc = int(w * skala)

    nowa_wysokosc = int(h * skala)



    if skala < 1.0:

        return cv2.resize(obraz, (nowa_szerokosc, nowa_wysokosc))

    return obraz



# ============================================================

# TODO 1: Utworzenie detektora i matchera

# ============================================================



def utworz_detektor(nazwa_metody: str = "ORB"):



    nazwa_metody = nazwa_metody.upper()



    if nazwa_metody == "ORB":

        return cv2.ORB_create(

            nfeatures=4000,

            scaleFactor=1.2,

            nlevels=8,

            edgeThreshold=10,

            patchSize=31,

            fastThreshold=20,

            scoreType=cv2.ORB_HARRIS_SCORE

        )

    elif nazwa_metody == "BRISK":

        return cv2.BRISK_create()

    elif nazwa_metody == "SIFT":

        return cv2.SIFT_create()

    else:

        raise ValueError(f"Nieznana metoda: {nazwa_metody}")





def utworz_matcher(nazwa_metody: str = "ORB"):

    nazwa_metody = nazwa_metody.upper()



    if nazwa_metody in ["ORB", "BRISK"]:

        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    elif nazwa_metody == "SIFT":

        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    else:

        raise ValueError(f"Nieznana metoda: {nazwa_metody}")







# ============================================================

# TODO 2: Wyznaczanie punktów kluczowych i deskryptorów

# ============================================================



def wyznacz_cechy(

    obraz_szary: np.ndarray,

    detektor

) -> Tuple[list, Optional[np.ndarray]]:

    punkty_kluczowe, deskryptory = detektor.detectAndCompute(obraz_szary, None)

    return punkty_kluczowe, deskryptory





# ============================================================

# TODO 3: Dopasowanie deskryptorów

# ============================================================



def dopasuj_deskryptory(

    deskryptory_wzorca: np.ndarray,

    deskryptory_obrazu: np.ndarray,

    matcher

):

    if deskryptory_wzorca is None or deskryptory_obrazu is None:

        return []



    if len(deskryptory_wzorca) == 0 or len(deskryptory_obrazu) == 0:

        return []



    # KNN w obie strony

    knn_12 = matcher.knnMatch(deskryptory_wzorca, deskryptory_obrazu, k=2)

    knn_21 = matcher.knnMatch(deskryptory_obrazu, deskryptory_wzorca, k=2)



    def ratio_test(knn_matches, prog=0.75):

        dobre = {}

        for para in knn_matches:

            if len(para) < 2:

                continue

            m, n = para

            if m.distance < prog * n.distance:

                dobre[m.queryIdx] = m

        return dobre



    dobre_12 = ratio_test(knn_12, prog=0.75)

    dobre_21 = ratio_test(knn_21, prog=0.75)



    # Symmetry test

    wzajemne = []

    for qidx, m in dobre_12.items():

        train_idx = m.trainIdx

        if train_idx in dobre_21:

            m2 = dobre_21[train_idx]

            if m2.trainIdx == qidx:

                wzajemne.append(m)



    wzajemne = sorted(wzajemne, key=lambda x: x.distance)



    MAX_DOPASOWAN = 100

    return wzajemne[:MAX_DOPASOWAN]





# ============================================================

# TODO 4: Lokalizacja obiektu

# ============================================================



def zlokalizuj_obiekt(

    punkty_wzorca,

    punkty_obrazu,

    dopasowania,

    ksztalt_wzorca,

    ksztalt_obrazu

):

    MIN_DOPASOWAN = 12

    MIN_INLIEROW = 10

    MIN_INLIER_RATIO = 0.4



    if dopasowania is None or len(dopasowania) < MIN_DOPASOWAN:

        return None, None



    pkt_wzorca = np.float32([

        punkty_wzorca[m.queryIdx].pt for m in dopasowania

    ]).reshape(-1, 1, 2)



    pkt_obrazu = np.float32([

        punkty_obrazu[m.trainIdx].pt for m in dopasowania

    ]).reshape(-1, 1, 2)



    H, maska_inlierow = cv2.findHomography(

        pkt_wzorca,

        pkt_obrazu,

        cv2.RANSAC,

        3.0

    )



    if H is None or maska_inlierow is None:

        return None, None



    liczba_inlierow = int(maska_inlierow.sum())

    inlier_ratio = liczba_inlierow / len(dopasowania)



    if liczba_inlierow < MIN_INLIEROW or inlier_ratio < MIN_INLIER_RATIO:

        return None, None



    wysokosc, szerokosc = ksztalt_wzorca[:2]



    narozniki_wzorca = np.float32([

        [0, 0],

        [szerokosc - 1, 0],

        [szerokosc - 1, wysokosc - 1],

        [0, wysokosc - 1]

    ]).reshape(-1, 1, 2)



    narozniki_obrazu = cv2.perspectiveTransform(narozniki_wzorca, H)



    if not czy_poprawny_czworokat(narozniki_obrazu, ksztalt_obrazu, ksztalt_wzorca):

        return None, None



    return narozniki_obrazu, maska_inlierow



def pole_wielokata(pts):

    pts = pts.reshape(-1, 2)

    return abs(cv2.contourArea(pts.astype(np.float32)))



def czy_poprawny_czworokat(narozniki, ksztalt_obrazu, ksztalt_wzorca):

    if narozniki is None or len(narozniki) != 4:

        return False



    h_img, w_img = ksztalt_obrazu[:2]

    h_ref, w_ref = ksztalt_wzorca[:2]



    pts = narozniki.reshape(-1, 2)



    area = pole_wielokata(pts)

    area_ref = w_ref * h_ref



    if area < 0.03 * area_ref:

        return False

    if area > 8.0 * area_ref:

        return False



    hull = cv2.convexHull(pts.astype(np.float32))

    if len(hull) != 4:

        return False



    d01 = np.linalg.norm(pts[0] - pts[1])

    d12 = np.linalg.norm(pts[1] - pts[2])

    d23 = np.linalg.norm(pts[2] - pts[3])

    d30 = np.linalg.norm(pts[3] - pts[0])



    min_bok = min(d01, d12, d23, d30)

    max_bok = max(d01, d12, d23, d30)



    if min_bok < 20:

        return False

    if max_bok / max(min_bok, 1e-6) > 10:

        return False



    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))

    if x > w_img or y > h_img or x + w < 0 or y + h < 0:

        return False



    return True





# ============================================================

# TODO 5: Wizualizacja wyniku

# ============================================================



def narysuj_obiekt(obraz_bgr: np.ndarray, narozniki) -> np.ndarray:

    wynik = obraz_bgr.copy()



    if narozniki is not None:

        pts = np.int32(narozniki)

        cv2.polylines(wynik, [pts], True, (0, 255, 0), 3)



    return wynik



def narysuj_dopasowania(

    obraz_wzorca,

    punkty_wzorca,

    obraz_testowy,

    punkty_testowe,

    dopasowania,

    maks_liczba: int = 50

):

    wybrane_dopasowania = dopasowania[:maks_liczba]



    obraz = cv2.drawMatches(

        obraz_wzorca,

        punkty_wzorca,

        obraz_testowy,

        punkty_testowe,

        wybrane_dopasowania,

        None,

        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    )



    return obraz





# ============================================================

# Część 1 i 2: obraz wzorcowy + obraz testowy

# ============================================================



def przetworz_obraz(referencja: str, obraz_testowy: str, metoda: str):



    obraz_wzorca_bgr = cv2.imread(referencja)

    obraz_testowy_bgr = cv2.imread(obraz_testowy)



    if obraz_wzorca_bgr is None:

        print(f"BŁĄD: Nie można otworzyć obrazu wzorcowego: {referencja}", file=sys.stderr)

        return



    if obraz_testowy_bgr is None:

        print(f"BŁĄD: Nie można otworzyć obrazu testowego: {obraz_testowy}", file=sys.stderr)

        return



    obraz_wzorca_szary = cv2.cvtColor(obraz_wzorca_bgr, cv2.COLOR_BGR2GRAY)

    obraz_testowy_szary = cv2.cvtColor(obraz_testowy_bgr, cv2.COLOR_BGR2GRAY)



    detektor = utworz_detektor(metoda)

    matcher = utworz_matcher(metoda)



    punkty_wzorca, deskryptory_wzorca = wyznacz_cechy(obraz_wzorca_szary, detektor)

    punkty_testowe, deskryptory_testowe = wyznacz_cechy(obraz_testowy_szary, detektor)



    dobre_dopasowania = dopasuj_deskryptory(deskryptory_wzorca, deskryptory_testowe, matcher)



    narozniki, maska_inlierow = zlokalizuj_obiekt(

        punkty_wzorca,

        punkty_testowe,

        dobre_dopasowania,

        obraz_wzorca_szary.shape,

        obraz_testowy_szary.shape

    )



    obraz_wynikowy = narysuj_obiekt(obraz_testowy_bgr, narozniki)

    obraz_dopasowan = narysuj_dopasowania(

        obraz_wzorca_bgr,

        punkty_wzorca,

        obraz_testowy_bgr,

        punkty_testowe,

        dobre_dopasowania

    )



    znaleziono = narozniki is not None

    tekst1 = f"Dobre dopasowania: {len(dobre_dopasowania)}"

    tekst2 = "Obiekt znaleziony" if znaleziono else "Obiekt NIE znaleziony"



    cv2.putText(obraz_wynikowy, tekst1, (20, 30),

                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(obraz_wynikowy, tekst2, (20, 65),

                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if znaleziono else (0, 0, 255), 2)



    obraz_wzorca_pokaz = zmniejsz_do_okna(obraz_wzorca_bgr)

    obraz_wynikowy_pokaz = zmniejsz_do_okna(obraz_wynikowy)

    obraz_dopasowan_pokaz = zmniejsz_do_okna(obraz_dopasowan)



    cv2.namedWindow("Obraz wzorcowy", cv2.WINDOW_NORMAL)

    cv2.namedWindow("Wykryty obiekt", cv2.WINDOW_NORMAL)

    cv2.namedWindow("Dopasowane punkty", cv2.WINDOW_NORMAL)



    cv2.imshow("Obraz wzorcowy", obraz_wzorca_pokaz)

    cv2.imshow("Wykryty obiekt", obraz_wynikowy_pokaz)

    cv2.imshow("Dopasowane punkty", obraz_dopasowan_pokaz)



    cv2.waitKey(0)

    cv2.destroyAllWindows()





# ============================================================

# Część 3: śledzenie wideo

# ============================================================



def przetworz_wideo(referencja: str, plik_wideo: str, metoda: str):

    obraz_wzorca_bgr = cv2.imread(referencja)

    if obraz_wzorca_bgr is None:

        print(f"BŁĄD: Nie można otworzyć obrazu wzorcowego: {referencja}", file=sys.stderr)

        return



    cap = cv2.VideoCapture(plik_wideo)

    if not cap.isOpened():

        print(f"BŁĄD: Nie można otworzyć pliku wideo: {plik_wideo}", file=sys.stderr)

        return



    obraz_wzorca_szary = cv2.cvtColor(obraz_wzorca_bgr, cv2.COLOR_BGR2GRAY)



    detektor = utworz_detektor(metoda)

    matcher = utworz_matcher(metoda)



    punkty_wzorca, deskryptory_wzorca = wyznacz_cechy(obraz_wzorca_szary, detektor)



    fps = cap.get(cv2.CAP_PROP_FPS)

    opoznienie = int(1000 / fps) if fps and fps > 1 else 20



    while True:

        poprawnie, klatka = cap.read()

        if not poprawnie:

            break



        klatka_szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)



        punkty_klatki, deskryptory_klatki = wyznacz_cechy(klatka_szara, detektor)

        dobre_dopasowania = dopasuj_deskryptory(deskryptory_wzorca, deskryptory_klatki, matcher)



        narozniki, maska_inlierow = zlokalizuj_obiekt(

            punkty_wzorca,

            punkty_klatki,

            dobre_dopasowania,

            obraz_wzorca_szary.shape,

            klatka_szara.shape

        )



        wynik = narysuj_obiekt(klatka, narozniki)



        znaleziono = narozniki is not None

        tekst1 = f"Dobre dopasowania: {len(dobre_dopasowania)}"

        tekst2 = "Obiekt znaleziony" if znaleziono else "Obiekt NIE znaleziony"



        cv2.putText(wynik, tekst1, (20, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(wynik, tekst2, (20, 65),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if znaleziono else (0, 0, 255), 2)



        cv2.imshow("Śledzenie obiektu w wideo", wynik)



        klawisz = cv2.waitKey(opoznienie) & 0xFF

        if klawisz in (ord("q"), 27):

            break



    cap.release()

    cv2.destroyAllWindows()





# ============================================================

# Funkcja główna

# ============================================================



def main():

    parser = argparse.ArgumentParser(description="LAB3 - wersja ćwiczeniowa: deskryptory")

    parser.add_argument("--reference", required=True, help="Ścieżka do obrazu wzorcowego, np. saw1.jpg")

    parser.add_argument("--image", help="Ścieżka do obrazu testowego")

    parser.add_argument("--video", help="Ścieżka do pliku wideo, np. sawmovie.mp4")

    parser.add_argument("--method", default="ORB", help="Metoda deskryptorów: ORB, BRISK, SIFT")



    args = parser.parse_args()



    if args.image is None and args.video is None:

        print("BŁĄD: Podaj --image lub --video", file=sys.stderr)

        return 1



    if args.image is not None:

        przetworz_obraz(args.reference, args.image, args.method)



    if args.video is not None:

        przetworz_wideo(args.reference, args.video, args.method)



    return 0





if __name__ == "__main__":

    raise SystemExit(main())
import argparse

import math

import sys



import cv2

import numpy as np





def buduj_maske_czerwieni(bgr: np.ndarray) -> np.ndarray:

    """Segmentacja czerwieni w HSV + czyszczenie maski (OPEN i CLOSE). Zwraca maskę 0/255."""

    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)



    lower1 = np.array([0, 150, 60])

    upper1 = np.array([6, 255, 255])



    lower2 = np.array([174, 150, 60])

    upper2 = np.array([179, 255, 255])



    mask1 = cv2.inRange(hsv, lower1, upper1)

    mask2 = cv2.inRange(hsv, lower2, upper2)



    maska = cv2.bitwise_or(mask1, mask2)



    kernel_open = np.ones((3, 3), np.uint8)

    kernel_close = np.ones((15, 15), np.uint8)



    maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel_open, iterations=1)

    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel_close, iterations=3)



    return maska





def obiekt_z_momentow(maska_0_255: np.ndarray, min_pole: float = 300.0):

    """

    Liczy momenty bez konturów (założenie: jest tylko jeden czerwony obiekt).

    Zwraca:

      (cx, cy), pole_px, promien_px  albo (None, None, None)

    """

    M = cv2.moments(maska_0_255)



    pole = M["m00"] / 255.0



    if pole < min_pole:

        return None, None, None



    cx = int(M["m10"] / M["m00"])

    cy = int(M["m01"] / M["m00"])



    promien = int(np.sqrt(pole / np.pi))



    return (cx, cy), pole, promien



def rysuj_paski_odchylenia(img: np.ndarray, cx: int, szerokosc: int, y: int = 90,

                           wysokosc_paska: int = 16, margines: int = 10):

    """

    Rysuje pasek pokazujący odchylenie lewo/prawo od środka kadru.

    """

    x_srodek = szerokosc // 2



    x1 = margines

    x2 = szerokosc - margines



    y1 = y

    y2 = y + wysokosc_paska



    # ramka paska

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)



    # kreska środka

    cv2.line(img, (x_srodek, y1), (x_srodek, y2), (255, 255, 255), 1)



    # wypełnienie od środka do obiektu

    lewo = min(x_srodek, cx)

    prawo = max(x_srodek, cx)



    cv2.rectangle(img, (lewo, y1), (prawo, y2), (255, 0, 255), -1)



def main() -> int:

    parser = argparse.ArgumentParser(description="LAB1: Detekcja czerwonego obiektu (momenty + bars)")

    parser.add_argument("--video", required=True, help="Ścieżka do pliku wideo, np. sample.mp4")

    parser.add_argument("--min-pole", type=float, default=300.0, help="Minimalne pole obiektu [px^2]")

    args = parser.parse_args()



    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():

        print(f"BŁĄD: Nie można otworzyć pliku wideo: {args.video}", file=sys.stderr)

        return 1



    fps = cap.get(cv2.CAP_PROP_FPS)

    opoznienie_ms = int(1000 / fps) if fps and fps > 1e-3 else 20



    while True:

        ok, klatka = cap.read()

        if not ok:

            break



        h, w = klatka.shape[:2]

        x_srodek = w // 2



        maska = buduj_maske_czerwieni(klatka)

        centrum, pole, promien = obiekt_z_momentow(maska, min_pole=args.min_pole)



        # Okno 1: oryginał + oznaczenia

        wiz = klatka.copy()



        # Linia środka kadru (pomocniczo)

        cv2.line(wiz, (x_srodek, 0), (x_srodek, h), (255, 255, 255), 1)



        if centrum is not None:

            cx, cy = centrum

            odchylenie = cx - x_srodek



            # Okrąg obejmujący obiekt (promień z pola, założenie: koło)

            cv2.circle(wiz, (cx, cy), max(promien, 1), (255, 0, 0), 2)

            cv2.circle(wiz, (cx, cy), 3, (255, 0, 0), -1)



            # Paski odchylenia (bars)

            rysuj_paski_odchylenia(wiz, cx=cx, szerokosc=w, y=40, wysokosc_paska=16)



        maska_bgr = cv2.cvtColor(maska, cv2.COLOR_GRAY2BGR)

        podglad = np.hstack([wiz, maska_bgr])

        podglad_small = cv2.resize(podglad, None, fx=0.5, fy=0.5)

        cv2.imshow("Podglad", podglad_small)



        klawisz = cv2.waitKey(opoznienie_ms) & 0xFF

        if klawisz in (ord("q"), 27):  # q lub ESC

            break



    cap.release()

    cv2.destroyAllWindows()

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


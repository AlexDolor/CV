import cv2
import numpy as np

VIDEO_PATH = "Seminars/Seminar_4/data/video_cat.mp4"

sigma_value = 0      # положение ползунка (целое число)
paused = False       # флаг паузы


def on_sigma_change(pos):
    """Callback для трекбара: просто обновляем глобальное значение sigma."""
    global sigma_value
    sigma_value = pos


def main():
    global paused

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео. Проверь путь к файлу.")

    # читаем первый кадр
    ret, prev = cap.read()
    if not ret or prev is None:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Original / Blurred", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Temporal difference |It|", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("sigma_x2", "Original / Blurred", 0, 20, on_sigma_change)
    # фактическая sigma = sigma_value / 2.0

    while True:
        if not paused:
            # читаем следующий кадр
            ret, frame = cap.read()

            if not ret or frame is None:
                # дошли до конца видео: перематываем на начало и читаем снова
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # перейти к кадру 0[web:101][web:102]
                ret, frame = cap.read()
                if not ret or frame is None:
                    # если даже после сброса кадр не прочитался — выходим
                    break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            sigma = sigma_value / 2.0
            if sigma > 0:
                gray_blur = cv2.GaussianBlur(gray, (9, 9), sigmaX=sigma, sigmaY=sigma)
            else:
                gray_blur = gray

            It = cv2.absdiff(gray_blur, prev_gray)

            merged = np.hstack([
                cv2.putText(gray.copy(), "original", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2),
                cv2.putText(gray_blur.copy(), f"blur sigma={sigma:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            ])

            cv2.imshow("Original / Blurred", merged)
            cv2.imshow("Temporal difference |It|", It)

            # обновляем prev_gray только если не на паузе
            prev_gray = gray_blur.copy()
        else:
            # если на паузе — просто обновляем отображение окна, чтобы оно не висло
            cv2.imshow("Original / Blurred", merged)
            cv2.imshow("Temporal difference |It|", It)

        # обработка клавиш
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            # выход
            break
        elif key == ord('p'):
            # переключаем паузу
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

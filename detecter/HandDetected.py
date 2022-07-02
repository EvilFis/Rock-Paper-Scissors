from time import time
import numpy as np
import mediapipe as mp
import cv2

class HandDetected:

    def __init__(self, camera_num:int|str=0,
                static_image_mode:bool=False,
                model_complexity:int=0,
                min_detection_confidence:float=0.5,
                min_tracking_confidence:float=0.5,
                max_num_hands:int=1,
                debug:bool=False,
                game_size:tuple=(1920, 1080)
                ):
        # Общие переменные
        self.debug_mode = debug
        self.flip_flag_hand=None

        # Переменные библиотеки по поиску рук
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        ) 

        # Перемнные видео настроек
        self.cap = cv2.VideoCapture(camera_num)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame = None

        # Размер игрового экрана
        if 2 < len(game_size) <= 3:
            raise ValueError('Должно быть два значения')
        
        self.game_size = (
            game_size[0]/self.frame_width,
            game_size[1]/self.frame_height
        )

        # Перемнные для расчёта fps
        self.frame_start = 0
        self.frame_end = 0

        # Переменные с координатами необходимых точек
        self.palm_point = [0,5,9,13,17]
        self.fingers_point = [[8,6], [12, 10], [16, 14], [20, 18]]

        # Графические материалы
        self.source = {
            "img": {
                "palm": "./detecter/img/hand.png",
                "fist": "./detecter/img/fist.png"
            } 
        }

        # Подгрузка изображений
        self.img_hand = cv2.imread(self.source['img']['palm'])
        self.img_hand_h, self.img_hand_w, _ = self.img_hand.shape


    def _change_font() -> None:

        """
            Тестовая функция, служит временной заглушкой
        """

        ft = cv2.freetype.createFreeType2()
        ft.loadFontData(fontFileName='Ubuntu-R.ttf', id=0)
                

    def _show_fps(self, frame:np.array, 
                position:tuple=(0, 0), 
                font:int=cv2.FONT_HERSHEY_SIMPLEX,
                fontHeight:int=1, 
                color:tuple=(0,0,0), 
                bold:int=3, 
                type_line:int=cv2.LINE_AA) -> None:

        """
            Функция по расчету кадров в секунду
            @param: 
                    position:tuple - Место где будет распологаться текстовая информация
                    font:int - Стиль текста
                    fontHeight:int - Размер шрифта
                    color:tuple - Цвет отображаемого текста
                    bold:int - Толщина текста
                    type_line:int - Тип линий текста
            @return: 
                    None
        """

        self.frame_start = time()
        fps = str(int(1/(self.frame_start-self.frame_end)))
        self.frame_end = self.frame_start

        cv2.putText(frame, fps, position, font, fontHeight, color, bold, type_line)

    
    def _flip_hand_img(self, hands_res, position) -> None:
        """
            Разворот изображения при показе другой руки
            @param: 
                    hands_res - Координаты точек руки
                    position - Какая позиция нужна (Right или Left)
            @return: 
                    None
        """
        if hands_res.multi_handedness[0].classification[0].label == position:
            if not self.flip_flag_hand == position:
                # self.img_hand = cv2.flip(self.img_hand, 1)
                self.img_hand = self.img_hand[:,::-1]
                print(position) if self.debug_mode else None
            self.flip_flag_hand = position


    def _calc_center_point(self, search_w:list, search_f:list) -> tuple:
        """
            Функция по расчету центральных точек
            @param:
                search_w:list - Где ищем данные
                search_f:list - От куда берем данные
            @return: 
                    center_points:tuple - Значения центральных точек по осям Х и Y
        """
        x, y = 0, 0

        for point in search_f:
            if type(point) is list:
                x += search_w[point[0]][0]
                y += search_w[point[0]][1]

            else:
                x += search_w[point][0]
                y += search_w[point][1]

        return (int(x/len(search_f)), int(y/len(search_f)))


    def _state_hand(self, frame, finger_cords:tuple, palm_cords:tuple) -> None:
        """
            Функция по базовой проверке, открыта рука или нет
            @param:
                frame - Переменная необходимая для вывода текста
                finger_cords:tuple - Кординаты пальцев
                palm_cords:tuple - Кординаты лодони
            @return: 
                    None
        """
        center_x = finger_cords[0] - palm_cords[0]
        center_y = finger_cords[1] - palm_cords[1]

        # Проверяем, что у нас кулак или лодошка 
        if  -25 <= center_y <= 35 and -25 <= center_x <= 35:
            cv2.putText(frame, " fist is visible", (80, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (22, 56, 30), 2, cv2.LINE_AA)

            self.img_hand = cv2.imread(self.source['img']['fist'])
            self.img_hand_h, self.img_hand_w, _ = self.img_hand.shape

        else:
            cv2.putText(frame, " fist is not visible", (80, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (22, 56, 30), 2, cv2.LINE_AA)

            self.img_hand = cv2.imread(self.source['img']['palm'])
            self.img_hand_h, self.img_hand_w, _ = self.img_hand.shape


    def _move_hand(self, frame, mask, palm_cords:tuple) -> np.array:
        """
            Функция отвечающая за перемещение картинки руки
            @param:
                frame - Кадр, на котором будет отображаться картинка
                mask - Маска картинки
                palm_cords:tuple - Кординаты лодони
            @return: 
                    frame - Изминенный кадр
        """
        h_f, w_f, _ = frame.shape
        h_i, w_i = self.img_hand_h, self.img_hand_w

        try:
            roi = frame[palm_cords[1]-int(h_i/2):palm_cords[1]+int(h_i/2),
                        palm_cords[0]-int(w_i/2):palm_cords[0]+int(w_i/2)]

            # Ось X
            if palm_cords[0] <= w_i:
                roi = frame[palm_cords[1]-int(h_i/2):palm_cords[1]+int(h_i/2),
                            0:w_i]

            elif palm_cords[0] >= w_f-w_i:
                roi = frame[palm_cords[1]-int(h_i/2):palm_cords[1]+int(h_i/2),
                            w_f-w_i:w_f]

            # Ось Y
            elif palm_cords[1] <= h_i:
                roi = frame[0:h_i, 
                            palm_cords[0]-int(w_i/2):palm_cords[0]+int(w_i/2)]

            elif palm_cords[1] >= h_f-h_i:
                roi = frame[h_f-h_i:h_f, 
                            palm_cords[0]-int(w_i/2):palm_cords[0]+int(w_i/2)]

            roi[np.where(mask)] = 0
            roi += self.img_hand
        except:
            pass

        return frame


    def _processing_hand(self, frame, mask) -> None:
        """
            Функция отвечающая за детектирование руки и реализацию других функций
            @param:
                frame - Кадр, на котором будет отображаться картинка
                mask - Маска картинки
            @return:
                    None
        """

        self.frame.flags.writeable = False
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        hands_res = self.hands.process(self.frame)
        self.frame.flags.writeable = True
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        if hands_res.multi_hand_landmarks:
            handList = []

            for hand_landmarks in hands_res.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    self.frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                ) if self.debug_mode else None
                
                self._flip_hand_img(hands_res, 'Right')
                self._flip_hand_img(hands_res, 'Left')

                for _, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = (lm.x*w), (lm.y*h)
                    handList.append((float(cx), float(cy)))

                palm_x, palm_y = self._calc_center_point(handList, self.palm_point)
                fingers_x, fingers_y = self._calc_center_point(handList, self.fingers_point)
                
                self._state_hand(frame, palm_cords=(palm_x, palm_y), 
                                finger_cords=(fingers_x, fingers_y))

            self._move_hand(frame=frame, mask=mask, palm_cords=(palm_x, palm_y))

    def start_system(self):
        
        """
            Запуск всей системы
        """

        print('frame dimensions (HxW):', self.frame_height,"x", self.frame_width)

        while self.cap.isOpened():
            success, self.frame = self.cap.read()

            if not success:
                raise RuntimeError("Не удалось прочитать кадры")

            self.frame = cv2.flip(self.frame, 1)
            frame_game = self.frame.copy()

            self.frame = cv2.resize(self.frame, (0,0), fx=0.35, fy=0.35)
            frame_game = cv2.resize(frame_game,
                                    (0,0),
                                    fx=self.game_size[0],
                                    fy=self.game_size[1])

            self._show_fps(self.frame, (15, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 255, 0), 1, cv2.LINE_AA)

            img2gray = cv2.cvtColor(self.img_hand, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            self._processing_hand(frame=frame_game, mask=mask)

            cv2.imshow("Main video", self.frame) if self.debug_mode else None
            cv2.imshow("Rock-Paper-Scissors", frame_game) 

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    pr = HandDetected(game_size=(640, 480), debug=True)
    pr.start_system()
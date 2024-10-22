<div style="text-align: left;">
 
### **Заказчик**

**Renue** – IT-компания из Екатеринбурга, разрабатывает высоконагруженные и отказоустойчивые решения для крупных российских заказчиков, для бизнеса и государства.

### **Описание проекта**
Необходимо разработать решение для отслеживания и сортировки мусора на конвейере – выделять пластиковые бутылки в общем потоке предметов.

### **Данные**
 - Датасет (изображения + разметка) в нескольких форматах: MOT, COCO, CVAT
 - Примеры видеозаписей работы конвейера

### **Цель**
 - Решение должно выдавать координаты центра обнаруженных пластиковых бутылок для каждого кадра.
 - Скорость обработки должна быть не более 100 мс.

### **Рассмотренные трекеры и их результаты**

![image](https://github.com/user-attachments/assets/7beb52f4-1001-4161-908f-d037d5c0b9fd)

### **Структура директории**
```
PlasticWasteTracking/
├── src/
│   ├── inference.py
│   ├── utils.py
├── models/
│   ├── SFSORT.py
│   ├── mc_SMILEtrack.py
│   ├── sort.py
│   └── sam2s/
│       ├── build_sam.py
│       └── sam2_camera_predictor.py
├── notebooks/
│   ├── renue-tracking-eda.ipynb
│   ├── renue-baseline-sorts.ipynb
│   ├── renue-segmentations-sam2.ipynb
│   ├── renue-search-new-trackers.ipynb
│   └── renue-smiletrack.ipynb
└── README.md
```
</div>



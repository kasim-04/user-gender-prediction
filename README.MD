# Предсказание пола пользователей соцсетей

Проект выполнен в рамках образовательной программы [проекты для студентов](https://education.vk.company/education_projects) от VK. 

## О проекте

В рамках кейса решалась задача классификации пола пользователей социальной сети. Данные были предоставлены организаторами программы.

### Основные этапы работы:

**Шаг 1 — данных**

Была проведена обработка предоставленных данных для формирования единого обучающего набора.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17MgffR9Tph_seOMW89eiUn2pG1NZ2mrs?usp=sharing)

**Шаг 2 — Анализ данных**

Был проведен разведывательный анализ данных для понимания структуры, распределений и взаимосвязей признаков.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1knI0mpH8nQUp-Y51mE9SHjIta1TTIcby?usp=sharing)

**Шаг 3 — Разработка и обучение модели**

Была разработана модель *случайного леса с нуля*. Модель была обучена на подготовленных данных. Метрики на тесте: 

<!-- METRICS_START -->

| accuracy | roc_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| 0.81 | 0.81 | 0.80 | 0.80 | 0.80 |

<!-- METRICS_END -->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_mON63ODDzO4CXpE3OnuGcV-SFf3OWRw?usp=sharing)

**Шаг 4 — Инференс**

Было выполнено предсказание пола для новых пользователей.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wVQ2fTZVOY2rGV3ut08stZLMJ7II8L9G?usp=sharing)

## Сертификат

![Сертификат об окончании курса VK](certificate.jpg)

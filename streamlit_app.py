import streamlit as st
import numpy as np
import joblib

# === Загрузка модели и scaler ===
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')

# === Заголовок и описание ===
st.set_page_config(page_title="🛳️ Titanic Survival Predictor", page_icon="🚢")
st.title("🛳️ Titanic Survival Prediction")
st.markdown("Введите данные пассажира и узнайте, каков его шанс выжить!")

st.divider()

# === Ввод от пользователя ===
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Пол", ["Мужчина", "Женщина"])
    pclass = st.selectbox("Класс обслуживания", [1, 2, 3])
    age_group = st.selectbox("Возрастная группа", ["Ребёнок (0-12)", "Подросток (13-18)", "Взрослый (19-55)", "Пожилой (55+)"])

with col2:
    fare = st.number_input("Цена билета ($)", min_value=0.0, value=30.0, step=1.0)
    embarked = st.selectbox("Порт посадки", ["Southampton", "Cherbourg", "Queenstown"])
    family_size = st.slider("Размер семьи (включая себя)", 1, 10, 1)

# === Признаки ===
is_alone = 1 if family_size == 1 else 0
sex = 1 if sex == "Женщины" else 0
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]
age_group = {"Ребёнок (0-12)": 0, "Подросток (13-18)": 1, "Взрослый (19-55)": 2, "Пожилой (55+)": 3}[age_group]

features = np.array([[sex, pclass, fare, embarked, family_size, is_alone, age_group]])
features_scaled = scaler.transform(features)

# === Предсказание ===
if st.button("🔍 Предсказать выживание"):
    proba = model.predict_proba(features_scaled)[0][1]  # вероятность выживания
    percentage = round(proba * 100, 2)

    st.markdown("### 🧾 Вероятность выживания:")
    st.progress(proba, text=f"{percentage}%")

    if proba > 0.8:
        st.success(f"🎉 Пассажир почти точно выживет! ({percentage}%)")
    elif proba > 0.5:
        st.info(f"⚠️ Пассажир, скорее всего, выживет. ({percentage}%)")
    else:
        st.error(f"💀 К сожалению, шанс невелик. ({percentage}%)")

    # Показать детали (опционально)
    with st.expander("📊 Подробнее"):
        st.write("**Введённые признаки:**")
        st.json({
            "Пол": "Женщина" if sex == 1 else "Мужчина",
            "Класс": pclass,
            "Возрастная группа": age_group,
            "Цена билета": fare,
            "Порт посадки": embarked,
            "Размер семьи": family_size,
            "Один путешествовал": bool(is_alone)
        })

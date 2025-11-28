# ============================================================
# STREAMLIT TEXT CLASSIFIER DASHBOARD (FINAL + METRICS UI)
# ============================================================

import streamlit as st          # Streamlit – веб дээр dashboard хийх сан
import pandas as pd             # Pandas – өгөгдөл боловсруулах сан
import numpy as np              # NumPy – тооцоолол, матриц, массив
import re                       # re – Regular Expression (текст цэвэрлэх)

from sklearn.model_selection import train_test_split  # Өгөгдөл сургалт/тестэд хуваах
from sklearn.pipeline import Pipeline                 # Pipeline – дараалсан алхамууд нэгтгэх
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Текст → тоо болгох
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes ангилагч
from sklearn.linear_model import LogisticRegression    # Ложистик регресс ангилагч
from sklearn.metrics import (                          # Загварын үнэлгээний метрикууд
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt          # Matplotlib – график зурах
import seaborn as sns                    # Seaborn – илүү гоё график
from mpl_toolkits.mplot3d import Axes3D  # 3D график үүсгэх
from matplotlib import cm                # Өнгөний схем
import pandas as pd                      # Pandas дахин импортловол асуудалгүй
import plotly.express as px              # Plotly – интерактив график хийх

# Streamlit апп-ийн үндсэн тохиргоо
st.set_page_config(page_title="Twitter Sentiment Classifier", layout="wide")

# -----------------------------
# Cleaning function
# -----------------------------
def clean_tweet(text):
    text = str(text).lower()                                # Текстийг жижиг үсэг болгох
    text = re.sub(r"http\S+|www\.\S+", "", text)            # URL устгах
    text = re.sub(r"@\w+", "", text)                        # @username устгах
    text = re.sub(r"&amp;", "and", text)                    # &amp → and болгох
    text = re.sub(r"rt[\s]+", "", text)                     # RT (retweet) устгах
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # Тэмдэгтүүдийг зай болгох
    text = re.sub(r"\s+", " ", text).strip()                # Нэмэлт зай цэвэрлэх
    return text                                              # Цэвэрлэсэн текст буцаах

# -----------------------------
# Load CSV
# -----------------------------
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="latin-1",                              # Twitter dataset латин кодлогддог
            header=None,                                     # Х багануудгүй тул өөрөө нэр өгнө
            names=["target","id","date","flag","user","text"],  # Багануудын нэр
            quoting=1, quotechar='"', escapechar="\\",      # Тэмдэгтүүдийг зөв тайлах
            engine="python", on_bad_lines="skip"            # Алдаатай мөрүүдийг алгасах
        )
        return df
    except Exception as e:
        st.error(f"CSV уншихад алдаа: {e}")                  # Хэрэв алдаа гарвал Streamlit-д харуулна
        return None

# -----------------------------
# Prepare Data
# -----------------------------
def prepare_df(df):
    if "target" not in df.columns or "text" not in df.columns:
        st.error("CSV нь Twitter dataset хэлбэртэй байх ёстой.")   # Алдаатай файл шалгах
        return None
    if set(df["target"].unique()) == {0,4}:                        # 0=negative, 4=positive
        df["target"] = df["target"].map({0:0, 4:1})                # 4 → 1 болгон map хийх
    df["clean_text"] = df["text"].astype(str).apply(clean_tweet)  # Текстийг цэвэрлэж шинэ баганад хийх
    return df

# -----------------------------
# Build pipelines
# -----------------------------
def build_pipelines():
    nb = Pipeline([
        ("vect", CountVectorizer(max_features=15000, ngram_range=(1,2))),  # 1-2 үгийн нийлбэр → тоо
        ("clf", MultinomialNB())                                           # Naive Bayes classifier
    ])
    lr = Pipeline([
        ("vect", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),   # TF-IDF vectorization
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))       # Logistic Regression
    ])
    return {"NaiveBayes": nb, "LogisticRegression": lr}                     # 2 загварыг dict болгож буцаах

# -----------------------------
# Evaluate
# -----------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)                   # Урьдчилсан таамаг
    y_proba = model.predict_proba(X_test)[:,1]       # Positive class-ийн магадлал
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "acc": accuracy_score(y_test, y_pred),       # Accuracy тооцох
        "prec": precision_score(y_test, y_pred),     # Precision
        "rec": recall_score(y_test, y_pred),         # Recall
        "f1": f1_score(y_test, y_pred),              # F1 Score
        "auc": roc_auc_score(y_test, y_proba),       # AUC
        "cm": confusion_matrix(y_test, y_pred),      # Confusion matrix
        "report": classification_report(y_test, y_pred, digits=4)  # Дэлгэрэнгүй тайлан
    }

# -----------------------------
# Streamlit UI эхэлж байна
# -----------------------------
st.title("📊 Twitter Sentiment Classifier Dashboard")  
st.markdown("Upload CSV → Train → Compare models → Posterior + Prediction + Correctness + Metrics")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])    # CSV upload control
test_size = st.slider("Test size (%)", 10, 50, 30)              # Тестийн хувийг сонгох

if uploaded_file:
    df_raw = load_csv(uploaded_file)                             # CSV файлыг унших
    if df_raw is not None:
        st.subheader("CSV Preview")
        st.write(df_raw.head())                                  # Эхний 5 мөрийг харуулах

        df = prepare_df(df_raw)                                  # Текстийг цэвэрлэх
        if df is not None:
            st.subheader("Dataset Summary")
            st.write(df["target"].value_counts())                # Positive / Negative тоо

            X = df["clean_text"]                                 # Feature (текст)
            y = df["target"]                                     # Label (0/1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, stratify=y, random_state=42
            )

            st.subheader("Training Models...")
            models = build_pipelines()                           # 2 ML model
            results = {}                                         # Үр дүн хадгалах dict
            bar = st.progress(0)                                 # Progress bar

            # Загваруудыг сургаж, үнэлгээ авах
            for i, (name, model) in enumerate(models.items(), start=1):
                model.fit(X_train, y_train)                      # Загварыг сургах алгоритм хэрэгжүүлэлт
                results[name] = evaluate(model, X_test, y_test)  # Үнэлгээ хийх
                bar.progress(int(i/len(models)*100))             # Прогресс %

            st.success("✅ Training Done!")  # 🔹 Загвар сургалт бүрэн дууссаныг Streamlit дээр ногоон нотолгоогоор харуулна.

            # -----------------------------
            # Model Metrics Overview
            # -----------------------------
            st.subheader("📌 Model Metrics Overview")  # 🔹 Хоёр ангилагчийн (NB, LR) гол үзүүлэлтүүдийг харуулах гарчиг
            for name, r in results.items():            # 🔹 results dict доторх бүх моделиудын нэр (name) болон үр дүн (r)-г давталтаар авах
                st.markdown(f"### {name}", unsafe_allow_html=True)   # 🔹 Загварын нэрийг том гарчиг болгон хэвлэх

                # 4 metric-г багана болгож харуулах
                col1, col2, col3, col4 = st.columns(4)  # 🔹 4 багана бүхий layout үүсгэх (Accuracy, Precision, Recall, F1-г хажуудаа харагдуулна)
                col1.metric("Accuracy", f"{r['acc'] * 100:.2f}%")   # 🔹 Accuracy-г хувь болгон харуулах
                col2.metric("Precision", f"{r['prec'] * 100:.2f}%") # 🔹 Precision хувь
                col3.metric("Recall", f"{r['rec'] * 100:.2f}%")     # 🔹 Recall хувь
                col4.metric("F1 Score", f"{r['f1'] * 100:.2f}%")     # 🔹 F1-score хувь

            # -----------------------------
            # Posterior Table
            # -----------------------------
            st.subheader("Posterior Probability Table (First 10 Tweets)")  # 🔹 Эхний 10 өгөгдлийн posterior магадлалыг хүснэгтээр харуулна
            posterior_table = pd.DataFrame({"Tweet": X_test[:10], "True Label": y_test[:10]})  
            # 🔸 Test датаас эхний 10 бичлэгийг Tweet болон жинхэнэ Label хэлбэрээр авч DataFrame үүсгэж байна

            # Posterior value, Prediction value, Correct эсэхийг нэмэх
            for name, r in results.items():    # 🔹 Хоёр моделиудын үр дүнг давталтаар авах
                posterior_table[name+"_Posterior"] = r["y_proba"][:10]   # 🔹 Posterior магадлалын эхний 10 бичлэгийг нэмэх
                posterior_table[name+"_Prediction"] = r["y_pred"][:10]   # 🔹 Урьдчилсан ангилал (0 эсвэл 1)
                posterior_table[name+"_Correct?"] = r["y_pred"][:10] == y_test[:10]  # 🔹 Зөв таасан эсэхийг Boolean (True/False хэлбэрээр) нэмнэ

            st.dataframe(posterior_table, height=400)  # 🔹 Бэлэн болсон хүснэгтийг Streamlit-д харуулах

            # -----------------------------
            # Interactive Posterior View
            # -----------------------------
            st.subheader("🔎 Interactive Posterior + Prediction")  # 🔹 Сонгосон Tweet-ийн posterior-г интерактив харах UI гарчиг
            idx = st.slider("Select Tweet Index", 0, len(X_test)-1, 0)  
            # 🔹 Хэрэглэгч test өгөгдлийн индексийг сонгох слайдер (0-с эхлээд хамгийн сүүлийн Tweet хүртэл)

            st.write("Tweet:", X_test.iloc[idx])              # 🔹 Сонгосон индексийн Tweet-г харуулах
            st.write("Require True Label:", y_test.iloc[idx]) # 🔹 Зөв хариуг харуулах

            # Хоёр загварын posterior-г харуулах
            for name, r in results.items():               # 🔹 Моделиудыг давталтаар шалгах
                posterior = r["y_proba"][idx]            # 🔹 Сонгосон индексийн posterior probability
                pred = r["y_pred"][idx]                  # 🔹 Сонгосон индексийн prediction (0/1)
                correct = "✅ True" if pred == y_test.iloc[idx] else "❌ False"  
                # 🔹 Prediction зөв эсэхийг тэмдэглэгээтэй харуулна

                st.metric(label=f"{name} Posterior (+ probability)", value=f"{posterior * 100:.2f}%")  
                # 🔹 Posterior probability-г хувь хэлбэрээр харуулах
                st.write(f"{name} Prediction:", pred, "| Correct?", correct)  
                # 🔹 Prediction болон зөв эсэх

            # -----------------------------
            # Confusion matrices
            # -----------------------------
            st.subheader("🔎 Confusion Matrices (T/F row & column labels)")  
            # 🔹 Загвар тус бүрийн хүрээ матриц (Confusion Matrix)–г харуулах

            cols = st.columns(len(results))  # 🔹 Моделиудын тоотой тэнцүү хэмжээтэй баганыг UI-д үүсгэнэ (2 model → 2 column)

            for col, (name, r) in zip(cols, results.items()):  # 🔹 Багана бүрт нэг загвар байрлуулна
                with col:
                    st.markdown(f"### {name}")  # 🔹 Загварын нэр

                    cm_vals = r["cm"]           # 🔹 Confusion matrix утгуудыг авч байна

                    fig, ax = plt.subplots(figsize=(5, 4))  # 🔹 Heatmap зурах шинэ Figure үүсгэнэ
                    sns.heatmap(
                        cm_vals,                # 🔹 Confusion matrix-ийн тоон өгөгдөл
                        annot=True,             # 🔹 Тоонуудыг дотроо харуулна
                        fmt="d",                # 🔹 Тоонуудыг integer хэлбэрээр харуулах
                        cmap="YlGnBu",          # 🔹 Өнгөний схем
                        ax=ax,                  # 🔹 Хаана зурахыг зааж өгч байна
                        annot_kws={"size":12, "weight":"bold"}, # 🔹 Аннотацийн фонтын стиль
                        linewidths=1,           # 🔹 Хил шугамын өргөн
                        linecolor="white",      # 🔹 Хил шугамын өнгө
                        cbar=True               # 🔹 Color bar харуулах эсэх
                    )

                    ax.set_yticklabels(['T','F'], rotation=0)  # 🔹 Y тэнхлэгийн (Actual) тэмдэглэгээ: T=True, F=False
                    ax.set_xticklabels(['F','T'], rotation=0)  # 🔹 X тэнхлэгийн (Predicted) тэмдэглэгээ

                    ax.set_xlabel("Predicted")                 # 🔹 X тэнхлэгийн нэр
                    ax.set_ylabel("Actual")                   # 🔹 Y тэнхлэгийн нэр
                    ax.set_title(f"{name} Confusion Matrix (T/F)")  # 🔹 Графикийн гарчиг

                    st.pyplot(fig, use_container_width=True)  # 🔹 Streamlit-д графикийг хэвлэх

            # -----------------------------
            # 3D Posterior Histogram
            # -----------------------------
            st.subheader("🔎Posterior Probability Distribution (3D View)")  
            # 🔹 Posterior магадлалын тархалтыг 3D багана графикаар харуулах (загвар тус бүрт)

            cols = st.columns(len(results))  # 🔹 Моделиудын тоотой тэнцэх багана

            for col, (name, r) in zip(cols, results.items()):
                with col:
                    fig = plt.figure(figsize=(5,4))                 # 🔹 Шинэ Figure үүсгэнэ
                    ax = fig.add_subplot(111, projection='3d')      # 🔹 3D subplot үүсгэнэ

                    hist, bins = np.histogram(r["y_proba"], bins=25)  # 🔹 Posterior probability-г histogram болгох
                    xpos = (bins[:-1] + bins[1:]) / 2                # 🔹 Багануудын X байрлал
                    ypos = np.zeros_like(xpos)                       # 🔹 Y байрлал бүгд 0 (3D бар-нд dummy)
                    zpos = np.zeros_like(xpos)                       # 🔹 Багана Z эхлэл 0
                    dx = (bins[1]-bins[0]) * np.ones_like(xpos)      # 🔹 Багануудын өргөн (X)
                    dy = np.ones_like(xpos)                          # 🔹 Y зузаан
                    dz = hist                                        # 🔹 Багануудын өндөр

                    colors = cm.viridis(dz / dz.max())               # 🔹 Histogram өндөрт суурилсан өнгө

                    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, edgecolor='k')  
                    # 🔹 3D бар график зурж байна

                    ax.set_xlabel('Posterior Probability')  # 🔹 X тэнхлэгийн нэр
                    ax.set_ylabel('Y (dummy)')              # 🔹 Dummy Y тэнхлэг
                    ax.set_zlabel('Frequency')              # 🔹 Frequency буюу давтамж
                    ax.set_title(f"{name} Posterior Probability (3D)")  # 🔹 Гарчиг

                    ax.view_init(elev=30, azim=-60)  # 🔹 Камерын харах өнцөг тохируулах

                    st.pyplot(fig, use_container_width=True)  # 🔹 Streamlit-д хэвлэх

            # -----------------------------
            # Scatter Plots
            # -----------------------------
            st.subheader("Interactive Scatter Plots: Posterior Comparison & Correctness")  
            # 🔹 NB ба LR хоёрын posterior-г хооронд нь харьцуулах интерактив Plotly график

            df_scatter = pd.DataFrame({
                "NB_Proba": results["NaiveBayes"]["y_proba"],            # 🔹 NB posterior probability
                "LR_Proba": results["LogisticRegression"]["y_proba"],    # 🔹 LR posterior probability
                "True_Label": y_test.values,                             # 🔹 Жинхэнэ Label
                "Tweet": X_test.values,                                  # 🔹 Tweet текст
                "NB_Correct": results["NaiveBayes"]["y_pred"] == y_test, # 🔹 NB зөв таасан эсэх
                "LR_Correct": results["LogisticRegression"]["y_pred"] == y_test # 🔹 LR зөв таасан эсэх
            })

            cols = st.columns(2)  # 🔹 2 графикийг зэрэгцүүлэн харагдуулах хоёр багана

            # Posterior comparison chart
            with cols[0]:
                fig1 = px.scatter(
                    df_scatter,
                    x="NB_Proba",     # 🔹 NB posterior probability X тэнхлэг
                    y="LR_Proba",     # 🔹 LR posterior probability Y тэнхлэг
                    color="True_Label",   # 🔹 Жинхэнэ Label-ийг өнгөөр ялгана
                    hover_data={"Tweet": True},   # 🔹 Mouse-г tweet дээр авчрахад харуулах текст
                    title="Posterior Probability Comparison"  # 🔹 Гарчиг
                )
                st.plotly_chart(fig1, use_container_width=True)  # 🔹 Streamlit-д хэвлэх

            # Correctness comparison chart
            with cols[1]:
                df_scatter["Both_Correct"] = df_scatter["NB_Correct"] & df_scatter["LR_Correct"]  
                # 🔹 Хоёр загвар хоёулаа зөв байсан эсэхийг тооцох

                fig2 = px.scatter(
                    df_scatter,
                    x="NB_Proba",         # 🔹 NB posterior
                    y="LR_Proba",         # 🔹 LR posterior
                    color="Both_Correct", # 🔹 Хэрвээ NB болон LR хоёулаа зөв → True, үгүй бол False
                    hover_data={"Tweet": True},  # 🔹 Tweet текст харуулах
                    title="Correct vs Incorrect Predictions"  # 🔹 Графикийн нэр
                )
                st.plotly_chart(fig2, use_container_width=True)  # 🔹 Streamlit-д графикийг харуулах

        else:
                st.info("⏳ Upload CSV to train and see posterior probability, prediction, correctness, and metrics.")  
                # 🔹 Хэрвээ CSV upload хийгдээгүй бол мэдээллийн мессеж харуулах

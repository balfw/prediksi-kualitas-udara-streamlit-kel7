import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt # Diperlukan untuk visualisasi
import seaborn as sns
import plotly.express as px 

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kualitas Udara", page_icon="🌍", layout="wide")

# Fungsi untuk load model dengan cache agar aplikasi lebih cepat
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Sidebar
menu = st.sidebar.selectbox("Pilih Menu", ["Home", "EDA", "Klasifikasi", "Prediksi CSV Klasifikasi", "Prediksi CSV Regresi"])

# Load Dataset
df = pd.read_csv("DatasetPolutionScaling.csv")

# Halaman Home
if menu == "Home":
    st.title("🌍 Sistem Prediksi Kualitas Udara")
    st.markdown("""
    Selamat datang di aplikasi pemantau kualitas udara berbasis kecerdasan buatan. 
    Aplikasi ini dirancang untuk membantu memprediksi tingkat polusi udara berdasarkan 
    berbagai parameter lingkungan.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mengapa Kualitas Udara Penting?")
        st.write("""
        Kualitas udara yang buruk berdampak langsung pada kesehatan pernapasan, 
        kesehatan kardiovaskular, dan lingkungan hidup secara keseluruhan. 
        Dengan melakukan prediksi dini, kita dapat mengambil langkah mitigasi yang tepat.
        """)
    with col2:
        st.info("💡 **Catatan:** Input data yang dimasukkan harus sudah sesuai dengan skala yang digunakan dalam proses pelatihan model.")

    st.divider()
    st.subheader("Fitur Utama")
    st.write("- **EDA:** Eksplorasi data untuk memahami sebaran polutan.")
    st.write("- **Klasifikasi:** Menentukan kategori kualitas udara (Poor, Moderate, Good, Excellent).")
    st.write("- **Regresi:** Memprediksi nilai numerik konsentrasi PM2.5.")
    
    st.divider()

    st.subheader("🏆 Perbandingan Model Klasifikasi")

    performa = pd.DataFrame({
        "Model": [
            "KNN",
            "Decision Tree",
            "SVM",
            "Neural Network"
        ],
        "Accuracy": [
            0.86,
            0.90,
            0.93,
            0.92
        ],
        "Accuracy HPO": [
            0.83,
            0.91,
            0.89,
            0.89
        ]
    })
    

    st.dataframe(performa)

    st.bar_chart(
        performa.set_index("Model")
    )

    st.info("""
    Grafik ini digunakan untuk membandingkan performa
    setiap algoritma klasifikasi berdasarkan nilai akurasi.
    """)
    
    st.subheader("🏆 Perbandingan Model Regresi")

    performa = pd.DataFrame({
        "Model": [
            "KNN",
            "Decision Tree",
            "SVM",
            "Neural Network"
        ],
        "Accuracy": [
            0.98,
            0.-82,
            0.99,
            0.99
        ],
        "Accuracy HPO": [
            0.98,
            0.-83,
            0.99,
            0.99
        ]
    })
    

    st.dataframe(performa)

    st.bar_chart(
        performa.set_index("Model")
    )

    st.info("""
    Grafik ini digunakan untuk membandingkan performa
    setiap algoritma regresi berdasarkan nilai akurasi.
    """)

# Halaman EDA

elif menu == "EDA":

    st.title("📊 Exploratory Data Analysis")

    # Penjelasan Kolom Dataset
    st.subheader("📖 Penjelasan Kolom Dataset")

    kolom_df = pd.DataFrame({
        "Kolom": [
            "Air Quality",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO"
        ],
        "Deskripsi": [
            "Kategori kualitas udara",
            "Partikel halus ≤ 2.5 mikrometer",
            "Partikel halus ≤ 10 mikrometer",
            "Kadar Nitrogen Dioksida",
            "Kadar Sulfur Dioksida",
            "Kadar Karbon Monoksida"
        ]
    })

    st.dataframe(kolom_df)

    # Preview Dataset
    st.subheader("📋 Preview Dataset")

    with st.expander("Lihat 5 Data Pertama"):
        st.dataframe(df.head())

    st.info("""
    Dataset digunakan untuk proses klasifikasi dan regresi.
    """)

    # Distribusi Air Quality
    st.subheader("📈 Distribusi Air Quality")

    air_quality_counts = df["Air Quality"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    warna = [
        "#ef4444",  # merah
        "#f59e0b",  # orange
        "#22c55e",  # hijau
        "#3b82f6"   # biru
    ]

    bars = ax.bar(
        air_quality_counts.index.astype(str),
        air_quality_counts.values,
        color=warna
    )

    ax.set_xlabel("Kategori Air Quality")
    ax.set_ylabel("Jumlah Data")
    ax.set_title("Distribusi Kualitas Udara")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    st.info("""
    Grafik menunjukkan jumlah data pada masing-masing kategori kualitas udara.
    """)

    # Statistik Deskriptif
    st.subheader("📊 Statistik Deskriptif")

    st.dataframe(df.describe())

    st.info("""
    Statistik deskriptif digunakan untuk melihat nilai rata-rata,
    minimum, maksimum dan standar deviasi.
    """)

    # Heatmap
    st.subheader("🔥 Heatmap Korelasi")

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.info("""
    Heatmap digunakan untuk melihat hubungan antar variabel.
    """)

    # Scatter Plot
    st.subheader("📍 Scatter Plot PM2.5 vs PM10")

    fig2, ax2 = plt.subplots(figsize=(6, 3))

    ax2.scatter(
        df["PM2.5"],
        df["PM10"]
    )

    ax2.set_xlabel("PM2.5")
    ax2.set_ylabel("PM10")
    ax2.set_title("Hubungan PM2.5 dan PM10")

    st.pyplot(fig2)

    st.info("""
    Scatter plot digunakan untuk melihat pola hubungan
    antara PM2.5 dan PM10.
    """)

    # Kesimpulan
    st.subheader("📌 Kesimpulan EDA")

    st.success("""
    1. Dataset memiliki beberapa variabel polutan.
    2. Heatmap menunjukkan hubungan antar variabel.
    3. Scatter Plot menunjukkan hubungan PM2.5 dan PM10.
    4. Distribusi Air Quality menunjukkan sebaran kategori kualitas udara.
""")

# Halaman Klasifikasi
elif menu == "Klasifikasi":

    st.title("🤖 Klasifikasi Kualitas Udara")
    st.markdown("""
    ### 🌍 Apa itu Klasifikasi Kualitas Udara?

    Klasifikasi kualitas udara merupakan proses untuk menentukan kategori
    kualitas udara berdasarkan parameter lingkungan dan polutan yang ada.

    ### 🎯 Tujuan Klasifikasi

    Sistem ini menggunakan algoritma Machine Learning untuk memprediksi
    kategori kualitas udara berdasarkan data yang dimasukkan pengguna.

    Kategori yang digunakan:

    - 🔴 Poor (Buruk)
    - 🟠 Moderate (Sedang)
    - 🟢 Good (Baik)
    - 🔵 Excellent (Sangat Baik)

    Semakin baik kualitas udara maka semakin aman bagi kesehatan manusia
    dan lingkungan.
    """)

    model_map = {
        "Decision Tree HPO": "modelJb_DS-HPO.joblib",
        "SVM": "modelJb_SVM.joblib",
        "Neural Network": "modelJb_NN.joblib"
    }
    pilihan = st.selectbox(
        "Pilih Model Klasifikasi",
        list(model_map.keys())
    )
    

    model = load_model(model_map[pilihan])
    
    st.success(f"Model Klasifikasi: {pilihan}")

    st.subheader("📖 Penjelasan Variabel")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature")
        st.caption(
        "Suhu udara lingkungan. Suhu yang tinggi dapat mempercepat pembentukan polutan di atmosfer."
        )
        humidity = st.number_input("Humidity")
        st.caption(
        "Tingkat kelembaban udara (%). Kelembaban mempengaruhi penyebaran partikel polusi."
        )
        pm2_5 = st.number_input("PM2.5")
        st.caption(
        "Partikel udara berukuran ≤ 2.5 mikrometer. Semakin tinggi nilainya semakin buruk kualitas udara."
        )
        pm10 = st.number_input("PM10")
        st.caption(
        "Partikel udara berukuran ≤ 10 mikrometer. Semakin tinggi nilainya semakin buruk kualitas udara."
        )
        no2 = st.number_input("NO2")
        st.caption(
         "Nitrogen Dioxide berasal dari kendaraan bermotor dan aktivitas industri."
        )

    with col2:
        so2 = st.number_input("SO2")
        st.caption(
        " Sulfur Dioxide berasal dari pembakaran bahan bakar fosil dan aktivitas industri."
        )
        co = st.number_input("CO")
        st.caption(
        " Carbon Monoxide merupakan gas beracun yang banyak dihasilkan kendaraan bermotor."
        )
        
        proximity = st.number_input(
            "Proximity_to_Industrial_Areas"
        )
        st.caption(
        "Menunjukkan kedekatan lokasi dengan kawasan industri. Semakin dekat biasanya tingkat polusi lebih tinggi."
        )

    if st.button("🔍 Prediksi Kualitas Udara"):
        
        # Data input
        input_df = pd.DataFrame(
            [[temperature, humidity, pm2_5, pm10, no2, so2, co, proximity]],
            columns=[
                "Temperature",
                "Humidity",
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "Proximity_to_Industrial_Areas"
            ]
        )

        # Prediksi
        prediction = model.predict(input_df)[0]

        label = {
            0: "Poor",
            1: "Moderate",
            2: "Good",
            3: "Excellent"
        }

        prediction_label = label.get(int(prediction), prediction)

        # Menampilkan data input
        st.subheader("📋 Data Input")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # Hasil prediksi
        st.subheader("🎯 Hasil Prediksi")

        st.success(
            f"Hasil prediksi menggunakan model {pilihan}: {prediction_label}"
        )
        
        # Probabilitas
        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(input_df)[0]
            hasil_download = input_df.copy()

            hasil_download["Model"] = pilihan
            hasil_download["Hasil Prediksi"] = prediction_label

            hasil_download["Prob_Poor"] = proba[0]
            hasil_download["Prob_Moderate"] = proba[1]
            hasil_download["Prob_Good"] = proba[2]
            hasil_download["Prob_Excellent"] = proba[3]

            if hasattr(model, "classes_"):
                class_names = [
                    label.get(int(cls), str(cls))
                    for cls in model.classes_
                ]
            else:
                class_names = [
                    f"Class {i}"
                    for i in range(len(proba))
                ]

            proba_df = pd.DataFrame({
                "Kategori": class_names,
                "Probabilitas": proba
            })

            st.subheader("📊 Probabilitas Prediksi")

            st.dataframe(
                proba_df,
                use_container_width=True
            )

        else:
            st.info(
                "Model ini tidak mendukung probabilitas prediksi."
            )

        # File hasil download
        hasil_download = input_df.copy()

        hasil_download["Model"] = pilihan
        hasil_download["Hasil Prediksi"] = prediction_label

        csv = hasil_download.to_csv(index=False)

        st.download_button(
            label="Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
   


# Halaman Regresi
elif menu == "Regresi":

    st.title("📈 Prediksi PM2.5")
    st.markdown("""
    ### 🌫 Apa itu PM2.5?

    PM2.5 adalah partikel udara dengan ukuran kurang dari atau sama dengan
    2.5 mikrometer. Karena ukurannya sangat kecil, partikel ini dapat masuk
    ke dalam paru-paru bahkan aliran darah manusia.

    ### 🎯 Tujuan Prediksi

    Menu ini digunakan untuk memprediksi konsentrasi PM2.5 berdasarkan
    kondisi lingkungan seperti suhu, kelembaban, gas pencemar, dan
    kedekatan dengan kawasan industri.

    Semakin tinggi nilai PM2.5 maka kualitas udara semakin buruk dan
    berpotensi menyebabkan gangguan kesehatan.
    """)

    reg_map = {
    "SVM Regressor": "modelJb_SVMRegressor.joblib",
    "SVM Regressor HPO": "modelJb-HPO_SVMRegressor.joblib",
    "Neural Network Regressor": "modelJb_NNRegressor.joblib",
    }

    regresi_model = st.selectbox(
        "Pilih Model Regresi",
        list(reg_map.keys())
    )

    model_regresi = load_model(
        reg_map[regresi_model]
    )


    st.success(
        f"Model Regresi Aktif: {regresi_model}"
    )
    st.subheader("📖 Penjelasan Parameter")

    col1, col2 = st.columns(2)

    with col1:
        
        temperature = st.number_input("Temperature")
        st.caption(
        "Suhu udara lingkungan. Suhu yang tinggi dapat mempercepat pembentukan polutan di atmosfer."
        )
        humidity = st.number_input("Humidity")
        st.caption(
        "Tingkat kelembaban udara (%). Kelembaban mempengaruhi penyebaran partikel polusi."
        )
        pm10 = st.number_input("PM10")
        st.caption(
        " Partikel udara berukuran ≤ 10 mikrometer. Semakin tinggi nilainya semakin buruk kualitas udara."
        )
        no2 = st.number_input("NO2")
        st.caption(
         "Nitrogen Dioxide berasal dari kendaraan bermotor dan aktivitas industri."
        )

    with col2:
        so2 = st.number_input("SO2")
        st.caption(
        " Sulfur Dioxide berasal dari pembakaran bahan bakar fosil dan aktivitas industri."
        )
        co = st.number_input("CO")
        st.caption(
        " Carbon Monoxide merupakan gas beracun yang banyak dihasilkan kendaraan bermotor."
        )
        pm25 = st.number_input("PM2.5")
        st.caption(
        "Menunjukkan kedekatan lokasi dengan kawasan industri. Semakin dekat biasanya tingkat polusi lebih tinggi."
        )

    if st.button("📈 Prediksi PM2.5"):

        input_df = pd.DataFrame(
                [[
                    temperature,
                    humidity,
                    pm25,
                    pm10,
                    no2,
                    so2,
                    co
                ]],
                columns=[
                    "Temperature",
                    "Humidity",
                    "PM2.5",
                    "PM10",
                    "NO2",
                    "SO2",
                    "CO"
                ]


        )
        # st.write("Kolom input_df:")
        # st.write(list(input_df.columns))

        # st.write("Kolom model:")
        # st.write(list(model_regresi.feature_names_in_))
        hasil = model_regresi.predict(input_df)[0]

        st.subheader("📋 Data Input")
        st.dataframe(input_df)

        st.subheader("📈 Hasil Prediksi PM2.5")
        st.success(f"Hasil Prediksi PM2.5: {hasil:.2f}")

        # Metric Card
        st.metric(
            label="Nilai Prediksi PM2.5",
            value=f"{hasil:.2f}"
        )

        # Interpretasi
        st.subheader("📝 Interpretasi")

        if hasil <= 12:
            st.success("Kualitas udara baik (Good)")
        elif hasil <= 35.4:
            st.info("Kualitas udara sedang (Moderate)")
        elif hasil <= 55.4:
            st.warning("Kualitas udara tidak sehat bagi kelompok sensitif")
        elif hasil <= 150.4:
            st.warning("Kualitas udara tidak sehat")
        else:
            st.error("Kualitas udara sangat tidak sehat")

        # Download hasil
        hasil_download = input_df.copy()

        hasil_download["Model"] = regresi_model
        hasil_download["Prediksi_PM2.5"] = round(hasil, 2)

        csv = hasil_download.to_csv(index=False)

        st.download_button(
            label="⬇ Download Hasil Prediksi PM2.5",
            data=csv,
            file_name="hasil_prediksi_pm25.csv",
            mime="text/csv"
        )
# ==========================================
# HALAMAN PREDIKSI CSV Regresi 
# ==========================================
elif menu == "Prediksi CSV Regresi":

    st.title("📂 Prediksi CSV Regresi")

    reg_map = {
        "SVM Regressor": "modelJb_SVMRegressor.joblib",
        "SVM Regressor HPO": "modelJb_SVMRegressor-HPO.joblib",
        "Neural Network Regressor": "modelJb_NNRegressor.joblib"
    }

    regresi_model = st.selectbox(
        "Pilih Model Regresi",
        list(reg_map.keys())
    )

    model = load_model(
        reg_map[regresi_model]
    )

    st.info("""
        Dataset harus memiliki kolom:

        Temperature,
        Humidity,
        PM10,
        NO2,
        SO2,
        CO,
        Proximity_to_Industrial_Areas
        """)

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("📋 Preview Dataset")
        st.dataframe(data.head())

        st.write(
            f"Jumlah Data : {len(data)}"
        )

        if st.button("📈 Prediksi Semua Data"):

            fitur_model = [
                "Temperature",
                "Humidity",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "Proximity_to_Industrial_Areas"
            ]

            if not all(
                col in data.columns
                for col in fitur_model
            ):
                st.error(
                    "Kolom dataset tidak sesuai."
                )
                st.stop()

            data_prediksi = data[
                fitur_model
            ]

            hasil = model.predict(
                data_prediksi
            )

            hasil_df = data.copy()

            hasil_df["Prediksi_PM2.5"] = hasil

            st.session_state.hasil_regresi = hasil_df 
            if "hasil_regresi" not in st.session_state:
                st.session_state.hasil_regresi = None

                if st.session_state.hasil_regresi is not None:

                    st.success(
                    "✅ Prediksi berhasil"
                )

                st.subheader(
                    "📊 Hasil Prediksi PM2.5"
                )

                st.dataframe(
                    st.session_state.hasil_regresi,
                    use_container_width=True
                )
                st.subheader("📈 Ringkasan Prediksi")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Jumlah Data",
                        len(st.session_state.hasil_regresi)
                    )

                with col2:
                    st.metric(
                        "Rata-rata PM2.5",
                        round(
                            st.session_state.hasil_regresi[
                                "Prediksi_PM2.5"
                            ].mean(),
                            2
                        )
                    )

                with col3:
                    st.metric(
                        "PM2.5 Maksimum",
                        round(
                            st.session_state.hasil_regresi[
                                "Prediksi_PM2.5"
                            ].max(),
                            2
                        )
                    )
                    
                    csv = (
                        st.session_state.hasil_regresi
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        label="💾 Simpan Hasil Prediksi",
                        data=csv,
                        file_name="hasil_regresi_pm25.csv",
                        mime="text/csv"
                    )

                    fig, ax = plt.subplots(figsize=(8,4))

                    ax.hist(
                        st.session_state.hasil_regresi[
                            "Prediksi_PM2.5"
                        ],
                        bins=20
                    )

                    ax.set_title(
                        "Distribusi Prediksi PM2.5"
                    )

                    st.pyplot(fig)

# ==========================================
# HALAMAN PREDIKSI CSV KLASIFIKASI
# ==========================================

elif menu == "Prediksi CSV Klasifikasi":

    if "hasil_klasifikasi" not in st.session_state:
        st.session_state["hasil_klasifikasi"] = None

    st.title("📂 Prediksi CSV Klasifikasi")

    model_map = {
        "Decision Tree HPO": "modelJb_DS-HPO.joblib",
        "SVM": "modelJb_SVM.joblib",
        "Neural Network": "modelJb_NN.joblib"
    }

    pilihan = st.selectbox(
        "Pilih Model Klasifikasi",
        list(model_map.keys())
    )

    model = load_model(model_map[pilihan])

    st.info("""
    Dataset harus memiliki kolom:

    Temperature,
    Humidity,
    PM2.5,
    PM10,
    NO2,
    SO2,
    CO,
    Proximity_to_Industrial_Areas
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("📋 Preview Dataset")
        st.dataframe(data.head())

        st.write(f"Jumlah Data : {len(data)}")

        if st.button("🤖 Prediksi Semua Data"):

            if "Air Quality" in data.columns:
                data = data.drop(columns=["Air Quality"])

            if "Population_Density" in data.columns:
                data = data.drop(columns=["Population_Density"])

            fitur_model = [
                "Temperature",
                "Humidity",
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "Proximity_to_Industrial_Areas"
            ]

            if not all(col in data.columns for col in fitur_model):
                st.error("Kolom dataset tidak sesuai.")
                st.stop()

            data_prediksi = data[fitur_model]

            hasil = model.predict(data_prediksi)

            label = {
                0: "Poor",
                1: "Moderate",
                2: "Good",
                3: "Excellent"
            }

            hasil_df = data.copy()

            hasil_df["Prediksi_Air_Quality"] = [
                label.get(int(x), str(x))
                for x in hasil
            ]

            st.session_state["hasil_klasifikasi"] = hasil_df

    # ==================================
    # TAMPILKAN HASIL
    # ==================================

    if st.session_state["hasil_klasifikasi"] is not None:

        st.success("✅ Prediksi berhasil")

        st.subheader("📊 Hasil Prediksi")

        st.dataframe(
            st.session_state["hasil_klasifikasi"],
            use_container_width=True
        )

        st.subheader("📈 Distribusi Hasil Prediksi")

        summary = (
            st.session_state["hasil_klasifikasi"]
            ["Prediksi_Air_Quality"]
            .value_counts()
        )

        summary_df = summary.reset_index()
        summary_df.columns = [
            "Kategori",
            "Jumlah"
        ]

        fig = px.bar(
            summary_df,
            x="Kategori",
            y="Jumlah",
            color="Kategori",
            text="Jumlah",
            color_discrete_map={
                "Poor": "#FF4B4B",
                "Moderate": "#FFA500",
                "Good": "#00CC66",
                "Excellent": "#3399FF"
            }
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Kategori",
            yaxis_title="Jumlah Data",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Jumlah Data",
                len(st.session_state["hasil_klasifikasi"])
            )

        with col2:
            st.metric(
                "Kategori Terbanyak",
                summary.idxmax()
            )

        with col3:
            st.metric(
                "Jumlah Kategori Terbanyak",
                summary.max()
            )

        csv = (
            st.session_state["hasil_klasifikasi"]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="💾 Simpan Hasil Prediksi",
            data=csv,
            file_name="hasil_klasifikasi.csv",
            mime="text/csv"
        )

        if st.button("➕ Prediksi Baru"):
            st.session_state["hasil_klasifikasi"] = None
            st.rerun()
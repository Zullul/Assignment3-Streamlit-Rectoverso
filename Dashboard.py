import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Judul dashboard
st.title("📊 Dashboard Sampah Organik & Anorganik")

# Deskripsi awal
st.markdown("""
Dashboard ini memberikan informasi mengenai jenis-jenis sampah organik dan anorganik, serta membantu dalam memantau data pengelolaan sampah secara sederhana dan edukatif.
""")

# Informasi singkat + penjelasan dampak
with st.expander("ℹ️ Apa itu Sampah Organik dan Anorganik?"):
    st.markdown("""
    ### ♻️ Sampah Organik
    Sampah organik adalah jenis sampah yang **berasal dari makhluk hidup** dan **dapat terurai secara alami** oleh mikroorganisme tanpa mencemari lingkungan.  
    Contohnya:
    - Sisa makanan
    - Daun kering
    - Kulit buah
    - Ranting pohon

    **Dampak Positif**: Jika dikelola dengan baik, sampah organik bisa dijadikan **kompos** atau **pupuk alami** untuk tanaman.

    ---

    ### 🧴 Sampah Anorganik
    Sampah anorganik adalah sampah yang **tidak mudah terurai secara alami** dan biasanya berasal dari bahan buatan manusia atau industri.  
    Contohnya:
    - Plastik
    - Kaleng
    - Botol kaca
    - Styrofoam

    **Dampak Negatif**: Sampah anorganik bisa bertahan **ratusan tahun** di lingkungan, mencemari tanah, air, dan membahayakan hewan. Pengelolaan yang bijak seperti **daur ulang** sangat penting untuk mengurangi dampak ini.
    """)



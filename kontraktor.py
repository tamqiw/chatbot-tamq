import streamlit as st
import openai
from PyPDF2 import PdfReader

# Fungsi untuk mengekstrak teks dari file PDF yang diunggah
def get_pdf_text(pdf_docs):
    """
    Membaca dan mengekstrak teks dari setiap halaman file PDF.

    Args:
        pdf_docs: File PDF yang diunggah melalui Streamlit.

    Returns:
        String berisi gabungan teks dari semua halaman PDF.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        st.error(f"Gagal membaca file PDF: {e}")
        return None
    return text

# Fungsi untuk menganalisis teks menggunakan OpenAI API
def analyze_contract_with_openai(api_key, contract_text):
    """
    Mengirimkan teks kontrak ke OpenAI API untuk dianalisis.

    Args:
        api_key (str): Kunci API OpenAI.
        contract_text (str): Teks dari dokumen kontrak.

    Returns:
        String berisi hasil analisis dari model AI.
    """
    try:
        # Menginisialisasi client OpenAI dengan API key yang diberikan
        client = openai.OpenAI(api_key=api_key)
        
        # Template prompt yang lebih terstruktur untuk analisis kontrak
        prompt_template = f"""
        Anda adalah seorang asisten hukum AI yang ahli dalam menganalisis dokumen kontrak.
        Tugas Anda adalah untuk menganalisis teks kontrak berikut dan memberikan ringkasan yang jelas dan terstruktur.

        Fokus pada poin-poin kunci berikut:
        1.  **Para Pihak yang Terlibat**: Identifikasi semua pihak yang disebutkan dalam kontrak.
        2.  **Tanggal Efektif dan Durasi Kontrak**: Tentukan kapan kontrak mulai berlaku dan kapan berakhir.
        3.  **Klausul Pembayaran**: Ringkas ketentuan pembayaran, termasuk jumlah, jadwal, dan metode pembayaran.
        4.  **Kewajiban Utama**: Jelaskan kewajiban utama dari masing-masing pihak.
        5.  **Klausul Kerahasiaan**: Ringkas kewajiban terkait kerahasiaan informasi.
        6.  **Klausul Terminasi**: Jelaskan kondisi di mana kontrak dapat diakhiri oleh salah satu atau kedua belah pihak.
        7.  **Potensi Risiko atau Klausul Tidak Biasa**: Identifikasi setiap klausul yang mungkin berisiko, ambigu, atau tidak standar yang perlu perhatian khusus.

        Sajikan hasil analisis dalam format yang mudah dibaca menggunakan poin-poin atau sub-judul.

        Berikut adalah teks kontrak yang perlu dianalisis:
        ---
        {contract_text}
        ---
        """

        # Melakukan panggilan ke API OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # Menggunakan model yang kuat seperti gpt-4o atau gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "Anda adalah asisten hukum yang membantu menganalisis kontrak."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.3, # Mengurangi kreativitas untuk hasil yang lebih faktual
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        st.error("API Key OpenAI tidak valid atau salah. Silakan periksa kembali.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghubungi API OpenAI: {e}")
        return None


# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Analisis Kontrak AI", layout="wide")

# --- Sidebar untuk Input API Key ---
with st.sidebar:
    st.header("Pengaturan")
    openai_api_key = st.text_input(
        "Masukkan OpenAI API Key Anda", 
        type="password",
        help="Dapatkan API key Anda dari [platform.openai.com](https://platform.openai.com/account/api-keys)"
    )
    st.markdown("---")
    st.info("Aplikasi ini menggunakan model AI dari OpenAI untuk menganalisis dokumen kontrak Anda.")

# --- Halaman Utama Aplikasi ---
st.title("📄 Analisis Dokumen Kontrak dengan AI")
st.markdown("Unggah dokumen kontrak Anda dalam format PDF untuk mendapatkan ringkasan dan analisis poin-poin penting.")

# Komponen untuk mengunggah file PDF
uploaded_file = st.file_uploader("Pilih file PDF...", type="pdf")

# Tombol untuk memulai analisis
if st.button("Analisis Dokumen"):
    # Validasi input sebelum memproses
    if uploaded_file is not None:
        if openai_api_key:
            with st.spinner("Harap tunggu, AI sedang membaca dan menganalisis dokumen Anda..."):
                # Langkah 1: Ekstrak teks dari PDF
                raw_text = get_pdf_text(uploaded_file)
                
                if raw_text:
                    st.info(f"Ekstraksi teks berhasil. Total {len(raw_text)} karakter ditemukan.")
                    
                    # Langkah 2: Analisis teks dengan OpenAI
                    analysis_result = analyze_contract_with_openai(openai_api_key, raw_text)
                    
                    # Langkah 3: Tampilkan hasil
                    if analysis_result:
                        st.subheader("Hasil Analisis Kontrak")
                        st.markdown(analysis_result)
        else:
            st.warning("Harap masukkan OpenAI API Key Anda di sidebar sebelah kiri.")
    else:
        st.warning("Harap unggah file PDF terlebih dahulu.")

import streamlit as st
import whisper
from pyannote.audio import Pipeline
import torch
import os
import json
import asyncio
from langchain_community.chat_models import ChatOpenAI
from pyannote.audio.pipelines.utils.hook import ProgressHook
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
import random
from markdown_pdf import MarkdownPdf, Section
import os
import uuid

# Fix permissions
os.environ["STREAMLIT_HOME"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp"

# Fix asyncio loop on Spaces
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- API Keys from Environment ---
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- Whisper + PyAnnote Initialization ---
@st.cache_resource
def initialize_models(model_type="medium"):
    asr_model = whisper.load_model(model_type).to("cpu")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    return asr_model, diarization_pipeline

def align_segments(transcript, diarization):
    aligned = []
    for seg in transcript.get("segments", []):
        start, end, text = seg["start"], seg["end"], seg["text"].strip()
        overlaps = [
            (min(end, turn.end) - max(start, turn.start), spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
            if turn.start < end and turn.end > start
        ]
        speaker = max(overlaps, key=lambda x: x[0])[1] if overlaps else "Unknown"
        aligned.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return aligned

class ChatOpenRouter(ChatOpenAI):
    def __init__(self, model_name, openai_api_key=os.environ["OPENROUTER_API_KEY"], openai_api_base="https://openrouter.ai/api/v1", **kwargs):
        super().__init__(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model_name=model_name, **kwargs)

def enhance_with_llm(aligned_output, model_name="qwen/qwen3-4b:free"):
    """
    Kirim aligned segments ke LLM untuk koreksi grammar.
    Output dijamin JSON. Hasil mentah ditampilkan dan disimpan untuk debugging.
    """
    input_json = json.dumps(aligned_output, ensure_ascii=False)

    example_user = '''[
  {{ "speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "anak anak sedang bermain di luar ruamah" }},
  {{ "speaker": "SPEAKER_01", "start": 2.5, "end": 5.0, "text": "iya mereka kayaknya sangat senang" }}
]'''

    example_assistant = '''[
  {{ "speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Anak-anak sedang bermain di luar rumah." }},
  {{ "speaker": "SPEAKER_01", "start": 2.5, "end": 5.0, "text": "Iya, mereka sepertinya sangat senang." }}
]'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Anda adalah asisten AI yang ahli dalam tata bahasa Indonesia dan transkripsi audio. Dalam konteks ini, anda bertugas untuk merapikan hasil transkripsi gelar perkara polisi Indonesia.

Tugas Anda:
1. Perbaiki kesalahan tata bahasa dan ejaan di setiap objek.
2. Jangan ubah atau hilangkan label pembicara ataupun timestamp. Jangan ubah label `speaker`, `start`, atau `end`.
3. Gunakan istilah hukum yang sesuai (contoh: "tersangka", "barang bukti", "penyidik", dsb.).
4. Cukup perbaiki teks agar lebih alami dalam Bahasa Indonesia baku.
5. Jangan beri penjelasan, komentar, atau format lain.
6. Perbaiki hanya bagian `text`.
7. Format output HARUS berupa array JSON murni. **Kembalikan HANYA** JSON array seperti ini:

Contoh format:
[
  {{
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 2.5,
    "text": "Teks hasil koreksi."
  }},
  ...
]
"""),
        ("user", example_user),
        ("assistant", example_assistant),
        ("user", "{input}")
    ])

    # Jalankan LLM
    llm = ChatOpenRouter(model_name=model_name, temperature=0)
    result = (prompt | llm).invoke({"input": input_json})
    raw_output = result.content

    # Simpan hasil mentah ke file untuk debugging
    with open("llm_raw_response.json", "w", encoding="utf-8") as f:
        f.write(raw_output)

    return raw_output

# --- Berita Acara Prompt ---
def generate_nomor_berita_acara():
    return f"BA-{datetime.now().year}-{random.randint(1000,9999)}"

def format_tanggal_formal(dt):
    bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    return f"{dt.day} {bulan[dt.month - 1]} {dt.year}"

berita_acara_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Anda adalah notulis resmi dalam gelar perkara kepolisian Republik Indonesia,
yang harus menyusun berita acara berdasarkan Peraturan Kapolri Nomor 14 Tahun 2012
tentang Manajemen Penyidikan Tindak Pidana.

Tugas Anda adalah menyusun **berita acara gelar perkara** berdasarkan transkrip dialog berikut.
Berita acara harus ditulis dalam **Bahasa Indonesia yang formal dan sesuai struktur berikut:**

1. **Identitas Perkara dan Pihak yang Hadir**: Tulis deskripsi umum tentang kasus dan pihak-pihak yang terlibat.
2. **Waktu dan Tempat Pelaksanaan**: Tuliskan waktu dan lokasi gelar perkara (boleh dibuat fiktif jika tidak ada dalam data).
3. **Uraian Singkat Kasus**: Jelaskan pokok perkara yang dibahas dalam gelar perkara.
4. **Paparan Hasil Penyidikan oleh Penyidik**: Uraikan penjelasan dari penyidik.
5. **Tanggapan dan Masukan Peserta**: Ringkas komentar, pertanyaan, dan diskusi dari peserta.
6. **Kesimpulan dan Rekomendasi**: Nyatakan hasil akhir dari gelar perkara, misalnya "perkara ditingkatkan ke tahap penyidikan", "tersangka ditetapkan", dll.
7. **Penutup**: Tuliskan bahwa berita acara ini dibuat sebagai bagian dari administrasi penyidikan.

**Tambahkan di awal dokumen:**
- Judul: **Berita Acara Gelar Perkara**
- Nomor: {nomor_berita_acara}
- Tanggal: {tanggal}

Catatan penting:
- Gunakan Bahasa Indonesia yang formal dan sesuai format dokumen resmi.
- Hilangkan label speaker, ubah menjadi "Penyidik", "Pelapor", dll bila bisa disimpulkan.
- Jika tidak diketahui, gunakan penalaran wajar dari konteks.
- Jangan tulis ulang label, timestamp, atau format JSON. Jawaban harus langsung dalam bentuk Markdown yang bersih dan siap dipublikasikan.
"""),
    ("user", "{input}")
])

def generate_berita_acara(aligned_segments, model_name="qwen/qwen3-4b:free"):
    nomor = generate_nomor_berita_acara()
    tanggal = format_tanggal_formal(datetime.now())
    input_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in aligned_segments])
    llm = ChatOpenRouter(model_name=model_name, temperature=0.3)
    return (berita_acara_prompt.partial(nomor_berita_acara=nomor, tanggal=tanggal) | llm).invoke({"input": input_text}).content

# === PROCESS AUDIO ===
def process_audio(audio_path, asr_model, diarization_pipeline):
    transcript = asr_model.transcribe(
        audio_path,
        language="id",
        word_timestamps=True,
        initial_prompt=(
            "Rekaman ini berasal dari proses gelar perkara oleh kepolisian Indonesia. "
            "Harap transkripsi dalam Bahasa Indonesia formal. "
            "Istilah-istilah seperti tersangka, saksi, barang bukti, pasal, dan laporan polisi harus dikenali."
        ),
        verbose=False
    )
    with ProgressHook() as hook:
        try:
            diarization = diarization_pipeline(audio_path, hook=hook)
        except Exception as e:
            st.error("❌ Diarization failed.")
            st.exception(e)
            return transcript, None

# --- Streamlit UI ---
st.title("🎙️ Audio Transcription & Diarization with LLM Polishing")
st.write("Upload an audio file and get a polished, speaker-labeled transcript in Bahasa Indonesia.")

uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac"])
if uploaded:
    audio_id = uuid.uuid4().hex
    audio_path = f"/tmp/audio_{audio_id}.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded.read())
    
    asr_model, diar_pipeline = initialize_models()
    if st.button("Transcribe & Polish"):
        with st.spinner("Transcribing and diarizing..."):
            transcript, diarization = process_audio(audio_path, asr_model, diar_pipeline)
            aligned = align_segments(transcript, diarization)
            st.session_state["aligned"] = aligned

        with st.spinner("Polishing..."):
          try:
              polished = enhance_with_llm(aligned)
              st.code(polished, language="json")  # tampilkan output mentah dari LLM
              entries = json.loads(polished)
              st.session_state["entries"] = entries
          except json.JSONDecodeError:
              st.error("❌ Gagal parsing output JSON dari LLM.")
              st.text_area("Respon mentah dari LLM:", polished, height=250)
          except Exception as e:
              st.error("❌ Error saat menjalankan LLM.")
              st.exception(e)
    st.success(f"File successfully uploaded and saved as {audio_path}")

# --- Display Polished Transcript ---
if "entries" in st.session_state:
    entries = st.session_state["entries"]
    st.header("📄 Polished Transcript (JSON)")
    st.json(entries)

    st.header("🗂️ Structured Segments")
    for e in entries:
        st.markdown(f"**[{e['speaker']} | {e['start']:.1f}-{e['end']:.1f}]**  \n{e['text']}")

    if st.button("📝 Generate Berita Acara"):
        with st.spinner("Generating..."):
            st.session_state["berita_acara_md"] = generate_berita_acara(entries)

# --- Display & Download Berita Acara ---
if "berita_acara_md" in st.session_state:
    md = st.session_state["berita_acara_md"]
    st.header("📄 Berita Acara (Markdown)")
    st.markdown(md)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"berita_acara_{timestamp}.md"
    pdf_filename = f"berita_acara_{timestamp}.pdf"

    # Save Markdown
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(md)

    # Convert to PDF
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md))
    pdf.meta["title"] = "Berita Acara Gelar Perkara"
    pdf.save(pdf_filename)

    # Download buttons
    with open(md_filename, "rb") as f:
        st.download_button("⬇️ Download Markdown", data=f, file_name=md_filename, mime="text/markdown")
    with open(pdf_filename, "rb") as f:
        st.download_button("⬇️ Download PDF", data=f, file_name=pdf_filename, mime="application/pdf")
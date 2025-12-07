import re
import unicodedata
from flask import Flask, request, jsonify
import joblib
from gensim.models import FastText
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys


app = Flask(__name__)
CORS(app)  # Agar bisa diakses dari berbagai origin jika diperlukan


MAX_LENGTH = 50
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

print("Sedang memuat model... Mohon tunggu...")
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # B. Load Model 1D-CNN (.keras)
    model = tf.keras.models.load_model('model_cnn_judol.keras')
    nb_model = joblib.load('model_nb_final.joblib')
    rf_model = joblib.load('model_rf_final.joblib')
    ft_model = FastText.load('model_fasttext.bin')
    nb_bigrams = joblib.load('model_nb_bigram_final.joblib')
    print("✅ Model berhasil dimuat! Server siap.")
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    sys.exit(1)

def smart_cleaning(text):
    text = str(text)
    # Normalisasi Unicode (Ubah font aneh jadi huruf standar)
    text = unicodedata.normalize('NFKD', text)
    # Hapus sisa karakter non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Lowercase
    text = text.lower()
    # Hapus angka
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus Mention & Hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus Tanda Baca (Tapi pertahankan angka & huruf)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Hapus Spasi Berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def vectorize_text(token_list, model):
    vectors = [model.wv[word] for word in token_list if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# --- 3. ENDPOINT API ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menerima data JSON dari PHP
        data = request.get_json()
        kalimat = data.get('kalimat', '')

        if not kalimat:
            return jsonify({'error': 'Kalimat tidak boleh kosong'}), 400

        sequences = tokenizer.texts_to_sequences([kalimat])

        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
        prediksi_probabilitas = model.predict(padded)[0][0]
        # Ambang batas (Threshold): 0.5
        if prediksi_probabilitas > 0.5:
            label = 'spam'
            confidence = f"{prediksi_probabilitas * 100:.2f}%"
        else:
            label = 'ham'
            confidence = f"{(1 - prediksi_probabilitas) * 100:.2f}%"


        print('label : ', label )
        print('confidence : ', confidence)
        # --- 6. KIRIM HASIL KE PHP/EXTENSION ---
        return jsonify({
            'status': 'success',
            'kalimat': kalimat,
            'prediksi': label,  # 'spam' atau 'ham'
            'skor_keyakinan': confidence,
            'model_used': '1D-CNN Deep Learning'
        })

        # Preprocessing & Prediksi
#        tokens = kalimat.lower().split()
 #       vektor_input = vectorize_text(tokens, ft_model)
       # vektor_input = vektor_input.reshape(1, -1)
        print(smart_cleaning(kalimat))
      #  prediksi_label = nb_model.predict([smart_cleaning(kalimat)])[0]

        # Konversi hasil ke string yang mudah dibaca
        # Asumsi: label 1 = spam, 0 = ham (sesuaikan dengan training Anda)
        #hasil = 'spam' if prediksi_probabilitas == 1 or prediksi_probabilitas == 'spam' else 'ham'

        return jsonify({
            'status': 'success',
            'kalimat': kalimat,
            'prediksi': hasil
        })

    except Exception as e:
        print(f"Error prediksi: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Jalankan server di port 5000
    app.run(port=5000, debug=True)
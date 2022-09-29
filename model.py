# menyimpan dan membaca data ke dalam atau dari suatu file berformat .pkl
import pickle

# global variable
global model, scaler

def load():
    global model, scaler
    model = pickle.load(open('model/model_ds.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler_ds.pkl', 'rb'))

def prediksi(data):
    # transform(): parameter yang dihasilkan dari metode fit(), diterapkan pada model untuk menghasilkan kumpulan data yang diubah.
    data = scaler.transform(data)
    # metode yang dapat dieksekusi pada model terlatih untuk memprediksi label (atau kelas) aktual pada kumpulan data baru.
    prediksi = int(model.predict(data))
    # Metode menerima argumen tunggal yang sesuai dengan data di mana probabilitas akan dihitung dan mengembalikan larik daftar yang berisi probabilitas kelas untuk titik data input.
    nilai_kepercayaan = model.predict_proba(data).flatten()
    nilai_kepercayaan = max(nilai_kepercayaan) * 100
    nilai_kepercayaan = round(nilai_kepercayaan)

    if prediksi == 0:
        hasil_prediksi = "Tidak Resign"
    else:
        hasil_prediksi = "Akan Resign"
    return hasil_prediksi, nilai_kepercayaan
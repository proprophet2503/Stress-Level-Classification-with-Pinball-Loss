# Klasifikasi Stress Mahasiswa Menggunakan SVM dengan Flexible Pinball Loss

## Deskripsi
Proyek ini bertujuan untuk mengklasifikasikan tipe stress mahasiswa (Eustress, Distress, No Stress) menggunakan Support Vector Machine (SVM) yang dimodifikasi dengan Flexible Pinball Loss Function sebagai pengganti hinge loss.

Flexible Pinball Loss digunakan agar model:
- lebih robust terhadap outlier,
- mampu menangani ketidakseimbangan kelas,
- dan memberikan performa lebih stabil pada data survei berbasis Likert.

Dataset yang digunakan merupakan hasil survei nasional mengenai faktor penyebab stres mahasiswa yang mencakup aspek psikologis, fisiologis, akademik, lingkungan, hingga sosial.
---
## Sumber Paper : https://www.sciencedirect.com/science/article/abs/pii/S156849462400228X
## Sumber Dataset : https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets

## 🧠 Metodologi

### 1. **Preprocessing & Feature Engineering**
### 2. **Implementasi Flexible Pinball Loss**
- τ tinggi → penalti lebih besar untuk underestimation
- τ rendah → penalti lebih besar untuk overestimation

### 3. **Klasifikasi Stress Mahasiswa**
- No Stress
- Eustress
- Distress
---

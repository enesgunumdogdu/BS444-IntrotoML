# BS444-Introduction to Machine Learning Dersi Vize Projesi

Bu proje, insan kanındaki mikrobiyom verilerinin kanser teşhisi için bir gösterge olabileceğini incelemektedir.

## Data
4 farklı kanser türüne sahip 355 kişinin kan örneği verileri bulunmaktadır: Kolon kanseri, meme kanseri, akciğer kanseri ve prostat kanseri.
Etiket dosyası olan labels.csv, her bir kişinin örnek adını ve hastalık türünü içermektedir. Veriler data.csv dosyasında saklanmaktadır.
Her bir satır, karşılık gelen kişinin örneğinin adını ve geri kalan kısmı her bir mikroorganizma tipine (virüs veya bakteri) ait olan DNA parçacığı sayısını içermektedir. 
Toplamda 1836 farklı mikroorganizma özelliği bulunmaktadır.

## Geliştirme Ortamı
Python 3.11.0
Visual Studio Code
Makine öğrenme algoritması olarak Random Forest Classifier kullanıldı.
Arayüz için Tkinter kütüphanesi kullanıldı.

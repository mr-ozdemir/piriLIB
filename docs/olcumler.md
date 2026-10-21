# Ölçümler

Alfa testinde karşılaştırmak için geliştirme sırasında alınan ölçümler. Hepsi
sentetik veriyle, Windows 10 ve Python 3.14.3 üzerinde, tek çekirdekle alındı.

## Değerlendirici (Adım 2)

| Problem | Sefer | `evaluate` | `check_plan` (saf Python) |
|---|---|---|---|
| Örnek (`linehaul_mini`) | 20 | 0,2 ms/çağrı | - |
| 81 merkez, 30 gün, 40.500 talep | 51.191 | 22 ms | 1,15 sn |

## ACO hızı (Adım 3 ve 4)

81 merkez, rastgele koordinatlar, günde 1.350 talep, 3 spot araç tipi, varsayılan ayarlar (16 karınca, en çok 60 iterasyon, sabır 20).

| Sürüm | 1 gün | 3 gün | 7 gün |
|---|---|---|---|
| İlk sürüm (alt problemler sırayla) | 47,5 sn | - | - |
| Alt problemler yan yana (gruplu) | 5,2 sn | 14,9 sn | 45,1 sn |
| Sabit ağırlıklar önceden | - | 11,3 sn | - |
| Yerel arama ile | 9,2 sn | 24,9 sn | - |

Tam boyut, yerel arama ile: 30 gün, 2.430 alt problem, 40.500 talep: **381 sn**
(32.743 sefer; kalan ihlaller: minimum doluluk 377,9 desi ve taşınmayan 346,7 desi,
amaç fonksiyonu gereği bırakılan küçük talepler).

## ACO kalitesi

Doğrudan sevkiyata (her hat için dolu tırlar, artanı sığan en ucuz araç) göre, 1 gün, 81 merkez:

| Sürüm | Toplam maliyet | Tasarruf |
|---|---|---|
| Doğrudan sevkiyat | 43.206.124 | - |
| ACO | 38.074.260 | %11,9 |
| ACO + yerel arama | 35.496.812 | %17,8 |

Kesin çözücüye (HiGHS MILP) göre, 145 küçük rastgele problem (3-7 merkez, bölünebilir talep, limitsiz spot), amaç = maliyet + 1e6 x ihlaller:

| Sürüm | Optimumu bulma | İkisi de uygunken ortalama maliyet farkı | p90 | En kötü |
|---|---|---|---|---|
| ACO | %63 | %0,24 | %0,56 | %5,27 |
| ACO + yerel arama | %80 | %0,13 | %0,40 | %2,95 |
| Melez (`exact_nodes=9`) | %100 | %0 | %0 | %0 |

Kalan farkların çoğu iki boyutlu kapasitede (desi ve kg birlikte) minimum doluluğu tutturmak için yükleri farklı varışlarla karıştırmayı gerektiren durumlar.

## ABC ayarlayıcı (Adım 5)

Test fonksiyonları, birim küp, 150 değerlendirme, 20 tohum, medyan en iyi değer (küçük daha iyi):

| Fonksiyon | Boyut | ABC | Rastgele arama | ABC'nin daha iyi olduğu tohumlar |
|---|---|---|---|---|
| Sphere | 3 | 0,0006 | 0,0102 | %95 |
| Sphere | 7 | 0,025 | 0,133 | %95 |
| Rastrigin | 3 | 7,1 | 13,2 | %80 |
| Rastrigin | 7 | 35,1 | 73,8 | %100 |
| Rosenbrock | 3 | 0,15 | 1,99 | %90 |
| Rosenbrock | 7 | 37,2 | 113,2 | %95 |
| Gürültülü sphere | 7 | 0,023 | 0,133 | %95 |

ACO ayarı, 3 eğitim + 4 test problemi (30 merkez, 400 talep, minimum doluluk %10, yerel arama kapalı), bütçe 24 ayar x 3 problem x 2 tohum. Skorlar temel ayara göre göreli farktır; amaç cezalara baskın olduğu için oranlar büyük:

| Ayarlayıcı | Eğitim | Görülmemiş test | Süre |
|---|---|---|---|
| ABC | -%43,5 | -%24,2 | 152 sn |
| Rastgele arama | -%67,1 | -%70,0 | 118 sn |

Küçük problemlerde (temel ayar zaten optimum) eğitim farkı gürültü düzeyindeydi ve ABC'nin seçtiği ayar testte %3,9 kötüleşti; bu yüzden `min_improvement` eşiği eklendi.

## Tahmin motoru (v0.2)

### Prototip verisiyle denklik

Prototipin kendi verisi ve protokolü (289 hat, 1 Ocak - 28 Haziran 2026, eğitim 2 Şubat'tan, prototipin kötü gün ve tatil listeleri, prototipin ABC ile ayarladığı LightGBM-Tweedie parametreleri). Gün düzeyinde WMAPE. Veri yalnız yerelde okundu, repoya girmedi.

| Ölçüm | Prototip | pirilib LightGBM-Tweedie | pirilib mevsimsel ortalama (4 hafta) |
|---|---|---|---|
| Test haftası W26 | 19,24 (bias +1,92) | 18,85 (bias +1,79) | 19,75 |
| Doğrulama haftaları W20, W24, W25 (ortalama) | 19,21 (17,50 / 18,04 / 22,09) | 20,16 (18,03 / 18,18 / 24,28) | 21,73 |
| Walk-forward W12-W26 (havuzlanmış) | - | 24,82 (bias -5,80) | 25,87 |

- Test haftasında kütüphane prototipi geçiyor, üstelik temizleme istatistikleri yalnız geçmişten (prototipte gelecek veri de kullanılıyordu).
- Doğrulama haftalarındaki fark beklenen: prototipin parametreleri bu haftalar üzerinde ayarlandı.
- Walk-forward'da yaklaşık %-6 sistematik eksik tahmin var; ayrıca incelenmeli.
- Varlık filtresinin (35 gün) etkisi ihmal edilebilir (0,01-0,02 puan).

### Sentetik veride modeller

30 hat, 24 hafta, 8 haftalık walk-forward:

| Model | WMAPE | Mevsimsel ortalamaya göre |
|---|---|---|
| Random forest (log1p, smearing) | 26,25 | +%6,6 |
| LightGBM-Tweedie | 27,09 | +%3,6 |
| HistGradientBoosting (log1p) | 27,60 | +%1,8 |
| Mevsimsel ortalama (4 hafta) | 28,11 | - |
| Mevsimsel naive | 35,85 | -%27,5 |

### Kantil tamponunun kapsaması

Tampon ilk 6 katlamadan öğrenildi, sonraki 6 haftada (görülmemiş) ölçüldü; 20 hat, 2 tohum, 2 model:

| Hedef kantil | Gerçekleşen kapsama | Ek hacim |
|---|---|---|
| 0,5 | %50,0-51,3 | yaklaşık %0 |
| 0,8 | %78,9-82,7 | +%25-32 |
| 0,9 | %88,3-89,1 | +%41-48 |
| 0,95 | %91,5-93,6 | +%51-62 |

### Eksik tahmin (bias) incelemesi

LightGBM-Tweedie'nin walk-forward bias'ı (prototip verisi, 15 hafta) haftadan haftaya -32 ile +114 arasında oynuyor; en büyük sapmalar bayram ve tatil haftalarında. Winsorize ana neden değil (kapalıyken bias -5,17, açıkken -5,80). Denenen düzeltmeler, aynı 11 hafta üzerinde (ilk 4 hafta kalibrasyon için):

| Seçenek | Prototip WMAPE | Prototip bias | Sentetik tohum 2 / 7 WMAPE |
|---|---|---|---|
| Mevcut model | 25,10 | -6,29 | 26,87 / 27,38 |
| A: hattın son 28 gün seviyesine oranla öğrenme (`scale=True`) | 25,03 | -7,68 | 27,08 / 27,05 |
| B: önceki 4 haftanın toplam gerçek/tahmin oranıyla çarpma | 25,88 | -0,67 | 27,43 / 27,67 |
| Kantil 0,5 tamponu (önceki 4 haftadan) | 27,22 | -2,14 | 28,19 / 28,67 |
| Mevsimsel ortalama (karşılaştırma) | 27,08 | +0,60 | 27,65 / 27,08 |

Hiçbir seçenek doğruluğu düşürmeden bias'ı gidermiyor; varsayılan model değiştirilmedi.

## v0.3 adım 3a

- Zamanlı property testi: 2.000 rastgele planda yeni kurallar tetiklendi (araç limiti 951, erteleme 830, hazır olmadan yükleme 520, elleçleme 84 plan; gecikme cezası 1.374 plan); numpy değerlendirici ile saf Python doğrulayıcı arasında 0 uyuşmazlık.
- ACO, 10 merkez, 2 gün, 80 talep (sentetik): zamansız problem hızlı yolda 0,7 sn, uygun, 855.347. Aynı problem hız, saatlik maliyet, 09:00 / 17:00 hazır saatleri, 24 saat teslim, elleçleme ve bir merkezde 60.000 desi elleçleme kapasitesiyle genel yolda 1,4 sn, 1.059.052 (gecikme 12.230) ve **elleçleme kapasitesi 172.907 desi aşıldı**: koloni henüz kapasiteyi ve ertelemeyi bilmiyor (3b).

## v0.3 adım 3b

3a'daki sentetik problem yeniden kuruldu: `synthetic.generate(centers=10, days=2, demand_per_day=40, seed=3)`, bütün araç tiplerine 70 km/sa, bütün araçlara saatlik 320, çift sıradaki taleplere 17:00, tek sıradakilere 09:00 hazır saati, 24 saat teslim, gecikme cezası 0,5, dakikada 100 desi elleçleme ve M001'e (merkez 0) günlük 60.000 desi elleçleme kapasitesi. Zamansız hali 3a ile aynı sonucu veriyor (855.347); eski genel yolun aşımı birebir aynı (172.907,2 desi), maliyeti 1.058.717 (3a'da 1.059.052; saatlik maliyetin 3a'daki değeri bilinmediği için küçük fark). Süreler 7 çalıştırmanın en kısası (eski kod 3).

| Problem | Kod | Yol | Süre | Toplam maliyet | Gecikme | İhlal |
|---|---|---|---|---|---|---|
| Zamansız | main | hızlı | 0,31 sn | 855.347 | - | - |
| Zamansız | 3b | hızlı | 0,23 sn | 855.347 | - | - |
| 60.000 desi | main | genel | 1,70 sn | 1.058.717 | 12.230 | elleçleme 172.907 |
| 60.000 desi | 3b | doğrusal | 0,44 sn | 746.718 | 5.755 | kapsama 204.094 |
| 60.000 desi, 1 gün erteleme | 3b | doğrusal | 0,47 sn | 763.071 | 5.755 | kapsama 211.616 |
| 250.000 desi | main | genel | 1,70 sn | 1.070.403 | 5.755 | - |
| 250.000 desi | 3b | doğrusal | 0,58 sn | 1.032.126 | 5.755 | - |
| Kapasitesiz | main | genel | 1,68 sn | 1.070.403 | 5.755 | - |
| Kapasitesiz | 3b | doğrusal | 0,32 sn | 1.032.131 | 5.755 | - |

- 60.000 desi ile uygun plan yok: M001 günde yaklaşık 162.000 desi yükler ve boşaltır (iki günde 324.000). Gece yarısını aşan varışlarla 3. gün de sayılsa en çok 180.000 desi elleçlenebilir, yani en az 144.000 desi taşınamaz. 3b planı elleçleme, araç limiti, hazır olma ve erteleme ihlali olmadan 204.094 desi bırakıyor; M001'de günlere göre elleçlenen 60.000 / 60.000 / 0. Kalan farkın nedeni inşa sırasında her şeyin çıkış gününe yazılması: kapasitesiz planda M001'in 3. gün payı 31.187 desi, 3b bu odayı kullanamıyor.
- Uygunluk eşiği 230.000 (2.463 desi kalıyor) ile 250.000 arasında. 250.000'de 3b eski genel yoldan %3,6 ucuz ve 2,9 kat hızlı.
- Süre: 60.000 desi için zamansız hızlı yolun 1,94 katı (aynı kodla; main'in zamansız süresine göre 1,4 kat), 250.000 için 2,5 katı. Zamansız hızlı yol da hızlandı (0,31 sn'den 0,23 sn'ye): karıncalar boş satır kümeleri için artık işlem yapmıyor.
- Erteleme maliyeti ilk denemede yalnız gecikme ve araç kapasitesi başına sefer payıydı; erteleme her zaman ucuz göründü ve kapasitesiz problemde bile yükler ertesi güne kaydı (1 gün ertelemeyle 1.201.052, gecikme 263.667). Tek başına bir sefer maliyetiyle değiştirildi; şimdi kapasitesiz problemde ertelemeli ve ertelemesiz sonuç aynı (1.032.131).
- Değişmezlik testi: 150 rastgele zamanlı problemde (hız, saatlik maliyet, hazır saatleri, 4/8/24 saat teslim, 0-2 gün erteleme, merkezlerin %40'ında elleçleme kapasitesi, %30'unda araç limiti) çözücünün iç skoru ile `evaluate` sonucundan hesaplanan amaç arasındaki en büyük göreli fark 3,8e-16. Rota, yük, hazır olma, erteleme, kapasite ve bölme ihlali hiç yok; elleçleme 1, araç limiti 2 problemde kaldı (çıkış günü yaklaşımı; skora dahil).
- Zamansız gerileme: örnek problem, 40 rastgele problem (yerel aramalı, yerel aramasız, genel yol) ve 6 sentetik problem (`exact_nodes` ile ve olmadan), toplam 134 çalıştırmada planlar, skorlar ve amaç değerleri main ile birebir aynı.
- Aynı 60 rastgele zamanlı problemde main'in genel yoluna göre: elleçleme ihlali olan plan 0 (main 40), araç limiti ihlali olan plan 0 (main 27); amaç 27 problemde eşit ya da daha iyi. Kalanlarda 3b talebi taşımadan bırakıyor, main ise kapasiteyi aşarak taşıyordu (ikisi de birim başına 1e6 ceza), bir problemde de spot günlük limitini önce gelen çıkışlar tüketiyor (bilinen açgözlü paylaşım). Toplam süre 9,5 sn (main 31,0 sn).
- 81 merkez, 1 gün, 1.350 talep: zamansız 6,0 sn (main 7,1 sn), plan aynı (16.307.932). Zamanlı (aynı saat ve hız ayarları, en yoğun 5 merkezde toplam yükün %80'i kadar elleçleme kapasitesi): 3b 27,3 sn, 23.438.084, uygun; main genel yol 10,2 sn, 32.369.304, minimum doluluk ihlali 25.583 desi. Alt problemlerin çoğu sınırlı bir merkeze dokunduğu için tek tek çözülüyor (71 parti; zamansızda 5), süre zamansızın 4,5 katı.

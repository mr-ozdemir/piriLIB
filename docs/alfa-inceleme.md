# Alfa incelemesi: karar listesi (K01–K171)

Bu belge `docs/kararlar.md`'deki kararları üç sınıfa ayırır ki inceleme bir oturumda bitsin:

- **A — birlikte karar verilecekler (19):** kullanıcıyı, API'yi, sonuçları ya da doğruluğu etkiliyor ve makul bir sahip farklı seçebilir. Her biri için soru, öneri ve öneriyi değiştirecek durum yazıldı. Sırayla okunup "koru / değiştir" denmesi yeterli.
- **B — hızlı onay (96):** makul varsayılan, pişmanlığı düşük, sonra değiştirmesi kolay. Tabloyu tarayıp itiraz olmayan satırlar toplu "onaylandı" yapılır.
- **C — yalnız kayıt (36):** olgu ya da araç gereği, ya da tamamen iç. Okunması bile gerekmez; durumları "onaylandı" yapılır.

Sınıflama K01–K151 için yapıldı. Sonradan gelen K152–K161 (dalga bölme) ve K162–K171 (belge sitesi) B sınıfında sayılabilir; sondaki "Sonradan eklenenler" bölümüne bakın. Kararların metni kodla karşılaştırıldı; uyuşmayanlar ve birbirini geçersiz kılanlar "Tutarsızlıklar" bölümünde, her birinin yanında bugünkü durumu var.

Birlikte okunması gerekenler: K83 + K90 + K109 (ceza ağırlıkları), K55 + K26 (özmal modeli), K101 + K107 + K149 (ABC bütçesi), K119 + K125 + K129 (tahmin dürüstlüğü ve tampon), K131 + K140 + K142 (zamanlı kurallar ve 3b kapsamı).

## A: Birlikte karar verilecekler

### 1. K83 — Varsayılan amaç: maliyet + 1e6 × tüm ihlaller
**Soru.** Tüm yerleşik ihlaller aynı ağırlıkla cezalandırılıyor; `coverage` (taşınmayan desi) ile `min_fill` (eksik doluluk desisi) aynı birimde ve aynı ağırlıkta olduğundan çözücü, küçük bir talebi taşımak yerine bırakmayı "daha ucuz" bulabiliyor (ölçümde 30 günlük problemde 346,7 desi bırakıldı). Asıl soru: taşınmayan talep iş kuralı olarak doluluk açığından daha mı ağır?
**Öneri.** Değiştir: varsayılan amaçta `violation[coverage]` ağırlığını diğer ihlallerin üstüne çıkar (örn. 1e6 × ihlaller + 1e7 × coverage), CLI `--penalty` aynı kalsın. Böylece "her talep gider" varsayılan davranış olur, kullanıcı isterse eşitler.
**Değiştirecek durum.** "Minimum doluluk altı araç kalkmaz, küçük talep ertesi güne kalır" deniyorsa mevcut eşit ağırlık doğru; o zaman K90 ile birlikte "koru" ve kılavuzda açıkça yaz (şu an §3'te yazılı).
**Durum.** Karar verildi: varsayılan amaç yok, amaç her yerde zorunlu, cezası olmayan kural hata (K205+).

### 2. K90 — Yerel aramanın "bırak" hamlesi talebi taşımadan bırakabiliyor
**Soru.** Amaç fonksiyonuna sadakat gereği yerel arama, taşıması cezadan pahalı olan rotayı siliyor; kesin çözücü de aynı optimumu buluyor, yani matematiksel olarak tutarlı. Ama kullanıcı "plan talebi unuttu" diye okuyabilir.
**Öneri.** K83'teki ağırlık değişikliğiyle birlikte koru; hamle amaçla tutarlı olduğu sürece doğru. Kılavuza "unserved > 0 görürseniz amaç ağırlıklarına bakın" uyarısı eklensin.
**Değiştirecek durum.** Kapsama sert kısıt sayılıyorsa `drop` hamlesi `ACOConfig` ile kapatılabilir yapılır ve varsayılan kapalı olur.

### 3. K55 — Özmal maliyet modeli
**Soru.** `mandatory: true` özmal grupta sabit maliyet kullanılsın kullanılmasın her gün ödeniyor, seferde yalnız km maliyeti var; `mandatory: false`'ta sefer başı sabit + km. Bu, prototipteki özmal filonun yorumu; genel kütüphanede "kiralık ama bağlı araç" bunun hangisine giriyor belirsiz.
**Öneri.** Koru; `mandatory` adını ve kılavuz açıklamasını onayla.
**Değiştirecek durum.** Yarışma kuralında özmal sabit maliyetin yalnızca çıkan araç için ödendiği anlaşılırsa varsayılan `mandatory: false` olmalı.

### 4. K26 — Özmal varsayılanları: `mandatory` açık, `stopovers` kapalı, tek yön, günlük `count`
**Soru.** Varsayılanlar K55 ile tutarlı ama "yol üstü bırakma" özmal için varsayılan kapalı, spot için (K57) her zaman açık. Kullanıcı özmal tırın da ara durak yapmasını bekleyebilir.
**Öneri.** Koru; sentetik üreteç ve örnek dosya zaten `stopovers: true` yazarak gösteriyor.
**Değiştirecek durum.** Özmal hatların çoğu yol üstü bırakıyorsa varsayılan açık olmalı.

### 5. K57 — Spot araçlar kural açıkken her zaman yol üstü durak yapabilir
**Soru.** Spot teklifte ayrı bir `stopovers` alanı yok; sözleşmesi tek varış olan spot araç tipi tanımlanamıyor.
**Öneri.** Koru (prototipteki davranış), ama `SpotOffer.stopovers: bool = True` alanını yol haritasına yaz; şema geriye uyumlu kalır.
**Değiştirecek durum.** Spot araçların bir kısmı tek varışa kiralanıyorsa alan hemen eklenmeli.

### 6. K21 — Birincil boyut ve minimum doluluk yalnız birincil boyuta bakar
**Soru.** Çok boyutlu kapasitede (desi + kg) doluluk kuralı yalnız desiye uygulanıyor; `handling`, `service` ve `vehicle_limit` de birincil miktar üzerinden. Kg ile dolan ama desisi düşük araç "az dolu" sayılır.
**Öneri.** Koru; prototipte tek ölçüt desiydi ve çok boyutlu doluluk kuralını kullanıcı kendi kısıtıyla yazabilir.
**Değiştirecek durum.** Alfa kullanıcısı kg tabanlı doluluk isterse `MinFill` boyut başına sözlük olur.

### 7. K23 — Yol üstü varsayılanları: açık, `max_stretch` sınırsız, `forward_only` açık, `max_stops` sınırsız
**Soru.** Örnek problem ve sentetik üreteç 1.3 stretch yazıyor ama şema varsayılanı sınırsız; kullanıcı alanı unutursa km maliyeti tek fren olur. `forward_only` ise varsayılan katı.
**Öneri.** Koru; iş kuralını kullanıcı yazsın ilkesine uygun. Kılavuz §1'de "örnek 1.3" notu var.
**Değiştirecek durum.** Alfa testinde kullanıcılar sürekli 1.2–1.3 yazıyorsa varsayılanı 1.3 yapmak daha güvenli.

### 8. K28 — Maliyet: sefer başı sabit + km (+ K134 ile saat); birim (desi) başı maliyet yok
**Soru.** Kılavuz "Known limits"te de yazılı. Taşınan desi başına fiyatlanan spot sözleşmeleri ifade edilemiyor.
**Öneri.** Koru; `Cost.per_unit` alanı aday olarak yol haritasına girsin (değerlendiricide tek satır, çözücüde sezgisel değişmez).
**Değiştirecek durum.** Maliyet tablosunda birim başı kalem varsa hemen eklenmeli.

### 9. K85 — Günlük spot limiti açgözlü paylaşılıyor (aynı günün alt problemleri sırayla, büyük talepli çıkış önce)
**Soru.** Limit bağlayıcıysa plan bütünü için optimum garanti yok; K94 kesin çözücüde de aynı. Sonuç gruplamaya bağlı olduğu için tohum tekrar üretir ama "en iyi" değildir.
**Öneri.** Koru; limitsiz spot tipik durum. `SolveResult.stats`'a "spot limiti bağladı mı" sayacı eklensin ki kullanıcı görsün.
**Değiştirecek durum.** Alfa problemlerinde `available` yaygın ve sık dolarsa limitleri alt problemler arasında ikinci bir turla yeniden dağıtmak gerekir.

### 10. K119 — Temizleme istatistikleri yalnız tahmin başlangıcından önceki veriden (prototipten sapma)
**Soru.** Kural "tahmin motoru v1 prototip gibi"; burada bilinçli sapma var: prototip medyanları gelecek veriyle de hesaplıyordu (sızıntı). Ölçümde test haftasında kütüphane prototipi yine geçti (18,85 / 19,24), doğrulama haftalarında geride (20,16 / 19,21, parametreler o haftalarda ayarlanmıştı).
**Öneri.** Koru; dürüst backtest olmadan tampon (K125) ölçülemez.
**Değiştirecek durum.** Prototiple birebir sayı eşleşmesi isteniyorsa `clean(..., lookahead=True)` seçeneği eklenir, varsayılan kapalı kalır.

### 11. K125 — Kantil tamponu: isteğe bağlı, hat başına backtest kantili, az veride genel kantil
**Soru.** Varsayılan tahmin beklenen değer; tamponsuz plan günlerin yarısında açık kalır. CLI `--quantile` boşsa tampon yok. 0,8 kantil ölçümde +%25–32 hacimle %79–83 kapsama verdi.
**Öneri.** Koru, ama CLI `forecast --network` kullanıldığında tampon verilmemişse uyarı bas; kılavuzda 0,8 önerisi zaten var.
**Değiştirecek durum.** "Plan her zaman tamponlu" deniyorsa `--quantile 0.8` varsayılan olur.

### 12. K129 — Bias (-%6) için varsayılan model değiştirilmedi; `scale=True` deneysel
**Soru.** Denenen hiçbir düzeltme doğruluğu bozmadan bias'ı gidermiyor (B seçeneği bias -0,7 ama WMAPE +0,8). Tampon bias'ı dolaylı kapatıyor.
**Öneri.** Koru; `scale=True` deneysel etiketiyle kalsın (`docs/stability.md`'de öyle).
**Değiştirecek durum.** Gerçek veride bias haftalar arasında kararlı çıkarsa (ölçümde -32..+114 arası oynuyordu) kayan kalibrasyon varsayılan yapılabilir.

### 13. K131 — Yol süresi `speed_kmh`'den, hat × tip saat tablosu yok
**Soru.** Prototipte hat × araç tipi saat tablosu vardı; burada km / hız, bacak başına yukarı dakikaya yuvarlama. Saatlik maliyet ve gecikme cezası bu süreye dayanıyor, yani sonuç prototipten sapabilir.
**Öneri.** Koru v1 için; `matrix`'e isteğe bağlı `hours` (hat × tip) alanı ekle, verilmişse hızın yerine geçsin.
**Değiştirecek durum.** Alfa problemi saatlik maliyetliyse tablo hemen gerekli; değilse bekleyebilir.

### 14. K140 — Çözücüler zamanlı kuralları bilmiyor
**Soru.** Zamanlı problemde ACO genel yola düşüyor ve yalnız ceza yönlendiriyor (ölçümde elleçleme kapasitesi 172.907 desi aşıldı); kesin çözücü hata veriyor. Alfa bu haliyle mi çıkacak, yoksa 3b önce mi?
**Öneri.** Alfayı zamansız kapsamla duyur, 0.2 şema alanlarını "ölçülür ama planlanmaz" diye işaretle; 3b alfadan sonra.
**Değiştirecek durum.** Alfa kullanıcısı dalga/teslim saati senaryosu test edecekse 3b önce gelmeli.
**Durum.** 3b birleşti (K186–K199): ACO zamanı, elleçleme kapasitesini, araç limitlerini ve ertelemeyi planlıyor; kesin çözücü zamanlı problemi hâlâ reddediyor. Soru artık "3b alfaya deneysel olarak girsin mi?"; öneri: girsin, `docs/stability.md`'deki gibi deneysel etiketle.

### 15. K142 — Modellenmeyenler: kiralık araçların zorunlu çıkışı (prototip K7), çalışma saatleri, sürücü kuralları, bacaklar arası bekleme
**Soru.** Prototipin K7 kuralı iş kuralıydı; genel kütüphanede yok. Diğerleri kapsam dışı sayılabilir.
**Öneri.** K7'yi 3b sonrasına al (`SpotOffer.minimum_daily` gibi tek alan + `spot_availability`'nin alt sınırı); kalanını yol haritasında "1.0 sonrası" diye tut.
**Değiştirecek durum.** Yarışma değerlendirmesi K7'yi puanlıyorsa alfadan önce girmeli.

### 16. K149 (K101 ve K107'nin yerine) — ABC besin kaynağı sayısı bütçeye bağlı (bütçe // 8, 2..4), Latin hiperküp başlangıç
**Soru.** `BeeColony.sources` tipi `int | None` oldu (API değişikliği). Ölçüm (`docs/olcumler.md`): test fonksiyonlarında 24 değerlendirmede rastgele aramadan iyi ya da eşit; gerçek ACO ayarında görülmemiş testte -%24,2'den -%67,4'e çıktı, rastgele arama -%70,0 ile hâlâ biraz önde (tek tohum).
**Öneri.** Koru (commit edildi). Kılavuz §5 "60 ya da daha fazla ayar verin" diyor; `tune(budget=40)` ve CLI `--budget 40` varsayılanının 60'a çıkarılması değerlendirilsin.
**Değiştirecek durum.** Birkaç tohumla ölçümde 24 ayarda rastgele arama açıkça önde kalırsa varsayılan bütçe 60 olmalı.
**Durum.** 8 ayarlayıcı tohumuyla ölçüldü (`docs/olcumler.md`, "Varsayılan bütçe, birkaç tohumla"): test fonksiyonları ve aynı düzende ACO ayarı, bütçe 24, 40, 60 ve 80. 24 ayarda rastgele arama önde değil (ABC testte 8 tohumun 5'inde iyi), koşul gerçekleşmedi; varsayılan bütçe 40'ta kaldı, kılavuz §5 ölçüme göre yazıldı (K226). Ölçümde 3 kaynak 24-100 değerlendirmede 4'ten iyi çıktı; kaynak kuralı 120'nin altında 2..3, 120 ve üstünde 4 oldu (K227). 16 ABC çalışmasının 2'sinde aşırı uyum görüldü (K229). Uygulandı, commit edilmedi; sahibi onaylayacak.

### 17. K03 — Python >= 3.12; CI Linux 3.12/3.13/3.14, Windows ve macOS 3.14
**Soru.** Kurumsal ortamlarda 3.10/3.11 yaygın; kod `type` deyimi ve PEP 695 generics kullanıyor, bunları geri almak elle iş. Alfa kullanıcıları büyük olasılıkla Windows 3.12'de.
**Öneri.** 3.12 alt sınırını koru (SPEC 0); Windows'u CI matrisine 3.12 ile de ekle.
**Değiştirecek durum.** Hedef kullanıcı 3.11'deyse alt sınır 3.11'e çekilir; `type` deyimleri `TypeAlias`'a döner.
**Durum.** Windows 3.12 CI matrisine eklendi.

### 18. K01 — Lisans Apache-2.0
**Soru.** Patent maddesi ve NOTICE dosyasıyla kurumsal kullanıma uygun; MIT daha kısa. Yarışma ya da kurum politikası lisansı dayatabilir.
**Öneri.** Koru.
**Değiştirecek durum.** Kurum ya da TEKNOFEST şartnamesi farklı lisans istiyorsa MIT ya da istenen lisans.

### 19. K02 — NOTICE: "Copyright 2026 Muhammet Ozdemir and the pirilib contributors"; `authors` yalnız bir kişi
**Soru.** Ekip üyeleri ve kurum adı bilinmiyordu. PyPI sayfasında ve NOTICE'ta görünecek.
**Öneri.** Ekip adları ve varsa kurum yazılsın; aksi halde koru.
**Değiştirecek durum.** Telif kurumdaysa satır "Copyright 2026 <kurum>" olmalı.

## B: Hızlı onay

| # | Özet | Öneri |
|---|---|---|
| K04 | hatchling, src düzeni | onayla |
| K05 | Çalışma bağımlılıkları numpy, pydantic, pyyaml | onayla |
| K06 | PEP 735 `[dependency-groups]`, pip >= 25.1 | onayla |
| K07 | Sürüm `0.0.1.dev0`, `_version.py` | onayla (S04 ile birlikte) |
| K08 | ruff/mypy strict/pytest uyarı = hata | onayla |
| K09 | İngilizce kod ve belgeler, README'de Türkçe bölüm | onayla |
| K14 | pre-commit ruff sabit sürüm | onayla (v0.17.1'e çekildi) |
| K16 | Şema kimliği zorunlu `schema:` alanı | onayla (0.2 için K130) |
| K17 | Kimlikler kırpılır, NFC, harf duyarlı | onayla |
| K18 | Bilinmeyen alan, inf/NaN hata; frozen modeller | onayla |
| K19 | Talep her boyutu vermek zorunda | onayla |
| K20 | Tüm araç tipleri aynı boyutlar | onayla |
| K22 | Minimum doluluk [0, 1), varsayılan 0 | onayla |
| K24 | Stretch = rota km / doğrudan km (son durağa) | onayla |
| K25 | Talep bölme varsayılan açık | onayla |
| K27 | Araç tipi başına tek spot teklif | onayla (kademeli fiyat adayı) |
| K29 | Merkez yasağı çıkış+durak+varış | onayla |
| K31 | Takvim iki ucu dahil, dışı hata | onayla |
| K32 | Talep kimliği `D000001` | onayla |
| K33 | Doğrulamada yalnız "izinli araç var mı" | onayla |
| K34 | Haversine 6371 km, `factor` | onayla |
| K35 | Açık matris tam olmalı, `symmetric` doldurur | onayla |
| K36 | stdlib csv, utf-8-sig, `,`, `.` | onayla |
| K37 | Ondalık `,` → `.` binlik; boşluklar silinir | onayla |
| K38 | ISO tarih varsayılan, `date_format` | onayla |
| K39 | Hücre içi liste ayracı `\|` | onayla |
| K40 | Başlık kırpma, fazla sütun yok sayılır, boş hücre hata | onayla |
| K41 | Hatalar iki turda | onayla |
| K42 | En çok 50 sorun listelenir | onayla |
| K43 | Konum biçimi 0 tabanlı, CSV `dosya:satır` | onayla |
| K44 | `yaml.safe_load` (YAML 1.1) | onayla; kılavuza "kimlikleri tırnakla" notu |
| K47 | JSON Schema repoda, wheel'de değil | onayla |
| K48 | Üst seviye yüzey (K70/K88 ile genişledi) | onayla |
| K51 | Plan indeks tabanlı | onayla |
| K52 | Plan dizi düzeni CSR, OWNED=0/SPOT=1 | onayla |
| K54 | Tek yön rota, dönüş yok | onayla |
| K56 | Özmal sefer hattın çıkış→varışında | onayla |
| K58 | `forward_only` katı (eşitlik ihlal) | onayla |
| K60 | İhlal birimleri boyutlar üzerinden toplanır | onayla |
| K62 | Kapsama yanlış rotalı yükü de sayar | onayla |
| K63 | Yüksüz sefer/durak serbest | onayla |
| K64 | `overdelivery` ihlal | onayla |
| K65 | Uygun = tüm ihlaller 0; sert/yumuşak yok | onayla |
| K66 | Kendi kısıt: ad çakışması ve negatif → ValueError | onayla |
| K67 | Amaç = çağrılabilir; `Term`/`Sum` | onayla |
| K68 | `MaxRouteKm`, `MaxDispatchesPerDay`; YAML'da kısıt yok | onayla |
| K70 | Üst seviye API genişletildi | onayla |
| K72 | Kullanılabilirlik gün × grup/teklif | onayla |
| K73 | (gün, çıkış) alt problemleri | onayla |
| K78 | Yerleşik kurallar kurulum gereği sağlanır | onayla |
| K80 | MAX-MIN hypercube feromon | onayla |
| K81 | ACO varsayılanları (16/60/20/1/2/0.1/0/0.01) | onayla |
| K82 | Hızlı yol (doğrusal) / genel yol | onayla |
| K86 | `default_rng([seed, grup])` | onayla |
| K87 | numba/çok çekirdek yok | onayla (hız adımında yeniden bakılacak) |
| K88 | `pirilib.solve(...)`, `Solver` protokolü | onayla |
| K91 | Yerel arama saf Python, yalnız hızlı yol | onayla |
| K92 | Kesin kâhin HiGHS MILP, `pirilib[exact]` | onayla |
| K94 | Kesin çözücü sınırları (9 varış, bölünmez talep yok) | onayla |
| K95 | `exact_nodes` varsayılan 0 | onayla |
| K99 | ACO arama uzayı; iterasyon/sabır ayarlanmaz | onayla |
| K100 | ABC düzeltmeleri (sıra, katı kabul, MR 0.5, tek izci) | onayla |
| K102 | Ayar skoru göreli fark, sabit tohumlar | onayla |
| K103 | `min_improvement` 1e-3 | onayla |
| K104 | Süre amaca katılmaz | onayla |
| K105 | Optuna karşılaştırması yok | onayla |
| K106 | `pirilib.tuning` altında | onayla |
| K108 | argparse CLI, çıkış kodları 0/1/2 | onayla |
| K109 | CLI amaç = maliyet + `--penalty` × ihlaller | onayla (K83 ile) |
| K110 | Plan dosyası yük başına satır, isimlerle | onayla |
| K111 | Sayılar `repr`; `route_km`/`dispatch_cost` okunmaz | onayla |
| K112 | `kind`/`vehicle_type` `source` ile tutarlı olmalı | onayla |
| K113 | Excel yalnız yazılır | onayla |
| K114 | Sentetik üreteç (log-normal, kütle çekim, %10 yasak) | onayla |
| K116 | `docs/guide.md`; belge sitesi yok | onayla (K162 ile site geldi; S05) |
| K117 | Yayın iş akışı hazır, çalıştırılmadı | onayla (S02–S04) |
| K118 | Tahmin çekirdeği numpy; eksik gün NaN | onayla |
| K120 | Öznitelikler prototip gibi; ufuk ≤ 7 | onayla |
| K121 | Durumsuz `forecast(...)`; backtest kesilmiş geçmiş | onayla |
| K122 | Zorunlu baseline'lar | onayla |
| K123 | `Regression` adaptörü, LightGBM-Tweedie şampiyon | onayla |
| K124 | 35 günlük varlık filtresi | onayla |
| K127 | CLI `forecast` ağı gömer, 4 katlama tampon | onayla |
| K130 | Şema 0.2, 0.1 kabul, yeni alanlar kapalı | onayla |
| K132 | Elleçleme süresi birim × miktar, yukarı dakika | onayla |
| K133 | `start` isteğe bağlı; en erken başlangıç | onayla |
| K134 | Kullanım saati yükleme başı → son boşaltma | onayla |
| K135 | Gecikme yük başına, başlanan saat × miktar × ceza | onayla |
| K136 | Erteleme: son teslim sabit, ek maliyet yok | onayla |
| K137 | Elleçleme kapasitesi işlemin başladığı güne | onayla |
| K138 | Araç limiti merkez × tip günlük | onayla |
| K139 | Gün penceresi `deferral` kısıtında | onayla |
| K141 | Plan dosyasında `start` `HH:MM` | onayla |
| K148 | `Backtester` sınıfı + ince fonksiyonlar | onayla |
| K150 | `project.urls`, mutlak README bağlantıları, etiket denetimi | onayla (S08) |
| K151 | API kararlılık kuralları (`docs/stability.md`) | onayla |

## C: Kayıt

K10, K11, K12, K13, K15, K30, K45, K46, K49, K50, K53, K59, K61, K69, K71, K74, K75, K76, K77, K79, K84, K89, K93, K96, K97, K98, K101, K107, K115, K126, K128, K143, K144, K145, K146, K147

## Sonradan eklenenler

- **K152–K161 (dalga bölme):** B. Yalnız K152 (pencere: veri olan son 4 olağan gün) ve K153 (havuzlanmış oran, prototipteki ağırlıksız ortalamadan sapma) bir kez okunmalı.
- **K162–K171 (belge sitesi):** C. Sitenin yayımlanması S05'e bağlı.
- **K172–K185 (kıyas paketi, adaptörler):** B. K177 (doğrudan gönderim alt çizgisi) ve K180 (adaptörlerin reddettiği problemler) bir kez okunmalı.
- **K200 (spot limiti sayacı), K201–K203 (hız):** B. K203 çok çekirdeği ve numba'yı yine ertesi sürüme bırakıyor.
- **K186–K199 (3b):** A-14 ile birlikte. K188 Fable'ın tanımından sapıyor (erteleme maliyetine sefer maliyeti de eklendi, ölçümle); K199 kabul koşulunun neden 250.000 desiyle ölçüldüğünü anlatıyor.

## Tutarsızlıklar

Geçersiz kılınanlar. Her birinin karar metnine "**Güncel:**" notu eklendi:

1. **K101 → K149 → K227.** Varsayılan 6 kaynak yerine bütçeye bağlı 2..4 ve Latin hiperküp; K227 ile 120 değerlendirmenin altında 2..3, 120 ve üstünde 4. *K149 commit edildi, ölçüm `docs/olcumler.md`'de; K227 commit edilmedi.*
2. **K107 → K149 → K226.** A-16 8 tohumla ölçüldü: `tune(budget=40)` ve CLI `--budget 40` varsayılanı kaldı; kılavuz §5 "60 ya da daha fazla verin" yerine ölçümü anlatıyor (kazancın çoğu ilk 24 ayarda, 60-80 biraz daha iyi). *Commit edilmedi.*
3. **K97 → K146.** HiGHS çıktısı artık susturuluyor.
4. **K10 → K145.** Commit'ler yerelde atılıyor.
5. **K16 → K130.** Şema kimliği `pirilib/linehaul@0.2`.
6. **K30 → K136.** `max_deferral_days` (varsayılan 0) geldi.
7. **K28 → K134.** Maliyete `per_hour` eklendi.
8. **K48 → K70 → K88.** Üst seviye `pirilib` bugün 15 ad veriyor.
9. **K108.** CLI komutları 8: `validate`, `solve`, `check`, `tune`, `generate`, `backtest`, `forecast`, `schema`.
10. **K126 → K152–K161.** Dalga bölme 3c'de geldi, kararları yazıldı; kılavuz "Known limits" düzeltildi.
11. **K67.** Toplu amaç katmanı K82'deki doğrusal biçimle kapandı.

Metin / kod uyuşmazlıkları:

12. **K78.** Testte 40 tohum var, 500 problem tek seferlik ölçümdü. *Karar metnine not düşüldü.*
13. **K82.** Hızlı yol `scheduled` problemlerde de kapalı (K140). *Not düşüldü.*
14. **K151.** `Service` ve `Handling` modelleri `pirilib.spec`'ten dışa verilmiyordu. *Eklendi; `docs/stability.md` CLI satırı 8 komutun hepsini sayıyor.*
15. **Kılavuz §2 kısıt tablosu.** *`ready`, `deferral`, `handling`, `vehicle_limit` eklendi; `load_route` artık "day" demiyor.*
16. **K14.** *pre-commit ruff `v0.17.1`'e çekildi.*
17. **K143.** Kod ve yapılandırmada `#` yorum yok; README ve kılavuzdaki kod örneklerinde `#` açıklamalar var. *Soru S10'a eklendi.*
18. **K03.** *Windows 3.12 CI matrisine eklendi.*

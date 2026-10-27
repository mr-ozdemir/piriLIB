# Senin verdiğin kararlar

Geliştirme sırasında kullanıcıya sormadan verilen kararların tam listesi. İlk alfa
testine başlarken birlikte gözden geçirilecek. Her kararda: ne seçildi, neden,
alternatif ne, kodda nerede.

Durum sütunu: **açık** (gözden geçirilmedi), **onaylandı**, **değişti**.

Kullanıcının kendi verdiği kararlar burada değil: ACO + ABC çekirdeği, amaç
fonksiyonunu kullanıcının seçmesi, adaptör deseni, numpy ile optimize sıcak
döngüler, tahmin motoru v1'in prototip gibi olması, `piri` markası ve `pirilib`
dağıtım adı.

## Adım 0: Proje iskeleti

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K01 | Lisans Apache-2.0 | Patent hakkı maddesi var, kurumsal kullanıma uygun, araştırmadaki öneri | MIT, BSD-3 | `LICENSE`, `NOTICE` | açık |
| K02 | `NOTICE`: "Copyright 2026 Muhammet Ozdemir and the pirilib contributors"; yazar listesinde yalnız Muhammet Ozdemir | Git kullanıcı adı buydu; ekip üyeleri bilinmiyor | Ekip adları, kurum adı | `NOTICE`, `pyproject.toml` | açık |
| K03 | Python >= 3.12; CI 3.12/3.13/3.14 (Linux), Windows ve macOS yalnız 3.14 | SPEC 0 takvimi; `type` deyimi ve PEP 695 kullanılıyor | 3.10+ ile daha geniş kitle | `pyproject.toml`, `.github/workflows/ci.yml` | açık |
| K04 | Derleme arka ucu hatchling, src düzeni | Saf Python paket için sade ve standart | setuptools, uv_build, flit | `pyproject.toml` | açık |
| K05 | Çalışma bağımlılıkları yalnız numpy>=2.0, pydantic>=2.7, pyyaml>=6.0 | Az ve yaygın bağımlılık | pandas, polars, numba (sonra opsiyonel olabilir) | `pyproject.toml` | açık |
| K06 | Geliştirici bağımlılıkları PEP 735 `[dependency-groups]` ile; pip >= 25.1 gerekiyor | Standart yöntem, extras'a karışmıyor | `[project.optional-dependencies]` dev extra | `pyproject.toml` | açık |
| K07 | Sürüm `0.0.1.dev0`, tek kaynak `_version.py` | Ön-alfa; git etiketine bağlı değil | hatch-vcs ile git etiketinden | `src/pirilib/_version.py` | açık |
| K08 | ruff (satır 100, kurallar E W F I B UP SIM NPY PT RUF), mypy strict, pytest'te uyarılar hata | Erken aşamada sıkı kalite | Daha gevşek kurallar | `pyproject.toml` | açık |
| K09 | Kod, docstring ve dokümanlar İngilizce; README'de Türkçe bölüm | PyPI/GitHub kitlesi uluslararası | Tamamen Türkçe ya da iki dilli dokümantasyon | Tüm repo | açık |
| K10 | Hiçbir şey commit edilmedi; ana dal `main` (**Güncel:** K145 ile değişti: commit'ler yerelde atılıyor) | Kullanıcı commit istemedi | Her adım sonunda commit | git | açık |
| K11 | Paketi import etmek hiçbir şey yazdırmaz; yalnız `pirilib` logger'ına NullHandler eklenir | Kütüphane iyi pratiği | - | `src/pirilib/__init__.py` | açık |
| K12 | ABC modülü `tuning/bee_colony.py` olacak (`abc.py` değil) | `abc` standart kütüphane modülüyle çakışır | `abc_tuner.py` | (planlı) | açık |
| K13 | Örnek veriler tamamen sentetik ve CC0; yarışma verisi ve maliyetleri yok | Üçüncü taraf verisi dağıtılamaz | - | `examples/` | açık |
| K14 | pre-commit: pre-commit-hooks v5.0.0, ruff-pre-commit v0.8.0 | Bilinen kararlı sürümler | Güncel sürümler | `.pre-commit-config.yaml` | açık |
| K15 | Satır sonları LF (`.gitattributes`) | Windows'ta gereksiz diff olmasın | Git varsayılanı | `.gitattributes` | açık |

## Adım 1: Problem tanımı (`pirilib.spec`)

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K16 | İlk problem paketi "linehaul"; şema kimliği `pirilib/linehaul@0.1` ve belgede zorunlu `schema:` alanı (**Güncel:** K130 ile güncel kimlik `pirilib/linehaul@0.2`) | Sürümlü şema, ileride geçiş yapılabilir | Sürümsüz belge | `spec/models.py` | açık |
| K17 | Kimlikler (merkez, araç tipi, boyut) baş/son boşluk kırpılıp Unicode NFC'ye çevrilir; büyük/küçük harf duyarlı, benzer ad eşlemesi yok | Sessiz yanlış eşleme olmasın | Harf duyarsız eşleme | `spec/models.py` (`Name`) | açık |
| K18 | Sıkı doğrulama: bilinmeyen alan hata, inf/NaN hata, modeller değiştirilemez (frozen) | Yazım hataları sessizce yutulmasın | Bilinmeyen alanları yok saymak | `spec/models.py` | açık |
| K19 | Talep her kapasite boyutunu vermek zorunda; eksik boyut 0 sayılmaz | Sessiz varsayım yanlış plan üretebilir | Eksik boyutu 0 kabul etmek | `spec/models.py` | açık |
| K20 | Tüm araç tipleri aynı kapasite boyutlarına sahip olmalı | Dizi formu basit kalır | Eksik boyut = sınırsız | `spec/models.py` | açık |
| K21 | Birincil boyut varsayılanı ilk araç tipinin ilk kapasite boyutu; minimum doluluk yalnız birincil boyuta bakar | Prototipte desi tek ölçüttü | Her boyut için ayrı doluluk kuralı | `spec/models.py` (`Rules`) | açık |
| K22 | Minimum doluluk oranı [0, 1) aralığında; varsayılan spot 0, özmal 0 | %100 doluluk zorunluluğu anlamsız | - | `spec/models.py` (`MinFill`) | açık |
| K23 | Yol üstü varsayılanları: açık, `max_stretch` sınırsız, `forward_only` açık, `max_stops` sınırsız | Sapmayı km maliyeti zaten cezalandırıyor; iş kuralını kullanıcı girsin | Varsayılan stretch sınırı (örn. 1.2) | `spec/models.py` (`Stopover`) | açık |
| K24 | Stretch tanımı: rota uzunluğu / çıkıştan son durağa doğrudan mesafe | Prototipteki "yol üstü" ölçüsüyle aynı | Her ara durak için ayrı sapma | `spec/models.py` | açık |
| K25 | Talep bölme varsayılan açık | Prototip de bölüyordu | Varsayılan kapalı | `spec/models.py` | açık |
| K26 | Özmal araç: bir hatta bağlı, `count` günlük araç sayısı, tek yön; `mandatory` varsayılan açık (sabit maliyet her gün ödenir), `stopovers` varsayılan kapalı | Hepsiburada özmal filosunun yapısı | Dönüş seferlerini modellemek | `spec/models.py` (`OwnedVehicles`) | açık |
| K27 | Spot teklif: araç tipi başına tek teklif (tekrar hata), `available` günlük sayı, boş = sınırsız | Basit ve açık | Aynı tip için birden çok fiyat kademesi | `spec/models.py` (`SpotOffer`) | açık |
| K28 | Maliyet modeli sefer başına sabit + km başı; birim (desi) başı maliyet, köprü/otoyol ücreti yok (**Güncel:** K134 ile saat başı maliyet de var) | Prototip maliyet yapısı | Birim başı ve sabit ek kalemler | `spec/models.py` (`Cost`) | açık |
| K29 | Merkez yasağı aracın o merkezden çıkmasını, orada durmasını ve varmasını kapsar | En güvenli yorum | Yalnız varışı yasaklamak | `spec/models.py` (`Center`) | açık |
| K30 | Talep kendi gününde gönderilir; bekletme/erteleme yok (0.3 sürümünde) (**Güncel:** K136 ile `max_deferral_days` geldi) | Yol haritası | - | `spec/models.py` (`Demand`) | açık |
| K31 | Takvim iki ucu dahil; takvim dışı talep hata | Açık tanım | Takvim dışını yok saymak | `spec/models.py` (`Calendar`) | açık |
| K32 | Talep kimliği verilmezse sıra numarasından üretilir (`D000001`, ...); kullanıcı kimliğiyle çakışırsa "duplicate" hatası | Basit, tekrarlanabilir | UUID, satır numarası | `spec/models.py` | açık |
| K33 | "Bu talebe izinli araç var mı" kontrolü yalnız gerekli koşul; stretch ve minimum doluluk yüzünden imkânsız olanları değerlendirici bulur | Doğrulamada tam fizibilite çözmek pahalı | Doğrulamada ön çözüm | `spec/models.py` (`_demand_issues`) | açık |
| K34 | Haversine dünya yarıçapı 6371 km; `factor` tüm mesafeleri çarpar | Standart ortalama yarıçap | WGS84 elipsoid, OSRM | `geo.py`, `spec/instance.py` | açık |
| K35 | Açık (explicit) matris her sıralı merkez çiftini vermeli; `symmetric: true` eksik ters yönü doldurur; verilen ters yön ezilmez | Eksik mesafe sessizce tahmin edilmesin | Eksikleri haversine ile doldurmak | `spec/models.py`, `spec/instance.py` | açık |
| K36 | CSV okuma stdlib `csv` ile; varsayılan kodlama `utf-8-sig`, ayraç `,`, ondalık `.` | Ek bağımlılık yok | pandas | `spec/tables.py` | açık |
| K37 | Ondalık `,` seçilince `.` binlik ayraç sayılır; boşluk ve bölünmez boşluklar sayılardan silinir | Türk Excel çıktısı | Binlik ayracı ayrı ayar yapmak | `spec/tables.py` | açık |
| K38 | Tarih varsayılanı ISO (YYYY-MM-DD); başka biçim `date_format` ile | Belirsizlik olmasın | Otomatik tarih tahmini | `spec/tables.py` | açık |
| K39 | Hücre içi liste ayracı `\|` (örn. `tir\|kamyon`) | CSV ayracıyla çakışmaz | `;` veya `,` | `spec/tables.py` | açık |
| K40 | CSV başlıkları kırpılır; fazladan sütunlar yok sayılır; tekrar eden başlık hata; boş miktar hücresi hata; boş satırlar atlanır | Excel çıktısına tolerans, ama sessiz veri kaybı yok | Fazla sütunları hata saymak | `spec/tables.py` | açık |
| K41 | Hatalar iki turda: önce CSV hücre hataları, sonra model ve tutarlılık hataları | Bozuk hücreyle model doğrulaması gürültü üretir | Hepsini tek turda | `spec/tables.py`, `spec/loader.py` | açık |
| K42 | Bir mesajda en çok 50 sorun listelenir, gerisi sayılır | Okunabilirlik | Sınırsız | `spec/_format.py` | açık |
| K43 | Konum biçimi `demand[2].quantity.desi` (0 tabanlı); CSV'den gelen satırlar `demand.csv:4: ...` | Python ve pydantic alışkanlığı | 1 tabanlı "satır 3" | `spec/_format.py` | açık |
| K44 | YAML `yaml.safe_load` ile (YAML 1.1: `no` → false gibi tuhaflıklar var; kimlikler metin olmalı) | Güvenli ve yaygın | ruamel.yaml (YAML 1.2, satır numarası verir) | `spec/loader.py` | açık |
| K45 | Problem belgesine isteğe bağlı `name` alanı eklendi | Raporlarda etiket | - | `spec/models.py` | açık |
| K46 | Dizi formu: int64/float64/bool, salt okunur; spot limitsizse `UNLIMITED = -1` | Motorlarda kopyasız ve güvenli | float NaN ile sınırsız | `spec/instance.py` | açık |
| K47 | JSON Schema dosyası repoda (`schemas/`), wheel'e girmez; `python -m pirilib.spec.schema` ile üretilir; `pirilib.spec` onu dışa aktarmaz | runpy uyarısını önlemek için | Şemayı paket verisi yapmak | `spec/schema.py` | açık |
| K48 | Üst seviye `pirilib` yalnız `load_spec`, `PirilibError`, `SpecError`, `__version__` verir (**Güncel:** K70 ve K88 ile genişledi; üst seviye bugün 15 ad) | Küçük ve kararlı yüzey | Her şeyi üstte vermek | `src/pirilib/__init__.py` | açık |
| K49 | Şema testi için `jsonschema` test bağımlılığı | Yayınlanan şema gerçekten doğrulanıyor | Elle kontrol | `pyproject.toml` | açık |
| K50 | ruff'ta Türkçe `ı` harfine izin | Testlerde ve örneklerde Türkçe metin | Kaçış dizileri | `pyproject.toml` | açık |

## Adım 2: Plan, değerlendirici, kısıtlar, amaç, doğrulayıcı

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K51 | Plan indeks tabanlı (gün, merkez, talep, özmal grup, spot teklif = instance'taki sıra); isimle içe/dışa aktarma Adım 6'da | Motorlar indeksle çalışıyor | Planı isimlerle tutmak | `plan.py` | açık |
| K52 | Plan dizi düzeni: duraklar CSR (`stop_start`, `stop_center`), yükler tablosu (sefer, talep, boyut başı miktar); `OWNED = 0`, `SPOT = 1` | numpy ile hızlı ve kopyasız | Sefer başına nesne listesi | `plan.py` | açık |
| K53 | Yapısal hatalar (dizi boyları, aralık dışı indeks, negatif ya da sonsuz yük) `PlanError` fırlatır; iş kuralı ihlalleri asla hata değil, ölçülen ihlal miktarıdır | Çözücüler ihlalli planları da değerlendirebilmeli | Her ihlalde hata | `plan.py`, `evaluation.py` | açık |
| K54 | Rota tek yön: çıkış, sonra duraklar sırayla; km = ayakların toplamı; dönüş yolu yok | Hat taşımacılığı modeli | Dönüşü maliyete katmak | `evaluation.py` | açık |
| K55 | Maliyet: spot = sabit + km başı x rota km (sefer başı); özmal zorunlu (mandatory) = seferde yalnız km başı, sabit maliyet araç sayısı x gün sayısı kadar her durumda; özmal zorunlu değil = sefer başı sabit + km | Prototipteki özmal filo mantığı | Özmal sabitini kullanıma göre ödemek | `evaluation.py` | açık |
| K56 | Özmal sefer kendi hattının çıkışından kalkmalı ve son durağı hattın varışı olmalı; ara durak yalnız grup `stopovers: true` ve kural açıksa | Araç hatta bağlı | Özmal aracı hat dışında da kullanmak | `evaluation.py` | açık |
| K57 | Spot araçlar, `rules.stopover.enabled` açıksa her zaman yol üstü durak yapabilir | Prototipte spot araçlar yol üstü bırakıyordu | Spot için ayrı izin alanı | `evaluation.py` | açık |
| K58 | `forward_only` katı: her durak bir öncekinden kesinlikle daha uzak olmalı; eşit uzaklık ihlal | "Daha uzak" tanımı | Eşitliğe izin vermek | `evaluation.py` | açık |
| K59 | Stretch ihlali km olarak ölçülür: rota km - max_stretch x doğrudan km | Doğrudan mesafe 0 olursa bölme sorunu çıkmaz | Oran olarak ölçmek | `evaluation.py` | açık |
| K60 | İhlal birimleri: sayı ya da miktar; çok boyutta miktarlar boyutlar üzerinden toplanır (birimler karışır); `violations()` terimi ham toplar, ağırlığı kullanıcı verir | Basit ve şeffaf | Boyut başına ayrı ihlal | `constraints.py`, `objective.py` | açık |
| K61 | Tolerans göreli 1e-9 (kapasite kapasiteye, kapsama max(talep, 1)'e, stretch rota km'ye göre) | Yuvarlama yüzünden sahte ihlal olmasın | Mutlak tolerans | `evaluation.py` | açık |
| K62 | Kapsama, rotası yanlış olsa bile talebin tüm yüklerini sayar; yanlış rota ayrıca `load_route` olarak sayılır | Aynı hata iki kez sayılmasın | Yanlış rotalı yükü taşınmamış saymak | `evaluation.py` | açık |
| K63 | Yüksüz sefer ve yük bırakılmayan durak serbest; yalnız minimum doluluk > 0 ise ihlal; km maliyeti zaten cezalandırır | Gereksiz kural koymamak | Boş durağı ihlal saymak | `evaluation.py` | açık |
| K64 | Talepten fazla taşımak ihlal (`overdelivery`) | Prototipteki "hayalet talep" (574 desi) dersi | Fazlaya izin vermek | `evaluation.py` | açık |
| K65 | Uygun (feasible) = tüm ihlaller (kendi kısıtların dahil) sıfır; değerlendiricide sert/yumuşak ayrımı yok, çözücü karar verir | Tek ölçüm kaynağı | Kısıt başına sert/yumuşak bayrağı | `evaluation.py` | açık |
| K66 | Kendi kısıtın: `name` + `violation(evaluation)`; ad katalogla çakışamaz; miktar sonlu ve >= 0 olmalı, yoksa `ValueError` | Sessiz hata olmasın | Çakışan adı ezmek | `evaluation.py`, `constraints.py` | açık |
| K67 | Amaç: `Evaluation -> float` olan her çağrılabilir; `Term` ve `Sum` ile ağırlıklı toplama; birçok planı aynı anda değerlendiren toplu (batch) amaç katmanı Adım 3'e ertelendi (**Güncel:** toplu katman K82'deki doğrusal biçimle kapandı) | Önce sade API | Baştan toplu API | `objective.py` | açık |
| K68 | Hazır örnek kısıtlar `MaxRouteKm`, `MaxDispatchesPerDay`; kendi kısıtların şimdilik yalnız Python API ile, YAML ile değil | Kapsamı dar tutmak | YAML'da kısıt tanımı | `constraints.py` | açık |
| K69 | Bağımsız, saf Python doğrulayıcı `check_plan`; property testlerde değerlendiriciyle karşılaştırılır; yaklaşık 50 kat yavaş | Değerlendiricideki hataları yakalar | Tek uygulama | `validate.py` | açık |
| K70 | Üst seviye API genişledi: `evaluate`, `Evaluation`, `Plan`, `Dispatch`, `Load`, `check_plan`, `PlanError`; `objective` ve `constraints` modül olarak | Sık kullanılanlar üstte | - | `src/pirilib/__init__.py` | açık |
| K71 | Değerlendirme sonuçları salt okunur (diziler ve ihlal sözlüğü) | Yanlışlıkla değişmesin | Yazılabilir sonuçlar | `evaluation.py` | açık |
| K72 | Kullanılabilirlik (gün, özmal grup) ve (gün, spot teklif) başına sayılır | Günlük araç sayısı tanımı | Haftalık limit | `evaluation.py` | açık |

## Adım 3: ACO çözücü (numpy)

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K73 | Problem (gün, çıkış merkezi) alt problemlerine bölünür; alt problemler yalnız günlük spot limitleriyle birbirine bağlı | Araç bir merkezden bir günde kalkar | Tüm planı tek seferde aramak | `solvers/decompose.py` | açık |
| K74 | Alt problemde düğümler = talep olan varışlar + talebi olmayan özmal hat sonları; aynı varıştaki talepler bölünebilirse toplanır, bölünemezse kalem (item) olarak kalır | Rota kurmak için varış yeterli | Her talep satırı ayrı düğüm | `solvers/decompose.py` | açık |
| K75 | Bölünebilir talepte bir durağın yükü, o varıştaki her talep satırına aynı oranda dağıtılır | Kapsama tam ve basit | İlk satırları önce doldurmak | `solvers/decompose.py` | açık |
| K76 | Karınca kararı: önce (araç seçeneği, ilk durak) birlikte, sonra "sonraki durak ya da bitir"; özmal rota bitince hat sonuna gider | Heterojen filo ve özmal hat kuralı tek akışta | Önce rota, sonra araç | `solvers/_colony.py` | açık |
| K77 | Sezgisel (heuristic): ilk seçimde "taşınan miktar / sefer maliyeti"; sonraki durakta "kendi aracıyla gitse maliyeti / sapmanın ek maliyeti x doldurduğu pay"; bitirme 1, minimum doluluk altında 0.1 | Clarke-Wright tasarruf fikri | Yalnız mesafe tabanlı sezgisel | `solvers/_colony.py` | açık |
| K78 | Minimum doluluk ve kapsama dışındaki tüm yerleşik kurallar kurulum gereği sağlanır (500 rastgele problemde doğrulandı) (**Güncel:** testte 40 tohum; 500 problem tek seferlik ölçüm) | Çözücü uygun plan üretsin | Ceza ile düzeltmek | `solvers/_colony.py`, `tests/test_aco.py` | açık |
| K79 | Stretch, her uzatmada yeni son durağa göre kontrol edilir; ara durumda aşan ama sonra düzelecek rotalara izin verilmez (biraz daha katı) | Basit ve güvenli | Yalnız rota sonunda kontrol | `solvers/_colony.py` | açık |
| K80 | Feromon: MAX-MIN hypercube; değerler [tau_min, 1], güncelleme ölçekten bağımsız; varsayılan iterasyon-en-iyisi ile güncelleme | Amaç ölçeğinden bağımsız (Birattari) | Klasik AS/ACS birikimi | `solvers/_colony.py` | açık |
| K81 | Varsayılan ayarlar: 16 karınca, 60 iterasyon, sabır 20, alpha 1, beta 2, rho 0.1, q0 0, tau_min 0.01 | Literatürde yaygın başlangıç; ABC ayarlayacak | - | `solvers/aco.py` | açık |
| K82 | Doğrusal biçimi olan amaç ve kendi kısıtın yoksa hızlı yol (her alt problem kendi başına); aksi halde genel yol (her karınca tüm planı kurar, `evaluate` + amaç) (**Güncel:** K140'tan beri `scheduled` problemlerde de genel yol) | Hız ile genelliği birlikte sunmak | Yalnız genel yol | `solvers/aco.py` | açık |
| K83 | Varsayılan amaç: toplam maliyet + 1e6 x tüm yerleşik ihlaller (**Güncel:** K205 ile değişti: varsayılan amaç yok) | İhlaller pratikte sert kısıt gibi davranır | Sert kısıt filtresi | `solvers/base.py` | açık |
| K84 | Hız için benzer boyuttaki alt problemler dolgulu dizilerde yan yana çalıştırılır (boyut kovaları 4, 8, 16, ...); bellek için en çok 40.000 satır ve 2 milyon ağırlık hücresi | numpy çağrı başı ek yükü baskındı (9 kat hızlanma) | numba ile JIT | `solvers/_colony.py`, `solvers/aco.py` | açık |
| K85 | Spot limiti varsa bir grupta bir günden en çok bir alt problem olur; aynı günün alt problemleri spot araçları sırayla tüketir (açgözlü) | Günlük limit doğru paylaşılsın | Limitleri baştan bölüştürmek | `solvers/aco.py` | açık |
| K86 | Rastgelelik: her grup için `default_rng([seed, grup])`; aynı tohum aynı plan, ama sonuç gruplamaya bağlı | Tekrarlanabilirlik | Alt problem başına ayrı akış | `solvers/aco.py` | açık |
| K87 | Ek hızlandırmalar (numba, çok çekirdek, tam araç yüklerini önceden atama) şimdilik yok | Önce doğruluk ve sade bağımlılık | Hemen numba | - | açık |
| K88 | Üst seviye `pirilib.solve(instance, objective, constraints, solver=, seed=)`; çözücüler `Solver` protokolüne uyar | Adaptör deseni | Tek sınıf | `solvers/__init__.py` | açık |

## Adım 4: Yerel arama ve kesin kâhin

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K89 | Yerel arama hamleleri: araç değiştir, birleştir, taşı (relocate), yeniden dengele, hizmet et, bırak; ilk iyileştirme, her hamleden sonra baştan | Kesin çözücüyle karşılaştırmada bulunan eksikleri kapatıyor | Daha az hamle | `solvers/local_search.py` | açık |
| K90 | "Bırak" hamlesi, amaç fonksiyonu öyle söylüyorsa talebi taşımadan bırakabilir (varsayılan cezalarla: minimum doluluk açığı taşınan miktardan büyükse) | Amaç fonksiyonuna sadakat; kesin optimum da böyle yapıyor | Talebi her zaman taşımak | `solvers/local_search.py` | açık |
| K91 | Yerel arama saf Python ve yalnız hızlı (doğrusal) yolda çalışır; büyük problemde çözüm süresini yaklaşık iki katına çıkarıyor (**Güncel:** K201-K202 ile hızlandı, hâlâ saf Python) | Önce doğruluk | numpy ile vektörleştirmek, genel yolda da çalıştırmak | `solvers/aco.py` | açık |
| K92 | Kesin kâhin scipy HiGHS MILP ile, isteğe bağlı bağımlılık `pirilib[exact]`; her uygun rota bir sütun; rota başına tamsayı araç, durak başına sürekli teslim payı; aynı rotadaki özdeş araçlar yükü eşit paylaşır, böylece minimum doluluk dahil kesin | Genel ve doğru bir ölçüt | Prototipteki alt küme DP'si (yalnız kendi kural uzayında kesin) | `solvers/exact.py` | açık |
| K93 | MILP sonrası araç sayıları sabit tutulup 1e-10 toleranslı LP ile yükler yeniden dağıtılır; `mip_rel_gap` 1e-9 | Varsayılan HiGHS toleransları sahte ihlal üretiyordu | Sonucu olduğu gibi kullanmak | `solvers/exact.py` | açık |
| K94 | Kesin çözücü sınırları: alt problem başına en çok 9 varış (varsayılan); bölünemez talep desteklenmez; spot limitleri çıkışlar arasında sırayla paylaşılır (limit varsa plan bütünü için optimum garanti değil) | Rota sayısı üstel büyüyor | Daha büyük sınır, sütun üretimi | `solvers/exact.py` | açık |
| K95 | Melez mod `ACOConfig(exact_nodes=N)`: en çok N varışlı alt problemler kesin çözülür; varsayılan 0 (kapalı) | scipy zorunlu olmasın | Varsayılan açık | `solvers/aco.py` | açık |
| K96 | Rota sıralama: `forward_only` açıkken her alt küme için tek sıra (çıkıştan uzaklığa göre); kapalıyken tüm permütasyonlar | Yalnız geçerli sıraları üretmek | - | `solvers/exact.py` | açık |
| K97 | Bazı çözümlerde HiGHS "HighsMipSolverData::transformNewIntegerFeasibleSolution" satırı basıyor (scipy 1.18.1 içindeki HiGHS); bastırılmadı (**Güncel:** K146 ile değişti: HiGHS çıktısı susturuluyor) | C seviyesinde çıktı | Alt süreçte çözmek | - | açık |

## Adım 5: ABC ayarlayıcı

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K98 | Ayarlayıcı birim küpte arar; `SearchSpace` parametreleri (ondalık, log ölçekli ondalık, tamsayı, seçenek) küpe çevirir | Ayarlayıcı problemden bağımsız kalır | Her tip için ayrı operatör | `tuning/space.py` | açık |
| K99 | Varsayılan ACO arama uzayı: ants 4-40, alpha 0-4, beta 0-6, rho 0.01-0.6 (log), q0 0-0.9, tau_min 1e-4-0.2 (log), update {iteration_best, best_so_far}; `iterations` ve `patience` ayarlanmaz | Süreyi kullanıcı belirlesin | Süreyi de ayarlamak | `tuning/space.py` | açık |
| K100 | ABC düzeltmeleri: sıra tabanlı gözcü seçimi, katı kabul (`<`), değişim oranı 0.5 (en az bir boyut), döngü başına en çok bir izci, sınır = kaynak x boyut, bütçe = amaç değerlendirme sayısı | Prototip denetiminde bulunan hatalar | Klasik Karaboga ABC | `tuning/bee_colony.py` | açık |
| K101 | Varsayılan 6 besin kaynağı (**Güncel:** K149 ile değişti) | Küçük bütçeler için | 10-20 kaynak | `tuning/bee_colony.py` | açık |
| K102 | Ayar skoru: temel ayara göre göreli fark, tüm eğitim problemleri ve tohumlar üzerinden ortalama; tohumlar sabit (ortak rastgele sayılar) | Ölçekten bağımsız ve adil karşılaştırma | Ham amaç değeri | `tuning/tune.py` | açık |
| K103 | Temel ayar ilk denenen noktadır; yeni ayar temeli en az %0,1 (`min_improvement`) geçmezse temel korunur | Gürültüye aşırı uyumu önlemek (ölçümde görüldü) | Her zaman en iyiyi döndürmek | `tuning/tune.py` | açık |
| K104 | Ayarlanan çözüm süresi amaca katılmaz | Kullanıcı iterasyonla süreyi kontrol eder | Süre cezası | `tuning/tune.py` | açık |
| K105 | Optuna karşılaştırması yapılmadı (bağımlılık eklenmedi); kıyas çizgisi rastgele arama | Bağımlılığı az tutmak | Optuna TPE | - | açık |
| K106 | Ayarlayıcı `pirilib.tuning` altında; üst seviyede değil | Ayrı kullanım alanı | `pirilib.tune` | `tuning/__init__.py` | açık |
| K107 | Bulgu: 24 ayarlık küçük bütçede rastgele arama ABC'den iyi çıktı; ABC 150 değerlendirmeli test fonksiyonlarında açıkça önde. Öneri: bütçe en az 60 ayar (**Güncel:** K149 ölçümüyle güncellendi: 24 ayarda fark kapandı, öneri hâlâ en az 60) | Ölçüm sonucu | Küçük bütçe için daha az kaynak | `docs/olcumler.md` | açık |

## Adım 6: Komut satırı, plan dosyaları, sentetik üreteç, belgeler

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K108 | CLI stdlib `argparse` ile; komutlar `validate`, `solve`, `check`, `tune`, `generate`, `schema`; çıkış kodu 0 başarı, 1 geçersiz girdi ya da kural ihlali (`check`), 2 yanlış kullanım (**Güncel:** komutlar bugün 8: validate, solve, check, tune, generate, backtest, forecast, schema) | Ek bağımlılık yok | click, typer | `cli.py` | açık |
| K109 | CLI amaç fonksiyonu: maliyet + `--penalty` (varsayılan 1e6) x ihlaller; CLI'dan özel amaç verilemez (**Güncel:** K208 ile değişti: `--penalty` yok, amaç belgeden) | Basit tutmak; özel amaç Python API ile | Amaç için ifade dili | `cli.py` | açık |
| K110 | Plan dosyası: yük başına bir satır, isimlerle; kaynak `fleet.spot[1]` gibi problem dosyasındaki konumla; boş sefer tek satır | Excel'de okunur, geri okunabilir | Sefer ve yük için iki ayrı tablo | `planio.py` | açık |
| K111 | Plan sayıları tam hassasiyetle yazılır (`repr`); `route_km` ve `dispatch_cost` okunurken yok sayılır, yeniden hesaplanır | Yuvarlama sahte ihlal yaratıyordu | 6 haneye yuvarlamak | `planio.py` | açık |
| K112 | Plan okurken `kind` ve `vehicle_type` sütunları `source` ile tutarlı olmalı | Elle düzenlenmiş planlarda hatayı yakalamak | Bu sütunları yok saymak | `planio.py` | açık |
| K113 | Excel yalnız yazılır (openpyxl, isteğe bağlı `pirilib[excel]`); problem tabloları ve planlar Excel'den okunmaz, CSV'den okunur | Kapsamı dar tutmak | xlsx okumak | `planio.py` | açık |
| K114 | Sentetik üreteç: rastgele merkez boyutu (log-normal), kütle-çekim modeliyle hat seçimi, log-normal desi; araç ve maliyetler örnek problemle aynı uydurma değerler; %10 merkezde tır yasağı | Gerçekçi ama tamamen uydurma | Düzgün dağılımlı talep | `synthetic.py` | açık |
| K115 | Konsol çıktısında kodlanamayan karakterler `?` ile değiştirilir | Windows konsolunda Türkçe adlar çökertmesin | UTF-8 zorlamak | `cli.py` | açık |
| K116 | Kullanım kılavuzu İngilizce `docs/guide.md`; ayrı belge sitesi (Sphinx/MkDocs) henüz yok | Önce içerik | Hemen belge sitesi | `docs/guide.md` | açık |
| K117 | Yayın iş akışı hazırlandı (etiket, TestPyPI, onaylı PyPI, trusted publishing) ama çalıştırılmadı; sürüm hâlâ `0.0.1.dev0`; yayın senin onayınla | Dışa açık ve geri alınamaz adım | Hemen TestPyPI'a yüklemek | `.github/workflows/release.yml` | açık |

## v0.2 (devam ediyor): Desi tahmin motoru

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K118 | Tahmin çekirdeği yalnız numpy (pandas yok); geçmiş hat x gün yoğun matris, kayıt olmayan gün NaN (sıfır değil) | Prototipte sıfırla doldurma kötü sonuç verdi (WMAPE 31 ve 25) | Eksik gün = 0 | `forecast/history.py` | açık |
| K119 | Temizleme prototip gibi: olağan dışı günler ve kayıtsız günler hat x haftanın günü medyanıyla doldurulur; medyan + 4/0,6745 x MAD üstü kırpılır; ama istatistikler yalnız tahmin başlangıcından önceki veriden | Prototip gelecek veriyi de kullanıyordu (sızıntı) | Prototipin birebiri | `forecast/prepare.py` | açık |
| K120 | Öznitelikler prototipteki gibi (gecikme 7/14/21/28, kayan ortalama 7/28, std 28, takvim, hat/çıkış/varış kodu, tatile uzaklık); ufuk en çok 7 gün | Doğrudan tahmin, özyineleme yok | Daha uzun ufuk için özyinelemeli tahmin | `forecast/prepare.py` | açık |
| K121 | Tahminci arayüzü durumsuz `forecast(history, start, days, calendar)`; backtest her katlamada kesilmiş geçmiş verir | Sızıntıyı yapısal olarak önlemek | fit/predict nesneleri | `forecast/models.py`, `forecast/evaluate.py` | açık |
| K122 | Zorunlu baseline'lar: mevsimsel ortalama (son 4 aynı gün, prototip V0) ve mevsimsel naive; `compare` her zaman ekler | Araştırma ilkesi; prototipte şampiyon baseline'ı ancak 1,4 puan geçti | İsteğe bağlı baseline | `forecast/evaluate.py` | açık |
| K123 | Adaptör `Regression(make, target)`: fit/predict olan her model; log1p hedefte Duan smearing; varsayılan şampiyon LightGBM-Tweedie prototipin ABC ile ayarlanmış parametreleriyle (vp=1,3), ekstra `pirilib[forecast]` | Prototip kararı (kullanıcı: v1 prototip gibi) | Tek model | `forecast/models.py` | açık |
| K124 | Varlık filtresi: son 35 günde o haftanın gününde kaydı olmayan hat için tahmin 0 | Prototipteki gibi | Filtre yok | `forecast/models.py` | açık |
| K125 | Plana köprü: tahmin + ağ (merkez, filo, kurallar) = problem belgesi; isteğe bağlı kantil tamponu (backtest gerçek/tahmin oranlarının hat başına kantili, az veride genel kantil) | Plan beklenen değere göre yapılırsa günlerin yarısı açık kalır | Sabit yüzde tampon | `forecast/bridge.py` | açık |
| K126 | Dalga (09:00/17:00) bölme ve senaryo değerlendirmesi v0.2 dışında; zaman dalgaları v0.3 (**Güncel:** dalga bölme 3c'de geldi (K152-K161)) | Kapsam | Hemen eklemek | - | açık |
| K127 | CLI `forecast` tahmini ağ dosyasıyla birleştirir; ağdaki CSV tabloları satır olarak gömülür, çıkan problem doğrulanmadan yazılmaz; tampon için varsayılan 4 katlamalık backtest | Bağımsız, taşınabilir problem dosyası | Ağı dosya referansıyla bırakmak | `cli.py` | açık |
| K128 | Bulgu: prototip verisinde walk-forward tahminler yaklaşık %6 düşük (bias -5,8); düzeltme (örneğin bias çarpanı) henüz yok | Ölçüm sonucu | Otomatik bias düzeltmesi | `docs/olcumler.md` | açık |
| K129 | Bias incelemesi sonucu varsayılan tahmin modeli değiştirilmedi; ölçekli hedef `Regression(scale=True)` deneysel ve isteğe bağlı eklendi; planlamada eksik tahmin kantil tamponuyla karşılanır (0,8 önerilir) | Ölçüm: hiçbir düzeltme doğruluğu bozmadan bias'ı gidermiyor | Kayan kalibrasyonu varsayılan yapmak (bias -0,7, WMAPE +0,8) | `forecast/models.py`, `docs/olcumler.md` | açık |

## v0.3 adım 3a: Zaman, gecikme, erteleme, elleçleme ve araç limitleri (şema, değerlendirici, doğrulayıcı)

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K130 | Şema `pirilib/linehaul@0.2`; 0.1 belgeleri de kabul edilir; yeni alanlar isteğe bağlı ve varsayılan olarak kapalı | Geriye uyumluluk | Yeni şema, eski belgeler geçersiz | `spec/models.py` | açık |
| K131 | Yol süresi araç tipinin `speed_kmh` değerinden (km / hız), her bacak yukarı tam dakikaya yuvarlanır; prototipteki hat x tip saat tablosu henüz yok (3b adayı) | Basit ve her problemde kullanılabilir | Hat x tip süre tablosu | `evaluation.py` | açık |
| K132 | Elleçleme süresi = birim başı dakika x birincil miktar, her işlemde yukarı yuvarlanır; çıkışta tüm yük yüklenir, her durakta o durağın yükü boşaltılır; durak başı sabit süre yok | Prototipteki gibi (100 desi = 1 dk) | Sabit durak süresi eklemek | `evaluation.py` | açık |
| K133 | Seferin `start` değeri isteğe bağlı; boşsa yükleri hazır olur olmaz (ve gününün 00:00'ından önce değil) başlar; 24 saati geçebilir | Çoğu plan için en erken başlangıç doğru seçim | Başlangıcı zorunlu kılmak | `plan.py`, `evaluation.py` | açık |
| K134 | Kullanım saati = çıkışta yüklemenin başlangıcından son duraktaki boşaltmanın bitişine; `per_hour` maliyeti bu saate uygulanır; rota içinde bekleme yok | Prototipteki `kullanim_saat` | Sürücü mola ve bekleme modeli | `evaluation.py` | açık |
| K135 | Son teslim: talebin `due` alanı ya da hazır olma + `service.due_hours`; gecikme yük başına (bölünmüş her parça ayrı) yukarı yuvarlanmış saat x birincil miktar x `penalty`; `service_cost` toplam maliyete dahil | Prototipin ACO hesabıyla aynı | Talep başına bir kez | `evaluation.py`, `objective.py` | açık |
| K136 | Erteleme: yük, talebin tarihinden en çok `max_deferral_days` gün sonra çıkabilir (varsayılan 0); son teslim sabit kalır, ek erteleme maliyeti yok, gecikme kendiliğinden işler | Prototipteki donmuş son teslim | Ayrı erteleme cezası | `evaluation.py` | açık |
| K137 | Elleçleme kapasitesi merkez başına günlük birincil miktar; yükleme ve boşaltma birlikte sayılır; işlemin başladığı güne yazılır (prototip gece yarısını aşan işlemi dakikayla bölüyordu) | Basit ve tutarlı | Dakikayla bölmek | `evaluation.py`, `validate.py` | açık |
| K138 | Araç limiti merkez x tip başına günlük: merkezde yüklemeye başlayan ya da oraya varan farklı araç sayısı; prototipteki tır kapasitesinin genellemesi; limit 0 ile yasak ayrı kavramlar | Her araç tipi için aynı kural | Yalnız tır için | `evaluation.py`, `validate.py` | açık |
| K139 | `load_route` artık günü kontrol etmiyor; gün penceresi yeni `deferral` kısıtında | Erteleme izin verilebilir hale geldi | - | `evaluation.py` | açık |
| K140 | ACO'nun hızlı yolu ve kesin çözücü zamanlı kuralları henüz bilmiyor: ACO genel yola düşer (ölçümde elleçleme kapasitesi aşıldı), kesin çözücü hata verir; 3b'de çözülecek (**Güncel:** K186-K199 ile ACO zamanlı kuralları planlıyor; kesin çözücü hâlâ reddediyor) | Önce doğru ölçüm, sonra arama | Hemen arama tarafını yazmak | `solvers/aco.py`, `solvers/exact.py` | açık |
| K141 | Plan dosyasında `start` sütunu `HH:MM` (24 saati geçebilir), okurken isteğe bağlı | Excel'de okunur | Dakika sayısı | `planio.py` | açık |
| K142 | Modellenmeyenler: kiralık araçların zorunlu çıkışı (prototip K7), çalışma saatleri, sürücü kuralları, bacaklar arası bekleme | Kapsam | - | - | açık |

## v0.3 adım 3b: Çözücüler zamanı ve merkez sınırlarını planlıyor

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K186 | Mimari (Fable kararı): zamanlı problemde sıralı açgözlü defter (`Ledger`: merkez x gün elleçleme, merkez x gün x tip araç yeri); `Coordinator` günleri tembel kurar, sıra (en az bolluk, -talep, merkez); kalkış en erken hazır saatte, `Plan.start` boş; saatlik maliyet ve gecikme vektörel skorda `service` altında, `linear_form` değişmedi; kesin çözücü zamanlı problemi hâlâ reddeder | Prototipin defteri kurulum gereği uygundu; fiyat/Lagrange koordinasyonu çok tur ve onarım ister | Fiyatla koordinasyon | `solvers/ledger.py`, `decompose.py`, `_colony.py`, `aco.py` | açık |
| K187 | Sınırlı merkeze (elleçleme kapasitesi ya da araç limiti olan) hiç dokunmayan alt problemler, spot günlük limiti yoksa aynı gün yan yana çözülür; diğerleri koordinatör sırasıyla tek tek | Birbirlerini etkileyemezler; tek tek çözüm 2 kat süre hedefini aşıyordu (0,70 sn -> 0,44 sn) | Hepsini tek tek | `solvers/aco.py` | açık |
| K188 | Erteleme maliyeti (birincil birim başına) = ertesi gün 00:00'da doğrudan seferle gecikme cezası + o seferin tam maliyeti / kalem miktarı (en ucuz izinli spot); `service` altında | Yalnız gecikmeyle erteleme hep ucuz görünüyor, kapasitesiz problemde bile yük kayıyordu (1.201.052'ye karşı 1.032.131) | Yalnız gecikme cezası (Fable tanımı) | `solvers/decompose.py` | açık |
| K189 | Skor elleçleme ve araç limiti aşımını gerçek zaman çizelgesiyle alt problemin odasına göre artımlı yazar; iç skor `evaluate` ile birebir (150 problemde en kötü fark 3,8e-16) | İnşadaki kalkış günü yaklaşımı nadiren aşım bırakıyor | Aşımı skora katmamak | `solvers/_colony.py`, `solvers/ledger.py` | açık |
| K190 | Alt problem toplam skoruna ertelenen talep katılmaz; aramada ertelenebilir artık `defer_cost`, son günü gelmiş artık kapsama ihlali | Ertelenen talep ertesi gün sayılıyor; iki kez sayılmasın | Tahmini erteleme maliyetini toplama katmak | `solvers/aco.py` | açık |
| K191 | Çıkış merkezinin elleçlemesi de odada: yüklenen miktar min(durak odası, çıkış odası); özmal araç başlarken hat sonundaki yeri ayırır | Değerlendirici çıkıştaki yüklemeyi de sayıyor | Yalnız durak odası | `solvers/_colony.py` | açık |
| K192 | Yerel arama araç değiştirme, birleştirme, taşıma ve talep ekleme hamlelerinde tip başına yer ve elleçleme odasını denetler | Tip değişince yer de değişiyor | Yalnız yeni durak | `solvers/local_search.py` | açık |
| K193 | `Ledger.charge(problem, timeline)`: rotalar `Timeline` içinde taşınır | Genel yolda karınca başına satırlar tek çağrıda | Ayrı `routes` argümanı | `solvers/ledger.py` | açık |
| K194 | Defter takvimden sonra 3 gün daha tutar; daha geç varışlar son sütuna | Gece yarısını aşan varışlar takvim dışına taşabiliyor | Dinamik büyütmek | `solvers/ledger.py` | açık |
| K195 | Bölünebilir talepte artık düğümün kalemlerine orantılı dağılır; son günü gelmiş kalemi olan düğüm pratikte tamamen taşınmalı | Plan yükleri durak miktarını orantılı paylaştırıyor | Aciliyet sırasıyla | `solvers/decompose.py`, `_colony.py` | açık |
| K196 | Koordinatör eşitlikte artan merkez sırası; zamansız yol eski sırayı korur, sonuçlar birebir (134 koşu) | Zamansız sonuçlar değişmesin | Zamansızda da koordinatör | `solvers/decompose.py` | açık |
| K197 | Boş satır kümelerinde inşa adımları erken döner; sınır yoksa oda işleri atlanır (81 merkezde 7,1 sn -> 6,0 sn, plan aynı) | Hız | - | `solvers/_colony.py` | açık |
| K198 | `synthetic.generate` araç tiplerini ve spot tekliflerini derin kopyalar (gizli hata: bir belgede `cost` düzenlemek sonrakileri bozuyordu) | Hata düzeltmesi | - | `synthetic.py` | açık |
| K199 | 3a ölçüm problemi yeniden kuruldu (betik yoktu); 60.000 desi kapasitede uygun plan fiziksel olarak yok (merkez günde ~162.000 desi elleçliyor), uygunluk 250.000 ile ölçüldü | Kabul koşulu bu kapasiteyle sağlanamaz | - | `docs/olcumler.md`, `tests/test_schedule.py` | açık |
| K200 | ACO, spot günlük limiti olan problemlerde `stats["spot_limits_reached"]` verir: limiti tükenen (gün, teklif) çifti sayısı; sıfırdan büyükse açgözlü paylaşım planı bağlamış olabilir (alfa inceleme A-9 önerisi); paylaşımın kendisi değişmedi (K85) | Kullanıcı açgözlü sınırın etkisini görsün | İkinci bir yeniden dağıtım turu | `solvers/aco.py` | açık |

## Hız: yerel arama

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K201 | Yerel arama hızlandırması saf Python'da (Fable kararı): hamle adayları önce float karşılaştırmasıyla elenir (kapasite `(1+2e-9)` payıyla, izinli düğüm kümesi, yol üstü izni), sonra mevcut `arranged`/`valid`/`value` yolu değişmeden çalışır; `nodes_set` yalnız boşaltma yapılan düğümleri tutar | Adayların %97'si kapasite ya da kuralda düşüyor; numpy yolu aday başına ~12 µs, ön kontrol ~0,3 µs; sayılar aynı numpy koduyla hesaplandığı için planlar ve skorlar bit düzeyinde aynı (92 çözümde 0 fark) | numba (ikinci kod yolu), hamleleri numpy ile vektörleştirmek, `_Stop.amount`'ı float demetine çevirmek (son bit farkı) | `solvers/local_search.py` | açık |
| K202 | Reddedilen adaylar alt problem içinde ezberlenir: rota damgası (`stamp`); anahtarlar `(damga, damga, seçenek)` birleştirme, `(damga, konum, damga)` taşıma, `(damga, seçenek)` araç değiştirme, küçülen kaynak rota `(damga, konum)`; yalnız "geçerli ve daha iyi aday yok" ezberlenir, `roomy` ve kullanılabilirlik reddi ezberlenmez; yeniden dengeleme (ve geri alınan denemede yuvarlama farkı kalırsa) rotaları yeni damgayla değiştirir | İlk iyileştirme her kabulde taramayı baştan başlatıyor; adayların %67-86'sı değişmemiş rota çiftlerinin tekrarı; aday sırası aynı, ilk kabul edilen hamle aynı | Son iyileştirmeden devam etmek (sonucu değiştirir), aday listesi | `solvers/local_search.py` | açık |
| K203 | Çok çekirdek ve numba bu sürümde de yok (K87 korunur); yerel arama alt problem başına deterministik, ileride sonuç değiştirmeden çok çekirdek eklenebilir ama Windows `spawn` havuzu 0,45 sn ve kullanıcı betiğinde `__main__` koruması ister; koloni partileri günlük spot limiti sırasıyla bağlı (K85). Hızlandırmadan sonra sürenin ~%85-90'ı koloni inşası | Ölçüm: 81 merkez x 7 günde 52,9 sn -> 20,4 sn, 30 günde 175 sn -> 100 sn | İsteğe bağlı `workers=` parametresi | `docs/olcumler.md` | açık |
| K204 | macOS'ta LightGBM tekerleği OpenMP (`libomp`) olmadan yüklenemiyor: CI macOS işinde `brew install libomp`; kütüphane yükleme hatasını (`OSError`) çözüm ipucuyla `ImportError`'a çevirir; kılavuza not | macOS CI'da `test_lightgbm_tweedie` düştü | Testi macOS'ta atlamak | `ci.yml`, `forecast/models.py`, `docs/guide.md` | açık |

## v0.3 adım 3c: Tahmini zaman dalgalarına bölmek

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K152 | Dalga payı penceresi: her hat ve haftanın günü için o günün toplamı pozitif olan son `weeks` (varsayılan 4) olağan gün | Prototipin "veri olan son 4 hafta"sı ve `SeasonalMean` ile aynı | Katı `[start - 28 gün, start)` penceresi | `forecast/waves.py` | açık |
| K153 | Pay formülü havuzlanmış oran: kullanılan günlerde dalga toplamı / tüm dalgaların toplamı | Prototipteki haftalık payların ağırlıksız ortalaması küçük günlere büyükler kadar ağırlık veriyordu | Ağırlıksız ortalama | `forecast/waves.py` | açık |
| K154 | Geri düşme: kendi hafta günü verisi `min_days=2`'den azsa hattın tüm günleri, o da yoksa tüm hatlar (küresel pay); dalga geçmişinde olmayan hatlar küresel payı alır; başlangıçtan önce hiç kayıt yoksa hata | Prototipin `min_weeks=2`'si | Sabit eşit paylar | `forecast/waves.py` | açık |
| K155 | `Calendar`'ın olağandışı saydığı günlerden pay öğrenilmez; CLI `--holidays`/`--unusual` takvimini bölücüye de verir | Prototipin kötü gün listesi | Tüm günleri kullanmak | `forecast/waves.py`, `cli.py` | açık |
| K156 | Dosyadaki tüm dalgalar tutulur; yalnız başlangıçtan sonra görülen dalganın payı 0 olur ve parçaları alt sınırla düşer | Geleceğe bakmamak | Böyle dalgaları paylardan çıkarmak | `forecast/waves.py` | açık |
| K157 | Kararlılık ölçüsü: hat x hafta günü için günlük payın standart sapması (ddof=0), yalnız kendi verisinden ve en az 2 günle; hücre dalgalar üstündeki en büyük değeri raporlar (iki dalgada prototipin `std09`'u) | Prototiple karşılaştırılabilir | Dalga başına ayrı rapor | `forecast/waves.py` | açık |
| K158 | API: durumsuz `WaveSplitter.fit` değişmez `WaveShares` döndürür; `planning_document(waves=...)` onu alır | Durumsuz tahminci kuralı (K121) | Durumlu bölücü | `forecast/waves.py`, `forecast/bridge.py` | açık |
| K159 | Dalga satırlarında `ready` `"HH:MM"` yazılır; şema kimliği değişmez (0.1 belgeleri de alanı kabul ediyor); `due` yazılmaz, `rules.service.due_hours` `ready`'den sayılır | Ağ belgesini bozmamak | Şemayı 0.2'ye zorlamak | `forecast/bridge.py` | açık |
| K160 | Dalga sütunu `spec/tables.py`'de yeni bir tablo düzeni ile okunur (dosya:satır hataları); saat `09:00`, `9:00`, `09:00:00` ya da `9` olabilir; sütun adı varsayılan `wave` | Hata raporlamasını yeniden kullanmak | Ayrı ayrıştırıcı | `spec/tables.py` | açık |
| K161 | CLI: `--waves` ayrı dosya, `--wave-column`, `--wave-weeks`; günlük geçmiş aynı dosya olabilir (`load_history` dalgaları toplar); dalgalı çıktıda `-o` CSV'ye `wave` sütunu eklenir | Mevcut seçenekler değişmesin | Tek dosya zorunluluğu | `cli.py` | açık |

## Belge sitesi

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K162 | Belge sitesi MkDocs + Material + mkdocstrings; `docs` bağımlılık grubu, `dev`'e dahil değil; CI'da `mkdocs build --strict` işi | Statik, ücretsiz, Python ekosistemi standardı | Sphinx | `mkdocs.yml`, `ci.yml`, `pyproject.toml` | açık |
| K163 | Docstring'lerdeki reST rolleri (`:class:` vb.) derleme sırasında bir griffe eklentisiyle bağlantıya çevrilir; docstring'ler değişmez; çözülemeyen ad bağlantı kalır, strict derleme yazım hatasını yakalar | ~80 docstring'i yeniden yazmamak | Docstring'leri Markdown'a çevirmek | `docs/extensions/docstrings.py` | açık |
| K164 | "Attributes:" maddeleri her özniteliğin kendi belgesine taşınır, ayrı tablo gizlenir | Her öznitelik bir kez, bağlanabilir çapayla | Tabloyu bırakmak | `docs/extensions/docstrings.py` | açık |
| K165 | Docstring'i olmayan public üyeler de gösterilir; alt çizgili adlar gizli (`__version__` hariç) | Strict derleme belgesiz özniteliklere bağlantıda düşüyordu | Yalnız belgeli üyeler | `mkdocs.yml` | açık |
| K166 | Google docstring stili, kaynak gösterilmez, üyeler kaynak sırasında, `__init__` sınıfa katılır | Okunaklı başvuru sayfası | Kategoriye göre gruplama | `mkdocs.yml` | açık |
| K167 | `pirilib` sayfası paketin dışa verdiği her şeyi gösterir (modül sayfalarıyla tekrar, bakım gerektirmez); spec, tuning, forecast tek sayfa; `decompose` sayfası eklendi; `errors` ve `geo` ayrı sayfa almadı | Kendiliğinden güncel | Elle yazılmış dizin | `docs/api/` | açık |
| K168 | Dış envanter (Python, numpy belgeleri) yok; derleme ağ gerektirmez | Çevrimdışı ve tekrarlanabilir derleme | Dış bağlantılar | `mkdocs.yml` | açık |
| K169 | `mkdocs<2` üst sınırı: MkDocs 2.0 eklentileri kaldırıyor | Material ve mkdocstrings uyumsuz olacak | Sınırsız | `pyproject.toml` | açık |
| K170 | Tema: teal/amber, açık/koyu geçişi, `repo_url` GitHub deposu, `site_url` boş (Pages kararı S05'e bağlı) | Varsayılana yakın, sade | - | `mkdocs.yml` | açık |
| K171 | `docs/kararlar.md` ve `docs/olcumler.md` siteye girmez; API kararlılık sayfası `docs/stability.md` adını aldı (`docs/api/` klasörüyle karışmasın) | Türkçe iç notlar ayrı | Hepsini yayımlamak | `mkdocs.yml`, `docs/` | açık |

## v0.4: Kıyas paketi ve dış çözücü adaptörleri

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K172 | Örnek aileleri (`tiny`, `small`, `routing`, `medium`, `no-split`, `spot-limits`, `mixed`) `synthetic.generate` çıktısı düzenlenerek kurulur; örnek i'nin ayarları crc32(aile adı), aile tohumu ve i ile tohumlanır; adlar `small/003` | Tekrarlanabilir, üreteç değişmeden | `generate`'e yeni parametreler | `bench/families.py` | açık |
| K173 | Bölünmez ailede hiçbir spot araca sığmayan talep eşit parçalara (0,1 desiye aşağı yuvarlanır) bölünür | Aksi halde belge doğrulamadan geçmiyor | Miktarı kırpmak | `bench/families.py` | açık |
| K174 | Spot limiti teklif başına `ceil(pay x günlük talep)`, pay {0,2, 0,3, 0,5} | Bağlayıcı ama çözülebilir | Sabit sayı | `bench/families.py` | açık |
| K175 | Koşucu süreyi kendisi ölçer ve her planı aynı amaçla yeniden değerlendirir; `ValueError`/`ImportError` "atlandı" sayılır, başka hata koşuyu durdurur | Tüm çözücüler aynı terazide | `SolveResult`'a güvenmek | `bench/runner.py` | açık |
| K176 | Özet: fark = (amaç - en iyi) / max(\|en iyi\|, 1), en iyi tablodaki tüm koşulardan; "en iyi payı" paydasında atlananlar da var; uygunluk ve farklar yalnız çözülenlerden | Basit, açık tanım | Bilinen optimuma göre fark | `bench/results.py` | açık |
| K177 | `DirectShipping` temel çözücü: her (gün, çıkış, varış) hattı doğrudan; bölünebilirse taşınan pay başına en ucuz tipte dolu araçlar + kalan için en ucuz uygun araç; bölünemezse azalan ilk sığan paketleme; araç seçimi amaçtan bağımsız sefer maliyetiyle; talep doluluk altında da taşınır | Her çözücünün geçmesi gereken alt çizgi | Amaca göre seçim | `solvers/direct.py` | açık |
| K178 | `SequentialSolver` taban sınıfı alt problem sırasını ve gün içi açgözlü spot limitini `DirectShipping` ve adaptörler için ortaklar; `ExactSolver` paralel iş yüzünden yeniden düzenlenmedi | Tekrarı önlemek | Her çözücüde ayrı | `solvers/_sequential.py` | açık |
| K179 | PyVRP ve OR-Tools adaptörleri ortak `RoutingAdapter` indirgemesiyle: tek yön için araç tipi başına bitiş deposu; yasaklar, özmal hat, ileri yön ve durak kuralları matristen çıkarılan yaylar; stretch süre sınırıyla tam kodlanır; `max_stops` ek yük boyutu (özmalda tutucu); spot limiti araç sayısı; isteğe bağlı müşteri ödülü en pahalı doğrudan seferin 100 katıyla sınırlı | Tek indirgeme, iki motor | Motor başına ayrı model | `solvers/adapters/` | açık |
| K180 | Adaptörlerde bölünmüş teslim yalnız dolu doğrudan seferlerle (önce gönderilir), kalan tek müşteri; bölünmez talep, 0'dan büyük minimum doluluk, zamanlı problem, doğrusal biçimi olmayan amaç, boş kapasite terimi, negatif ağırlık ve kendi kısıtlar `ValueError` ile reddedilir | Dürüst kapsam | Yaklaşık kodlamak | `solvers/adapters/base.py` | açık |
| K181 | OR-Tools varsayılanı en ucuz yay + açgözlü iniş (deterministik, tohumu yok sayar); PyVRP varsayılanı alt problem başına 1000 iterasyon, tohumlu; PyVRP'nin uygunsuz rotaları atılır, `stats["infeasible"]`'da sayılır | Kendiliğinden durur, tekrarlanabilir | Zaman sınırlı metasezgisel | `solvers/adapters/` | açık |
| K182 | OR-Tools Python 3.14'te uyarılar hata iken içe aktarmada çöküyor; adaptör `DeprecationWarning`'i susturarak içe aktarır, testler `find_spec` ile atlar | pytest `filterwarnings=error` | - | `solvers/adapters/ortools_solver.py` | açık |
| K183 | Ekstralar `pyvrp>=0.14`, `ortools>=9.15` (denenmiş sürümler); `test` grubunda değiller, CI adaptörleri koşmuyor | Ağır bağımlılıklar (matplotlib, pandas, protobuf) | CI'da koşmak | `pyproject.toml` | açık |
| K184 | `pirilib bench` komutu; bench ve adaptörler 1.0'da deneysel | Yeni ve ölçümü sınırlı | Kararlı API | `cli.py`, `docs/stability.md` | açık |
| K185 | Belge eklentisi standart kütüphane ve yerleşik adları (`ValueError`, `datetime.time`) bağlantı yerine kod olarak gösterir; dış envanter yok (K168) | Strict derleme bu adlarda düşüyordu | Python envanteri (ağ gerekir) | `docs/extensions/docstrings.py` | açık |

## Amaç fonksiyonu zorunlu (K83 kararı)

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K205 | Varsayılan amaç yok: `default_objective` ve `DEFAULT_PENALTY` kaldırıldı; `objective` `pirilib.solve`, `Solver.solve` (ACO, kesin, `SequentialSolver` ile `DirectShipping` ve adaptörler), `tuning.tune` ve `bench.Benchmark`'ta zorunlu; eksikse Python `TypeError`, `None` geçerse `ValueError("... needs an objective ...")` (`solvers.base.require_objective`) | Sahiplerin kararı: kullanıcı sistemini ve amacını kendisi tanımlar, hiçbir şey sessizce varsayılmaz | Varsayılanı korumak, `coverage` ağırlığını artırmak (alfa önerisi) | `solvers/`, `tuning/tune.py`, `bench/runner.py` | açık |
| K206 | Problem belgesinde isteğe bağlı `objective:` bölümü (`ObjectiveSpec`): `cost` (> 0, zorunlu) x `total_cost()`, isteğe bağlı `distance_km`, `dispatches`, `unused_capacity` (varsayılan 0, 0 ise terim eklenmez), `penalties` (kural adı -> birim ihlal ağırlığı, zorunlu); `build()` doğrusal biçimli `Sum` döndürür, her yerleşik kural için ceza terimi (ağırlık 0 olsa da); `LineHaulSpec.build_objective()` bölüm yoksa örnekli `SpecError`; şema kimliği `pirilib/linehaul@0.2` kaldı | Amaç belgede tanımlanan maliyetlerden kurulur; hızlı yol açık kalır; yalnız doğrulanan ya da denetlenen belgeler bölümsüz yüklenir | Ayrı amaç dosyası, şema 0.3 | `spec/models.py`, `spec/schema.py` | açık |
| K207 | Cezası olmayan kural hata: `penalties` anahtarları `CATALOG` adları ya da `other`; bilinmeyen anahtar hata; `other` yoksa her kural adıyla yazılmalı, eksikler mesajda adıyla listelenir; hatalar diğer alan hatalarıyla birlikte, konumuyla (`objective.penalties.tolls`); JSON Schema anahtarları `propertyNames` ile sınırlar; iç içe modelden gelen `SpecIssues` konumları artık alanın konumuyla öneklenir (`_format.validation_problems`) | Sahipler "kural sert sayılsın" yerine "hata" seçti; `other` açık bir toplu ağırlık | Eksik kuralı sert kısıt saymak, eksik kurala 0 vermek | `spec/models.py`, `spec/_format.py` | açık |
| K208 | CLI `--penalty` kaldırıldı (`solve`, `tune`, `bench`); amaç belgeden (`spec.build_objective()`); `objective:` bölümü olmayan belgede çıkış kodu 2 ve `error: <dosya>: <SpecError mesajı>`; `tune` tüm eğitim belgelerinde aynı bölümü ister, `bench` aile örneklerinin belgelerinden okur | Komut satırı da varsayılan amaç taşımasın; tek koşu tek amaç | Belgeler arasında ilkini almak, `--objective` dosyası | `cli.py` | açık |
| K209 | Üretilen ve gelen belgeler açık amaç taşır: `examples/linehaul_mini` ve `synthetic.generate` (`synthetic.OBJECTIVE`: `cost: 1`, `coverage: 10000000`, `other: 1000000`; `generate(objective=...)` ile başka bölüm), `bench` aileleri `generate` üzerinden; `forecast.planning_document` ağ belgesinin `objective:` bölümünü de kopyalar | Örnekler ve kıyaslar olduğu gibi çözülebilsin; taşınmayan talep diğer ihlallerden ağır (alfa A-1 önerisi) | Eski 1e6 eşit ağırlık | `synthetic.py`, `examples/`, `forecast/bridge.py` | açık |
| K210 | Python'da doğrudan verilen amaçların bütünlüğü denetlenmez (her kurala ağırlık verilmiş mi bakılmaz) | Kullanıcı onu bilerek yazdı; düz fonksiyon amaçlar da geçerli | Doğrusal biçimde eksik `violation[...]` aramak | `solvers/` | açık |

## Hız: koloni inşası

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K211 | Koloni inşası hızlandırması saf numpy'de ve sonuç değiştirmeden: yalnız alt probleme ve seçeneğe bağlı değerler (kapasite, tolerans payı, en az doluluk yükü, sahiplik, hat sonu, ara durak, araç tipi, izinli düğümler) inşa başında `option_*` düz tablolarına (`option_id = alt problem * seçenek sayısı + seçenek`), alt problem ve düğüme bağlı olanlar (ileri düğümler, mesafeler) `alt problem * k + düğüm` indeksli `node_*` tablolarına, satırın düğümdeki durumu (kalan talep, kalem, `settled`, başlangıç mesafesi) `row * k + node` indeksli `*_cells` dizilerine; düz tablolar `Batch`'teki aynı değerlerden önekleriyle ayrılıyor, `Build` içinde her değer tek adla okunuyor, seçenek indeksi `option_id` araç ve kalem yuvalarıyla (`slotted`, `item_slot`) karışmıyor; adım çok boyutlu dizilerden `take` ile tek indeksle topluyor; sahipli hattın hat sonu ve ötesi engeli izinli düğümlere, ziyaret edilmiş düğümler rota başında `open` maskesine katlanıyor; tablolar adımın kendi hesaplayacağı sayıların aynısı (aynı işlem, aynı sıra) | Adım başına ~100 küçük numpy çağrısı, süre dizi boyutundan çok çağrı sayısında; `take` 2 ve 3 boyutlu dizilerde fancy indekslemeden 2-3 kat hızlı, `nonzero()[0]` `flatnonzero`'dan iki kat | numba (K87/K203), adımı satır satır Python'da yazmak | `solvers/_colony.py` | açık |
| K212 | Oda olmayan inşada (zamansız problemler, zamanlı problemlerde serbest alt problemler) rota açma ağırlıkları `opening` (satır, seçenek, düğüm) önbelleğinde: bir hücrenin ağırlığı yalnız satır o düğümde yükleyince ya da seçeneğin aracı bitince değişir; ziyaret edilen hücreler `pending`'de birikir ve bir sonraki başlangıç adımında tek çağrıda yeniden hesaplanır, aracı biten seçenek o satırda sıfırlanır; odalı inşada (merkez sınırları) oda her ziyarette tüm düğümleri etkilediği için ağırlık her başlangıçta eskisi gibi baştan hesaplanır | Başlangıç adımındaki (satır, seçenek, düğüm) boyutlu 5-6 büyük işlem kalkıyor; yeniden hesaplanan hücrenin değeri aynı formülle, aynı değerlerden; `_reopen` `_opening`'in hücre hücre biçimi, ikisi birlikte değişmeli (docstring'lerde yazılı), `tests/test_aco.py` 20 rastgele problemde her başlangıç adımında `opening`'in `_opening` ile birebir aynı olduğunu sınıyor | Her başlangıçta baştan hesaplamak (eski), her ziyarette hemen güncellemek (adım başına 2-3 kat çağrı) | `solvers/_colony.py`, `tests/test_aco.py` | açık |
| K213 | Örnekleme kümülatif ağırlığı eşiği geçen ilk sütunu `argmax` ile buluyor (eşik altı sütun sayısını bool toplamıyla saymak yerine), hiçbiri geçmezse son sütun; satır toplamı (`sum`, numpy'nin ikili toplaması) ve kümülatif toplam (`cumsum`) aynı kaldı; sıfır sütunları atmak ya da toplamı `cumsum`'ın son elemanından almak yapılmadı | Ağırlıklar negatif değil, kümülatif ağırlık azalmaz: ilk geçen sütun eşik altı sütun sayısına eşit; rastgele sayılar aynı sırada çekiliyor; sütun atmak ya da toplamı değiştirmek toplama sırasını değiştirir (son bit farkı); `cumsum` artık örneklemenin en büyük kalemi | `searchsorted` satır döngüsü, devrik düzende `cumsum` (ölçümde kazanç yok) | `solvers/_colony.py` | açık |
| K214 | Adım içi kısaltmalar: uzatma adımında bitiş seçen ve araç dolduğu ya da durak hakkı bittiği için kapanan satırlar tek `_finish` çağrısında kapanır (durak kayıt sırası satırlar arasında değişir; rota içi sıra aynı, `_colony` durakları rotaya göre kararlı sıralıyor); `_visit` aracın dolduğunu döndürür, `_decide` yeniden hesaplamaz; `stops` yalnız `max_stops` varken tutulur; sahipli seçenek yoksa hat sonu dolaşması atlanır; tek boyutta `max` yerine doğrudan karşılaştırma; seçim kayıtları `column_stack` yerine sütun sütun | Çağrı sayısı; satırlar birbirinden bağımsız, sayılar ve rastgele sayı sırası aynı | Ayrı `_finish` çağrıları (eski) | `solvers/_colony.py` | açık |
| K215 | Doğrulama referansı değişiklikten önceki kaynağın kopyasından (`git archive 4803351`) ayrı süreçte üretildi: 319 çözüm (204 doğrusal, 115 genel mod) için planlar `np.array_equal`, `stats` (skor dahil) ve amaç `==`; 0 fark; yeni sabit test `test_whole_plans_did_not_change` genel modu eski kodun amaç değeri (1.000.768,9) ve sefer sayısıyla (52) sabitliyor | Eski modülü yeni paketin içinde başka adla yüklemek, değişmemiş yardımcı modüllerin yenisiyle karışabilir; genel mod için sabit test yoktu | Eski modülü aynı süreçte başka adla yüklemek | `tests/test_schedule.py`, `docs/olcumler.md` | açık |

## Depo kuralları

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K143 | Kodda ve yapılandırma dosyalarında `#` yorum satırı yok; satır sonu alan açıklamaları sınıf docstring'lerinin "Attributes" bölümüne taşındı; docstring'ler yorum sayılmadı ve kaldı (kütüphanenin API belgesi) | Kullanıcı kuralı | Docstring'leri de kaldırmak | tüm repo | açık |
| K144 | "OOP ve kütüphane yapısı" yorumu: durum ya da ayar taşıyan her şey sınıf (çözücüler, ayarlayıcılar, tahminciler, `Backtester`, plan, değerlendirme, spec modelleri); public fonksiyon ya durumsuz dönüşüm (`load_spec`, `read_plan`) ya da bir sınıfı kurup çağıran ince giriş noktası (`solve`, `evaluate`, `backtest`); kural `docs/guide.md` "Library structure" bölümünde; `instance.solve()` gibi metot takmaları yok (döngüsel import, tek `evaluate`) | Python kütüphane alışkanlığı | Her şeyi sınıf metodu yapmak | tüm repo | açık |
| K145 | Commit'ler yerelde atılır, push'u depo sahibi yapar | Kullanıcı kuralı | - | git | açık |
| K146 | HiGHS'in C++ tarafından standart çıktıya yazdığı tanı satırları, `milp`/`linprog` çağrıları sırasında dosya tanımlayıcısı boş aygıta yönlendirilerek susturulur | Kütüphane kullanıcının konsolunu kirletmemeli | Çıktıyı bırakmak | `solvers/exact.py` | açık |
| K147 | Depo kökündeki `test.py` (GitHub düzenleme denemesi) silindi; kütüphane yapısında değil, paketlenmiyor, test içermiyor; `54a46b2`'den geri alınabilir | Kütüphane yapısı kuralı | `tests/` altına taşımak | depo kökü | açık |
| K148 | `forecast.Backtester(folds, days, calendar, starts)` sınıfı: `.run`, `.compare`, `.starts`; `backtest`/`compare` ona yönlenen ince fonksiyonlar | Ortak ayarlar tek yerde (K144) | Yalnız fonksiyonlar | `forecast/evaluate.py` | açık |
| K149 | ABC başlangıç besin kaynakları Latin hiperküp; `sources` verilmezse bütçe // 8, 2 ile 4 arası (K101'deki 6 kaynağın yerine). Test fonksiyonlarında 24 değerlendirmede rastgele aramadan iyi ya da eşit, 60 ve 150'de eskisinden iyi (`docs/olcumler.md`) | K107'deki küçük bütçe zayıflığı | Sabit 6 kaynak, küçük bütçede rastgele aramaya düşmek | `tuning/bee_colony.py` | açık |
| K150 | Yayın hazırlığı: `project.urls` GitHub deposunu gösterir; README bağlantıları PyPI'de de çalışsın diye mutlak GitHub adresi; yayın iş akışı etiketin sürümle aynı olduğunu denetler ve tekerleği kurup `pirilib --help` ile dener | PyPI sayfası ve yanlış etiketle yayın riski | Göreli bağlantılar | `pyproject.toml`, `README.md`, `release.yml` | açık |
| K151 | API kararlılık kuralları ve 1.0 listesi `docs/stability.md`'de: semver, `__all__` ile public, bir küçük sürüm uyarılı kullanımdan kaldırma, 1.x tüm eski şema kimliklerini okur, plan CSV sütunları API'nin parçası; deneysel ve iç adlar ayrı listede | 1.0 dondurmasına hazırlık | Kuralları 1.0'da yazmak | `docs/stability.md` | açık |

## Sana sorulacaklar

Kendi başıma yapamadığım ya da senin hesabınla yapılması gereken işler. Bakıp karar verdiğinde ilgili satırı güncelleyeceğim.

| # | Soru / yapılacak iş | Neden sende | Önerim | Durum |
|---|---|---|---|---|
| S01 | GitHub "Contributors" listesi eski commit'leri önbellekten gösterebilir. Beklemek mi, yoksa depoyu silip boş açıp yeniden push'lamak mı? | Depo ayarı senin hesabında | Birkaç gün beklemek; düzelmezse depoyu silip boş açmak ve `main`'i yeniden push'lamak | bekliyor |
| S02 | test.pypi.org ve pypi.org'da `pirilib` için trusted publisher eklemek (depo `mr-ozdemir/piriLIB`, iş akışı `release.yml`, ortamlar `testpypi` ve `pypi`) | PyPI hesabı sende | Önce yalnız TestPyPI | bekliyor |
| S03 | GitHub'da `testpypi` ve `pypi` ortamlarını açmak, `pypi` ortamına onaylayıcı olarak kendini eklemek | Depo ayarı | Onaylayıcı zorunlu olsun | bekliyor |
| S04 | İlk yayın: sürümü `0.1.0`'a mı (ya da `0.1.0a1`'e mi) çekelim, etiketi ne zaman atalım? | Yayın kararı senin | `0.1.0a1` ile TestPyPI'de alfa | bekliyor |
| S05 | Depo özel; belge sitesi (GitHub Pages) özel depoda ücretli plan ister. Siteyi şimdilik yalnız yerelde mi tutalım? | Hesap planı | Depo açılana kadar yerelde `mkdocs serve` | bekliyor |
| S06 | Kararlar listesinin (K01...) alfa incelemesi | Senin kararın | İnceleme belgesinden sırayla | bekliyor |
| S07 | Senin eklediğin `test.py` silindi (K147). Başka bir amacı varsa geri getiririm | Dosya senindi | Silinmiş kalsın | bekliyor |
| S08 | README ve `project.urls` GitHub'a bağlı; depo özel kalırsa PyPI sayfasındaki bağlantılar açılmaz. Yayında depo açık mı olacak? | Depo görünürlüğü | İlk PyPI yayınında depoyu açmak; o zamana kadar yalnız TestPyPI | bekliyor |
| S09 | MkDocs 2.0 Material ve mkdocstrings ile uyumsuz olacak; şimdilik `<2` ile sabit. İleride başka araca (ör. Zensical) geçilsin mi? | Uzun vadeli araç seçimi | 2.0 çıkıp eklenti durumu netleşince bakmak | bekliyor |
| S10 | Kural "yorum satırı yok" kod ve yapılandırmaya uygulandı; README ve kılavuzdaki kod örneklerinde (`pip install ...   # açıklama` gibi) `#` açıklamalar duruyor. Belgelerdeki örnekler de temizlensin mi? | Kuralın kapsamı senin | Örneklerde kalsın (okuru yönlendiriyor); kod dosyalarında hiç yok | bekliyor |
| S11 | PyVRP ve OR-Tools adaptörleri CI'da koşmuyor (ağır bağımlılıklar). Ayrı, haftalık bir CI işi açalım mı? | CI süresi ve dakika kotası | Haftalık ayrı iş | bekliyor |

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
| K10 | Hiçbir şey commit edilmedi; ana dal `main` | Kullanıcı commit istemedi | Her adım sonunda commit | git | açık |
| K11 | Paketi import etmek hiçbir şey yazdırmaz; yalnız `pirilib` logger'ına NullHandler eklenir | Kütüphane iyi pratiği | - | `src/pirilib/__init__.py` | açık |
| K12 | ABC modülü `tuning/bee_colony.py` olacak (`abc.py` değil) | `abc` standart kütüphane modülüyle çakışır | `abc_tuner.py` | (planlı) | açık |
| K13 | Örnek veriler tamamen sentetik ve CC0; yarışma verisi ve maliyetleri yok | Üçüncü taraf verisi dağıtılamaz | - | `examples/` | açık |
| K14 | pre-commit: pre-commit-hooks v5.0.0, ruff-pre-commit v0.8.0 | Bilinen kararlı sürümler | Güncel sürümler | `.pre-commit-config.yaml` | açık |
| K15 | Satır sonları LF (`.gitattributes`) | Windows'ta gereksiz diff olmasın | Git varsayılanı | `.gitattributes` | açık |

## Adım 1: Problem tanımı (`pirilib.spec`)

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K16 | İlk problem paketi "linehaul"; şema kimliği `pirilib/linehaul@0.1` ve belgede zorunlu `schema:` alanı | Sürümlü şema, ileride geçiş yapılabilir | Sürümsüz belge | `spec/models.py` | açık |
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
| K28 | Maliyet modeli sefer başına sabit + km başı; birim (desi) başı maliyet, köprü/otoyol ücreti yok | Prototip maliyet yapısı | Birim başı ve sabit ek kalemler | `spec/models.py` (`Cost`) | açık |
| K29 | Merkez yasağı aracın o merkezden çıkmasını, orada durmasını ve varmasını kapsar | En güvenli yorum | Yalnız varışı yasaklamak | `spec/models.py` (`Center`) | açık |
| K30 | Talep kendi gününde gönderilir; bekletme/erteleme yok (0.3 sürümünde) | Yol haritası | - | `spec/models.py` (`Demand`) | açık |
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
| K48 | Üst seviye `pirilib` yalnız `load_spec`, `PirilibError`, `SpecError`, `__version__` verir | Küçük ve kararlı yüzey | Her şeyi üstte vermek | `src/pirilib/__init__.py` | açık |
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
| K67 | Amaç: `Evaluation -> float` olan her çağrılabilir; `Term` ve `Sum` ile ağırlıklı toplama; birçok planı aynı anda değerlendiren toplu (batch) amaç katmanı Adım 3'e ertelendi | Önce sade API | Baştan toplu API | `objective.py` | açık |
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
| K78 | Minimum doluluk ve kapsama dışındaki tüm yerleşik kurallar kurulum gereği sağlanır (500 rastgele problemde doğrulandı) | Çözücü uygun plan üretsin | Ceza ile düzeltmek | `solvers/_colony.py`, `tests/test_aco.py` | açık |
| K79 | Stretch, her uzatmada yeni son durağa göre kontrol edilir; ara durumda aşan ama sonra düzelecek rotalara izin verilmez (biraz daha katı) | Basit ve güvenli | Yalnız rota sonunda kontrol | `solvers/_colony.py` | açık |
| K80 | Feromon: MAX-MIN hypercube; değerler [tau_min, 1], güncelleme ölçekten bağımsız; varsayılan iterasyon-en-iyisi ile güncelleme | Amaç ölçeğinden bağımsız (Birattari) | Klasik AS/ACS birikimi | `solvers/_colony.py` | açık |
| K81 | Varsayılan ayarlar: 16 karınca, 60 iterasyon, sabır 20, alpha 1, beta 2, rho 0.1, q0 0, tau_min 0.01 | Literatürde yaygın başlangıç; ABC ayarlayacak | - | `solvers/aco.py` | açık |
| K82 | Doğrusal biçimi olan amaç ve kendi kısıtın yoksa hızlı yol (her alt problem kendi başına); aksi halde genel yol (her karınca tüm planı kurar, `evaluate` + amaç) | Hız ile genelliği birlikte sunmak | Yalnız genel yol | `solvers/aco.py` | açık |
| K83 | Varsayılan amaç: toplam maliyet + 1e6 x tüm yerleşik ihlaller | İhlaller pratikte sert kısıt gibi davranır | Sert kısıt filtresi | `solvers/base.py` | açık |
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
| K91 | Yerel arama saf Python ve yalnız hızlı (doğrusal) yolda çalışır; büyük problemde çözüm süresini yaklaşık iki katına çıkarıyor | Önce doğruluk | numpy ile vektörleştirmek, genel yolda da çalıştırmak | `solvers/aco.py` | açık |
| K92 | Kesin kâhin scipy HiGHS MILP ile, isteğe bağlı bağımlılık `pirilib[exact]`; her uygun rota bir sütun; rota başına tamsayı araç, durak başına sürekli teslim payı; aynı rotadaki özdeş araçlar yükü eşit paylaşır, böylece minimum doluluk dahil kesin | Genel ve doğru bir ölçüt | Prototipteki alt küme DP'si (yalnız kendi kural uzayında kesin) | `solvers/exact.py` | açık |
| K93 | MILP sonrası araç sayıları sabit tutulup 1e-10 toleranslı LP ile yükler yeniden dağıtılır; `mip_rel_gap` 1e-9 | Varsayılan HiGHS toleransları sahte ihlal üretiyordu | Sonucu olduğu gibi kullanmak | `solvers/exact.py` | açık |
| K94 | Kesin çözücü sınırları: alt problem başına en çok 9 varış (varsayılan); bölünemez talep desteklenmez; spot limitleri çıkışlar arasında sırayla paylaşılır (limit varsa plan bütünü için optimum garanti değil) | Rota sayısı üstel büyüyor | Daha büyük sınır, sütun üretimi | `solvers/exact.py` | açık |
| K95 | Melez mod `ACOConfig(exact_nodes=N)`: en çok N varışlı alt problemler kesin çözülür; varsayılan 0 (kapalı) | scipy zorunlu olmasın | Varsayılan açık | `solvers/aco.py` | açık |
| K96 | Rota sıralama: `forward_only` açıkken her alt küme için tek sıra (çıkıştan uzaklığa göre); kapalıyken tüm permütasyonlar | Yalnız geçerli sıraları üretmek | - | `solvers/exact.py` | açık |
| K97 | Bazı çözümlerde HiGHS "HighsMipSolverData::transformNewIntegerFeasibleSolution" satırı basıyor (scipy 1.18.1 içindeki HiGHS); bastırılmadı | C seviyesinde çıktı | Alt süreçte çözmek | - | açık |

## Adım 5: ABC ayarlayıcı

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K98 | Ayarlayıcı birim küpte arar; `SearchSpace` parametreleri (ondalık, log ölçekli ondalık, tamsayı, seçenek) küpe çevirir | Ayarlayıcı problemden bağımsız kalır | Her tip için ayrı operatör | `tuning/space.py` | açık |
| K99 | Varsayılan ACO arama uzayı: ants 4-40, alpha 0-4, beta 0-6, rho 0.01-0.6 (log), q0 0-0.9, tau_min 1e-4-0.2 (log), update {iteration_best, best_so_far}; `iterations` ve `patience` ayarlanmaz | Süreyi kullanıcı belirlesin | Süreyi de ayarlamak | `tuning/space.py` | açık |
| K100 | ABC düzeltmeleri: sıra tabanlı gözcü seçimi, katı kabul (`<`), değişim oranı 0.5 (en az bir boyut), döngü başına en çok bir izci, sınır = kaynak x boyut, bütçe = amaç değerlendirme sayısı | Prototip denetiminde bulunan hatalar | Klasik Karaboga ABC | `tuning/bee_colony.py` | açık |
| K101 | Varsayılan 6 besin kaynağı | Küçük bütçeler için | 10-20 kaynak | `tuning/bee_colony.py` | açık |
| K102 | Ayar skoru: temel ayara göre göreli fark, tüm eğitim problemleri ve tohumlar üzerinden ortalama; tohumlar sabit (ortak rastgele sayılar) | Ölçekten bağımsız ve adil karşılaştırma | Ham amaç değeri | `tuning/tune.py` | açık |
| K103 | Temel ayar ilk denenen noktadır; yeni ayar temeli en az %0,1 (`min_improvement`) geçmezse temel korunur | Gürültüye aşırı uyumu önlemek (ölçümde görüldü) | Her zaman en iyiyi döndürmek | `tuning/tune.py` | açık |
| K104 | Ayarlanan çözüm süresi amaca katılmaz | Kullanıcı iterasyonla süreyi kontrol eder | Süre cezası | `tuning/tune.py` | açık |
| K105 | Optuna karşılaştırması yapılmadı (bağımlılık eklenmedi); kıyas çizgisi rastgele arama | Bağımlılığı az tutmak | Optuna TPE | - | açık |
| K106 | Ayarlayıcı `pirilib.tuning` altında; üst seviyede değil | Ayrı kullanım alanı | `pirilib.tune` | `tuning/__init__.py` | açık |
| K107 | Bulgu: 24 ayarlık küçük bütçede rastgele arama ABC'den iyi çıktı; ABC 150 değerlendirmeli test fonksiyonlarında açıkça önde. Öneri: bütçe en az 60 ayar | Ölçüm sonucu | Küçük bütçe için daha az kaynak | `docs/olcumler.md` | açık |

## Adım 6: Komut satırı, plan dosyaları, sentetik üreteç, belgeler

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K108 | CLI stdlib `argparse` ile; komutlar `validate`, `solve`, `check`, `tune`, `generate`, `schema`; çıkış kodu 0 başarı, 1 geçersiz girdi ya da kural ihlali (`check`), 2 yanlış kullanım | Ek bağımlılık yok | click, typer | `cli.py` | açık |
| K109 | CLI amaç fonksiyonu: maliyet + `--penalty` (varsayılan 1e6) x ihlaller; CLI'dan özel amaç verilemez | Basit tutmak; özel amaç Python API ile | Amaç için ifade dili | `cli.py` | açık |
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
| K126 | Dalga (09:00/17:00) bölme ve senaryo değerlendirmesi v0.2 dışında; zaman dalgaları v0.3 | Kapsam | Hemen eklemek | - | açık |
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
| K140 | ACO'nun hızlı yolu ve kesin çözücü zamanlı kuralları henüz bilmiyor: ACO genel yola düşer (ölçümde elleçleme kapasitesi aşıldı), kesin çözücü hata verir; 3b'de çözülecek | Önce doğru ölçüm, sonra arama | Hemen arama tarafını yazmak | `solvers/aco.py`, `solvers/exact.py` | açık |
| K141 | Plan dosyasında `start` sütunu `HH:MM` (24 saati geçebilir), okurken isteğe bağlı | Excel'de okunur | Dakika sayısı | `planio.py` | açık |
| K142 | Modellenmeyenler: kiralık araçların zorunlu çıkışı (prototip K7), çalışma saatleri, sürücü kuralları, bacaklar arası bekleme | Kapsam | - | - | açık |

## Depo kuralları

| # | Karar | Neden | Alternatif | Yer | Durum |
|---|---|---|---|---|---|
| K143 | Kodda ve yapılandırma dosyalarında `#` yorum satırı yok; satır sonu alan açıklamaları sınıf docstring'lerinin "Attributes" bölümüne taşındı; docstring'ler yorum sayılmadı ve kaldı (kütüphanenin API belgesi) | Kullanıcı kuralı | Docstring'leri de kaldırmak | tüm repo | açık |
| K144 | "OOP ve kütüphane yapısı" yorumu: motorlar ve veriler sınıf (çözücüler, ayarlayıcılar, tahminciler, plan, değerlendirme, spec modelleri); `evaluate`, `solve`, `load_spec`, `backtest` gibi ince modül fonksiyonları numpy/scikit-learn tarzı giriş noktası olarak kaldı | Python kütüphane alışkanlığı | Her şeyi sınıf metodu yapmak | tüm repo | açık |
| K145 | Commit'ler yerelde atılır, push'u depo sahibi yapar | Kullanıcı kuralı | - | git | açık |

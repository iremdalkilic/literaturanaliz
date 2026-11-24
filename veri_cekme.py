# -*- coding: utf-8 -*-
"""
veri_cekme.py
OpenAlex akademik veri tabanından hızlı, paralel ve önbellekli makale çekme modülü.
"""

# Paralel (eş zamanlı) API istekleri atarak bekleme süresini azaltmak için ThreadPoolExecutor'ı içe aktarır.
from concurrent.futures import ThreadPoolExecutor
# Farklı uzunluktaki sorgu sonuç listelerini dönüşümlü (round-robin) gezmek için zip_longest'i içe aktarır.
from itertools import zip_longest
# Ortam değişkenlerini (ör. isteğe bağlı OPENALEX_MAILTO) okumak için os modülünü içe aktarır.
import os
# Özetlerdeki HTML/JATS etiketlerini temizlemek ve başlıkları sadeleştirmek için re modülünü içe aktarır.
import re
# Geçici ağ/sunucu hatalarında yeniden deneme öncesi kısa beklemeler için time modülünü içe aktarır.
import time

# OpenAlex HTTP isteklerini yönetmek için requests kütüphanesini içe aktarır.
import requests
# Önbellekleme (@st.cache_data ve @st.cache_resource) için Streamlit'i içe aktarır.
import streamlit as st

# ==============================================================================
# SABİTLER
# ==============================================================================

# Makale özetlerinin yapay zeka isteğini gereksiz şişirmemesi için maksimum karakter sınırı.
MAX_OZET_KARAKTER = 1200
# OpenAlex makale (works) API uç noktası adresi.
OPENALEX_URL = "https://api.openalex.org/works"
# OpenAlex'ten yalnızca gerçekten ihtiyacımız olan alanları istemek için alan filtresi (JSON boyutunu ~%85 küçültür).
OPENALEX_SELECT_CALISMA = (
    "id,doi,title,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,primary_location,open_access"
)
# Benzer çalışma (özgünlük) kontrolünde yalnızca temel kimlik ve atıf alanlarını ister (çok daha hızlı döner).
OPENALEX_SELECT_BENZER = "id,doi,title,publication_year,cited_by_count"
# Geçici sayılan ve yeniden denenmesi mantıklı olan HTTP hata kodları (kota, yoğunluk, sunucu hatası).
GECICI_HTTP_KODLARI = (429, 500, 502, 503, 504)
# İsteğe bağlı olarak ortam değişkeninden okunan OpenAlex Polite Pool e-posta adresi.
VARSAYILAN_OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()


# ==============================================================================
# HTTP OTURUMU (BAĞLANTI HAVUZU)
# ==============================================================================

# Her istekte sıfırdan TCP/SSL el sıkışması yapmamak için tek bir requests.Session nesnesini önbellekte tutar.
@st.cache_resource(show_spinner=False)
def _http_oturumu_al():
    # Kalıcı bağlantı havuzu (keep-alive) sağlayan yeni bir HTTP oturumu oluşturur.
    oturum = requests.Session()
    # OpenAlex'in önerdiği açıklayıcı User-Agent başlığını oturuma ekler.
    oturum.headers.update(
        {
            "User-Agent": "AkademikBoslukAnalizi/2.0 (Streamlit; OpenAlex API)",
            "Accept": "application/json",
        }
    )
    # Oluşturulan oturumu yeniden kullanılmak üzere döndürür.
    return oturum


# ==============================================================================
# ÖZET VE METİN İŞLEME YARDIMCILARI
# ==============================================================================

# OpenAlex'in ters dizin (inverted index) biçimindeki özet sözlüğünü okunabilir düz metne çevirir.
def ozeti_duzelt(inverted_index, max_karakter=MAX_OZET_KARAKTER):
    # Eğer makalenin özeti yoksa veya sözlük boşsa standart uyarı metni döndürür.
    if not inverted_index or not isinstance(inverted_index, dict):
        return "Özet bulunamadı."

    # Her konum numarasını (indeksi) o konumdaki kelimeyle eşleyecek boş sözlüğü açar.
    kelimeler = {}
    # Ters dizindeki her kelimeyi ve o kelimenin geçtiği indeks listesini tek tek dolaşır.
    for kelime, pozisyonlar in inverted_index.items():
        # Pozisyon listesi geçerli bir liste değilse bu kaydı atlar.
        if not isinstance(pozisyonlar, list):
            continue
        # Kelimenin geçtiği her konum numarasına ilgili kelimeyi yerleştirir.
        for poz in pozisyonlar:
            kelimeler[poz] = kelime

    # Eğer geçerli hiçbir kelime yerleştirilemediyse özet yok kabul eder.
    if not kelimeler:
        return "Özet bulunamadı."

    # Konum numaralarını küçükten büyüğe sıralayıp kelimeleri tek boşlukla birleştirir.
    metin = " ".join(kelimeler[i] for i in sorted(kelimeler.keys()))
    # Bazı özetlerin içinde kalan HTML veya XML/JATS etiketlerini (<jats:p> vb.) temizler.
    metin = re.sub(r"<[^>]+>", " ", metin)
    # Birden fazla ardışık boşluğu ve satır sonunu tek boşluğa indirger, kenar boşluklarını siler.
    metin = re.sub(r"\s+", " ", metin).strip()
    # Temizlik sonrası metin tamamen boşaldıysa standart uyarıyı döndürür.
    if not metin:
        return "Özet bulunamadı."

    # Metin izin verilen maksimum karakter sınırını aşıyorsa kelimeyi bölmeden kırpar.
    if len(metin) > max_karakter:
        # Sınıra kadar kesip son boşluktan ayırarak yarım kelime kalmasını önler.
        kirpilmis = metin[:max_karakter].rsplit(" ", 1)[0]
        # Kesilen metnin sonuna üç nokta ekleyerek döndürür.
        return (kirpilmis or metin[:max_karakter]) + "..."

    # Sınır içindeki temiz özet metnini doğrudan döndürür.
    return metin


# Aynı çalışmanın farklı kopyalarını (preprint/dergi sürümü) yakalamak için başlıktan sade bir anahtar üretir.
def _baslik_anahtari(makale):
    # Başlığı küçük harfe çevirir ve harf/rakam dışındaki tüm noktalama ile boşlukları siler.
    return re.sub(r"\W+", "", (makale.get("title") or "").lower())


# DOI veya OpenAlex bağlantısının her zaman geçerli bir http/https URL'si olmasını sağlar.
def _baglanti_normallestir(deger):
    # Değer boşsa doğrudan boş metin döndürür.
    if not deger:
        return ""
    # Kenardaki gereksiz boşlukları temizler.
    link = str(deger).strip()
    # Eğer DOI yalnızca "10.xxxx/..." biçiminde geldiyse başına resmi doi.org adresini ekler.
    if link.startswith("10."):
        return f"https://doi.org/{link}"
    # Zaten tam bağlantıysa olduğu gibi döndürür.
    return link


# ==============================================================================
# OPENALEX FİLTRE VE İSTEK YÖNETİMİ
# ==============================================================================

# Kullanıcının seçtiği kriterleri OpenAlex'in beklediği virgülle ayrılmış filter metnine dönüştürür.
def _filtre_olustur(baslangic_yili, bitis_yili, sadece_acik_erisim, min_atif, ulke_kodu, dil_kodu):
    # Her aramada mutlaka seçilen yıl aralığı ve özeti bulunma şartını temel listeye koyar.
    filtre_parcalari = [
        f"publication_year:{baslangic_yili}-{bitis_yili}",
        "has_abstract:true",
    ]

    # Kullanıcı yalnızca açık erişim istediyse is_oa filtresini ekler.
    if sadece_acik_erisim:
        filtre_parcalari.append("is_oa:true")

    # Minimum atıf sayısı 0'dan büyükse OpenAlex'in büyüktür (>) sözdizimiyle eşiği ekler.
    if min_atif and min_atif > 0:
        filtre_parcalari.append(f"cited_by_count:>{int(min_atif) - 1}")

    # Belirli bir yazar kurumu ülkesi seçildiyse ülke kodu filtresini ekler.
    if ulke_kodu and ulke_kodu != "ALL":
        filtre_parcalari.append(f"authorships.institutions.country_code:{ulke_kodu}")

    # Belirli bir yayın dili seçildiyse dil kodu filtresini ekler.
    if dil_kodu and dil_kodu != "ALL":
        filtre_parcalari.append(f"language:{dil_kodu}")

    # Tüm filtre parçalarını virgülle birleştirip OpenAlex'e hazır hale getirir.
    return ",".join(filtre_parcalari)


# OpenAlex'e kalıcı bağlantı havuzu üzerinden, kısa ve akıllı yeniden denemeli GET isteği atar.
def _openalex_get(params, mailto="", timeout=12, deneme=3):
    # Gelen parametre sözlüğünü dışarıda bozmamak için yerel bir kopyasını alır.
    istek_params = dict(params)
    # Kullanıcı veya ortam değişkeni e-posta sağladıysa OpenAlex Polite Pool (hızlı kuyruk) için ekler.
    etkin_mailto = (mailto or VARSAYILAN_OPENALEX_MAILTO).strip()
    if etkin_mailto:
        istek_params["mailto"] = etkin_mailto

    # Önbellekteki paylaşımlı HTTP oturumunu alır.
    oturum = _http_oturumu_al()
    # Tüm denemeler başarısız olursa fırlatılacak son hatayı tutar.
    son_hata = None

    # Belirtilen deneme sayısı kadar isteği güvenli biçimde dener.
    for deneme_no in range(deneme):
        try:
            # OpenAlex API'ye zaman aşımı korumalı GET isteğini gönderir.
            yanit = oturum.get(OPENALEX_URL, params=istek_params, timeout=timeout)
            # HTTP 4xx/5xx hata kodu döndüyse HTTPError fırlatmasını sağlar.
            yanit.raise_for_status()
            # Başarılı JSON yanıtını Python sözlüğüne çevirip döndürür.
            return yanit.json()
        except requests.exceptions.HTTPError as exc:
            # Hata yanıtındaki HTTP durum kodunu güvenle okur.
            kod = exc.response.status_code if exc.response is not None else None
            # Kalıcı hatalarda (400, 403, 404 vb.) boşa beklemeden hatayı hemen yukarı iletir.
            if kod not in GECICI_HTTP_KODLARI:
                raise
            # Geçici hatayı kaydeder ve varsa sunucunun önerdiği Retry-After süresini okur.
            son_hata = exc
            # Arayüzü dondurmamak için bekleme süresini en fazla 3 saniye ile sınırlar.
            bekleme = min(3.0, 0.8 * (2 ** deneme_no))
            if exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    bekleme = min(3.0, float(retry_after))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            # Zaman aşımı veya bağlantı kopmasında yeniden denemek üzere hatayı kaydeder.
            son_hata = exc
            # Üstel artan kısa bekleme süresini hesaplar (0.8 sn, 1.6 sn).
            bekleme = min(3.0, 0.8 * (2 ** deneme_no))

        # Son deneme değilse bir sonraki denemeden önce kısa süre bekler.
        if deneme_no < deneme - 1:
            time.sleep(bekleme)

    # Tüm denemeler başarısız olduysa son oluşan hatayı fırlatır (böylece cache'e boş veri yazılmaz).
    raise son_hata


# Tek bir arama sorgusu için OpenAlex'ten hafifletilmiş (select filtreli) makale listesi çeker.
# Not: Bu fonksiyon iş parçacıkları (thread) içinde çağrılacağı için hiçbir Streamlit UI çağrısı içermez.
def _tek_sorgu_ham_cek(
    sorgu,
    baslangic_yili,
    bitis_yili,
    sayfa_basi_adet,
    siralama_alani,
    sadece_acik_erisim,
    min_atif,
    ulke_kodu,
    dil_kodu,
    mailto="",
):
    # Arayüzdeki Türkçe sıralama seçimlerini OpenAlex sort parametrelerine eşler.
    siralama_map = {
        "Alaka Düzeyi": "relevance_score:desc",
        "Atıf Sayısı": "cited_by_count:desc",
    }
    # Geçerli sıralama parametresini alır; bulunamazsa alaka düzeyini varsayılan seçer.
    sort_degeri = siralama_map.get(siralama_alani, "relevance_score:desc")

    # Kullanıcının aktif filtrelerinden OpenAlex filter metnini üretir.
    filtre_metni = _filtre_olustur(
        baslangic_yili,
        bitis_yili,
        sadece_acik_erisim,
        min_atif,
        ulke_kodu,
        dil_kodu,
    )

    # Sadece gerekli alanları (select) isteyerek ağ yükünü ve yanıt süresini ciddi ölçüde azaltır.
    parametreler = {
        "search": sorgu,
        "per-page": sayfa_basi_adet,
        "sort": sort_degeri,
        "filter": filtre_metni,
        "select": OPENALEX_SELECT_CALISMA,
    }

    # OpenAlex'e isteği gönderir ve results listesini döndürür.
    veri = _openalex_get(parametreler, mailto=mailto, timeout=12)
    return veri.get("results", [])


# ==============================================================================
# PARALEL VE ÖNBELLEKLİ ÇOKLU SORGU BİRLEŞTİRME
# ==============================================================================

# Tüm sorguları paralel (eş zamanlı) çalıştırıp sonuçları 1 saat boyunca önbellekte saklayan fonksiyon.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def _paralel_sorgulari_cek(
    sorgu_tuple,
    baslangic_yili,
    bitis_yili,
    sayfa_basi_adet,
    siralama_alani,
    sadece_acik_erisim,
    min_atif,
    ulke_kodu,
    dil_kodu,
    mailto,
):
    # Her sorgunun sonuçlarını ve oluşabilecek hata mesajlarını tutacak listeleri hazırlar.
    sonuc_listeleri = []
    hatalar = []

    # Tek bir sorguyu hata yakalayarak çalıştıran yardımcı iç fonksiyon.
    def _gorev(sorgu_metni):
        try:
            # İlgili sorgu için OpenAlex sonuçlarını çeker.
            kayitlar = _tek_sorgu_ham_cek(
                sorgu_metni,
                baslangic_yili,
                bitis_yili,
                sayfa_basi_adet,
                siralama_alani,
                sadece_acik_erisim,
                min_atif,
                ulke_kodu,
                dil_kodu,
                mailto=mailto,
            )
            # Başarılı durumda sonuç listesini ve hata yok (None) bilgisini döner.
            return kayitlar, None
        except Exception as exc:
            # Sorgu başarısız olursa boş liste ve hata açıklamasını döner (diğer sorguları durdurmaz).
            return [], (sorgu_metni, str(exc))

    # Sorgu sayısı kadar (en fazla 4) eş zamanlı iş parçacığı açarak sorguları aynı anda gönderir.
    isci_sayisi = min(4, max(1, len(sorgu_tuple)))
    with ThreadPoolExecutor(max_workers=isci_sayisi) as havuz:
        # map, sonuçları giriş sırasını koruyarak toplar.
        for kayitlar, hata in havuz.map(_gorev, sorgu_tuple):
            sonuc_listeleri.append(kayitlar)
            if hata:
                hatalar.append(hata)

    # Eğer tüm sorgular hata verip hiç sonuç dönmediyse istisna fırlatır (hatalı sonuç önbelleğe alınmaz).
    if hatalar and not any(sonuc_listeleri):
        ilk_sorgu, ilk_hata = hatalar[0]
        raise RuntimeError(f"OpenAlex sorguları başarısız oldu ('{ilk_sorgu}'): {ilk_hata}")

    # Başarılı sorgu sonuçlarını ve varsa kısmi uyarıları döndürür.
    return sonuc_listeleri, hatalar


# Çoklu sorgudan gelen makaleleri paralel çekip round-robin birleştiren ve tekrarları eleyen ana fonksiyon.
def makaleleri_getir(
    sorgu_listesi,
    baslangic_yili,
    bitis_yili,
    makale_sayisi,
    siralama_alani,
    sadece_acik_erisim=False,
    min_atif=0,
    ulke_kodu="ALL",
    dil_kodu="ALL",
    mailto="",
):
    # Sorgu listesi boşsa hemen boş liste döndürür.
    if not sorgu_listesi:
        return []

    # Başlangıç yılı bitiş yılından büyük girildiyse iki değeri güvenli biçimde yer değiştirir.
    if baslangic_yili > bitis_yili:
        baslangic_yili, bitis_yili = bitis_yili, baslangic_yili

    # Önbellek anahtarı olabilmesi için sorgu listesini temizlenmiş bir tuple'a çevirir.
    sorgu_tuple = tuple(str(s).strip() for s in sorgu_listesi if str(s).strip())
    if not sorgu_tuple:
        return []

    # Tekrarlar elendikten sonra istenen makale sayısına ulaşmak için sorgu başına çekilecek makale adedi.
    sayfa_basi_adet = min(20, max(int(makale_sayisi) + 3, 6))

    try:
        # Paralel ve önbellekli sorgu çekme fonksiyonunu çağırır.
        sorgu_sonuclari, hatalar = _paralel_sorgulari_cek(
            sorgu_tuple,
            int(baslangic_yili),
            int(bitis_yili),
            sayfa_basi_adet,
            siralama_alani,
            bool(sadece_acik_erisim),
            int(min_atif or 0),
            ulke_kodu,
            dil_kodu,
            (mailto or "").strip(),
        )
    except Exception as exc:
        # Tüm sorgular başarısız olduysa kullanıcıya anlaşılır bir uyarı gösterir.
        st.warning(f"OpenAlex veri tabanına erişilirken sorun oluştu: {exc}")
        return []

    # Kısmi hata varsa (örneğin 3 sorgudan 1'i başarısız olduysa) kullanıcıyı bilgilendirir.
    for hatali_sorgu, hata_metni in hatalar:
        st.warning(f"OpenAlex sorgusu başarısız oldu ('{hatali_sorgu}'): {hata_metni}")

    # Birleştirilecek özgün makaleleri ve tekrar kontrol kümelerini hazırlar.
    birlesik_makaleler = []
    gorulen_idler = set()
    gorulen_basliklar = set()

    # Her sorgunun 1., sonra 2., sonra 3. sonuçlarını dönüşümlü (round-robin) alarak dengeli dağılım sağlar.
    for grup in zip_longest(*sorgu_sonuclari):
        for makale in grup:
            # Daha kısa listelerden gelen boş (None) yuvaları atlar.
            if not makale:
                continue

            # Makalenin benzersiz OpenAlex ID'sini ve normalize edilmiş başlık anahtarını alır.
            makale_id = makale.get("id")
            baslik_anahtari = _baslik_anahtari(makale)

            # Aynı ID veya aynı başlıkla daha önce eklenmiş makaleyi tekrar eklemez.
            if makale_id and makale_id in gorulen_idler:
                continue
            if baslik_anahtari and baslik_anahtari in gorulen_basliklar:
                continue

            # Görülenler kümelerini günceller ve makaleyi sonuç listesine ekler.
            if makale_id:
                gorulen_idler.add(makale_id)
            if baslik_anahtari:
                gorulen_basliklar.add(baslik_anahtari)
            birlesik_makaleler.append(makale)

            # İstenen hedef makale sayısına ulaşıldıysa iç döngüyü bitirir.
            if len(birlesik_makaleler) >= makale_sayisi:
                break
        # Hedef makale sayısına ulaşıldıysa dış döngüyü de bitirir.
        if len(birlesik_makaleler) >= makale_sayisi:
            break

    # Sıralama ölçütü atıf sayısıysa birleşik listeyi en çok atıf alandan aza doğru sıralar.
    if siralama_alani == "Atıf Sayısı":
        birlesik_makaleler.sort(
            key=lambda m: m.get("cited_by_count") or 0,
            reverse=True,
        )

    # İstenen adet kadar makaleyi döndürür.
    return birlesik_makaleler[:makale_sayisi]


# ==============================================================================
# ÜST VERİ ÇIKARMA VE İSTATİSTİK
# ==============================================================================

# Bir OpenAlex kaydından arayüzde, kaynakçada ve grafiklerde kullanılacak temiz üst verileri çıkarır.
def makale_bilgisi_cikar(makale):
    # Makalenin yazar/kurum kayıtlarını güvenle alır (boşsa boş liste).
    yazar_kayitlari = makale.get("authorships") or []

    # Yazar adlarını boş olmayanlar arasından sırayla toplar.
    yazarlar = [
        (a.get("author") or {}).get("display_name", "").strip()
        for a in yazar_kayitlari
        if (a.get("author") or {}).get("display_name")
    ]

    # Makaledeki benzersiz yazar kurumu ülke kodlarını bir kümede toplar.
    ulkeler = set()
    for a in yazar_kayitlari:
        # Yazar kaydının doğrudan ülke listesini ekler.
        for kod in a.get("countries") or []:
            if kod:
                ulkeler.add(kod)
        # Yazarın bağlı olduğu kurumların ülke kodlarını ekler.
        for kurum in a.get("institutions") or []:
            kod = kurum.get("country_code")
            if kod:
                ulkeler.add(kod)

    # Birincil yayın konumunu (dergi ve varsa doğrudan PDF bağlantısını) güvenle okur.
    birincil_konum = makale.get("primary_location") or {}
    # Dergi/kaynak adını alır.
    kaynak = (birincil_konum.get("source") or {}).get("display_name")
    # Açık erişim bilgisini ve varsa doğrudan erişilebilen PDF/OA bağlantısını okur.
    oa_bilgi = makale.get("open_access") or {}
    pdf_link = birincil_konum.get("pdf_url") or oa_bilgi.get("oa_url") or ""
    # Ana makale bağlantısını (DOI varsa DOI, yoksa OpenAlex ID) tam URL biçimine getirir.
    ana_link = _baglanti_normallestir(makale.get("doi") or makale.get("id") or "")

    # Temizlenmiş üst veri sözlüğünü döndürür.
    return {
        "baslik": makale.get("title") or "Başlık yok",
        "yil": makale.get("publication_year"),
        "atif": makale.get("cited_by_count") or 0,
        "link": ana_link,
        "pdf_link": _baglanti_normallestir(pdf_link),
        "yazarlar": yazarlar,
        "dergi": kaynak or "Belirtilmemiş",
        "ulkeler": sorted(ulkeler),
        "acik_erisim": bool(oa_bilgi.get("is_oa")),
    }


# Makale listesindeki yazar kurumu ülkelerinin dağılımını çoktan aza sıralı sözlük olarak hesaplar.
def ulke_dagilimi_hesapla(sonuclar):
    # Ülke kodu -> makale sayısı eşleşmesini tutacak sayaç sözlüğü.
    sayac = {}
    # Her makalenin ülkelerini dolaşıp sayacı 1 artırır.
    for m in sonuclar:
        for kod in makale_bilgisi_cikar(m)["ulkeler"]:
            sayac[kod] = sayac.get(kod, 0) + 1
    # Makale sayısına göre büyükten küçüğe sıralı sözlük döndürür.
    return dict(sorted(sayac.items(), key=lambda kv: kv[1], reverse=True))


# ==============================================================================
# PARALEL ÖZGÜNLÜK (BENZER ÇALIŞMA) KONTROLÜ
# ==============================================================================

# Tek bir benzerlik sorgusunu hafifletilmiş alanlarla OpenAlex'te arayan iş parçacığı uyumlu yardımcı.
def _benzer_ham_cek(ingilizce_sorgu, istenen_adet, mailto=""):
    # Sadece temel alanları (select) isteyerek hızlı yanıt alır.
    veri = _openalex_get(
        {
            "search": ingilizce_sorgu.strip(),
            "per-page": istenen_adet,
            "sort": "relevance_score:desc",
            "select": OPENALEX_SELECT_BENZER,
        },
        mailto=mailto,
        timeout=10,
    )
    # Bulunan sonuç listesini döndürür.
    return veri.get("results", [])


# Birden fazla tez önerisinin benzer yayın kontrolünü paralel ve önbellekli biçimde yürütür.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def _toplu_benzer_cek(sorgu_tuple, istenen_adet, mailto):
    # Tek bir önerinin sorgusunu güvenli biçimde çalıştıran iç fonksiyon.
    def _gorev(sorgu_metni):
        # Boş sorgu geldiyse istek atmadan boş liste döner.
        if not sorgu_metni or not sorgu_metni.strip():
            return [], None
        try:
            # OpenAlex'ten benzer çalışmaları çeker.
            return _benzer_ham_cek(sorgu_metni, istenen_adet, mailto=mailto), None
        except Exception as exc:
            # Hata olursa boş liste ile birlikte uyarı bilgisini döndürür.
            return [], str(exc)

    # Öneri sayısı kadar (en fazla 4) eş zamanlı iş parçacığı ile kontrolleri aynı anda yapar.
    isci_sayisi = min(4, max(1, len(sorgu_tuple)))
    with ThreadPoolExecutor(max_workers=isci_sayisi) as havuz:
        sonuclar = list(havuz.map(_gorev, sorgu_tuple))
    return sonuclar


# Tüm tez önerileri için OpenAlex'te benzer yayınları paralel olarak tarar ve mevcut kaynakları hariç tutar.
def toplu_benzer_calismalari_ara(sorgu_listesi, adet=3, haric_idler=(), mailto=""):
    # Sorgu listesi boşsa boş sözlük döndürür.
    if not sorgu_listesi:
        return {}

    # Analizde zaten yer alan makalelerin ID'lerini hızlı arama için kümeye dönüştürür.
    haric = set(haric_idler or ())
    # Hariç tutulanlar elendikten sonra yeterli sonuç kalması için çekilecek kayıt sayısını belirler.
    istenen_adet = min(20, int(adet) + len(haric))
    # Önbellek anahtarı için sorguları tuple biçimine çevirir.
    sorgu_tuple = tuple((s or "").strip() for s in sorgu_listesi)

    # Tüm önerilerin benzerlik taramasını tek adımda paralel çalıştırır.
    ham_sonuclar = _toplu_benzer_cek(sorgu_tuple, istenen_adet, (mailto or "").strip())

    # Öneri indeksi -> benzer yayın listesi eşleşmesini oluşturur.
    ozgunluk_haritasi = {}
    for idx, (kayitlar, hata) in enumerate(ham_sonuclar):
        # Bir öneride ağ hatası oluştuysa kullanıcıya bilgi verir.
        if hata:
            st.warning(f"Öneri {idx + 1} için benzer çalışma kontrolü yapılamadı: {hata}")
        # Mevcut kaynak makaleleri eleyerek temiz benzer yayın listesini hazırlar.
        benzerler = []
        for k in kayitlar:
            # Analiz edilen kaynak makalenin kendisini "benzer yayın" olarak listelemez.
            if k.get("id") in haric:
                continue
            benzerler.append(
                {
                    "id": k.get("id"),
                    "baslik": k.get("title") or "Başlık yok",
                    "yil": k.get("publication_year") or "-",
                    "atif": k.get("cited_by_count") or 0,
                    "link": _baglanti_normallestir(k.get("doi") or k.get("id") or ""),
                }
            )
            # İstenen benzer yayın sayısına ulaşıldığında bu öneri için döngüyü bitirir.
            if len(benzerler) >= adet:
                break
        ozgunluk_haritasi[idx] = benzerler

    # Tüm tez önerilerinin benzerlik sonuç haritasını döndürür.
    return ozgunluk_haritasi


# Tek bir sorgu için benzer çalışma aramak isteyen yerler için geriye dönük uyumlu sarmalayıcı fonksiyon.
def benzer_calismalari_ara(ingilizce_sorgu, adet=3, haric_idler=(), mailto=""):
    # Toplu paralel arama fonksiyonunu tek elemanlı liste ile çağırır.
    harita = toplu_benzer_calismalari_ara([ingilizce_sorgu], adet=adet, haric_idler=haric_idler, mailto=mailto)
    # İlk ve tek elemanın sonucunu döndürür.
    return harita.get(0, [])

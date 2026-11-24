# -*- coding: utf-8 -*-
"""
app.py
Akademik Literatür ve Araştırma Boşluğu (Research Gap) Analiz Sistemi — Streamlit Arayüzü.
Herkesin kendi Google Gemini API anahtarıyla giriş yapabildiği (BYOK), adım adım API rehberli,
hızlı ve akademik dürüstlük doğrulamalı ana uygulama dosyası.
"""

# Raporun ham verisini JSON formatında dışa aktarmak için json modülünü içe aktarır.
import json
# İsteğe bağlı erişim şifresini sabit zamanlı (güvenli) karşılaştırmak için hmac modülünü içe aktarır.
import hmac
# Ortam değişkenlerini (isteğe bağlı APP_PASSWORD) okumak için os modülünü içe aktarır.
import os
# Metin kaçışı, kanıt vurgulama ve [1, 5] atıflarını linke çevirmek için re modülünü içe aktarır.
import re
# Analizin kaç saniyede tamamlandığını ölçüp ekranda göstermek için time modülünü içe aktarır.
import time
# Bulgu doğrulama durumlarını saymak için Counter sınıfını içe aktarır.
from collections import Counter

# Tablo ve dağılım grafikleri oluşturmak için pandas kütüphanesini içe aktarır.
import pandas as pd
# Web arayüzü ve oturum durumu yönetimi için Streamlit kütüphanesini içe aktarır.
import streamlit as st

# PDF, Word (.docx) ve BibTeX (.bib) dışa aktarma fonksiyonlarını disa_aktarma modülünden içe aktarır.
from disa_aktarma import (
    FPDF_MEVCUT,
    bibtex_olustur,
    pdf_rapor_olustur,
    word_rapor_olustur,
)
# OpenAlex veri çekme, özet düzeltme, üst veri çıkarma ve paralel benzerlik arama fonksiyonlarını içe aktarır.
from veri_cekme import (
    makale_bilgisi_cikar,
    makaleleri_getir,
    ozeti_duzelt,
    toplu_benzer_calismalari_ara,
    ulke_dagilimi_hesapla,
)
# Gemini BYOK doğrulama, sorgu/analiz üretimi, kanıt doğrulama, istatistik, kaynakça ve sohbet fonksiyonlarını içe aktarır.
from yapay_zeka import (
    MODEL_SECENEKLERI,
    TASARIM_ETIKETLERI,
    ULKE_ADLARI,
    VARSAYILAN_MODEL,
    api_anahtari_dogrula,
    birlesik_analiz_uret,
    bosluk_istatistikleri_hesapla,
    bulgulari_dogrula,
    istatistik_metni_olustur,
    kaynakca_olustur,
    kullanilabilir_modelleri_getir,
    makale_matrisi_uret,
    matrisi_dogrula,
    on_istatistik_metni_olustur,
    rapor_sohbeti_yanitla,
    sorgu_ayrintilari_uret,
    yapilandirilmis_analiz_uret,
)

# ==============================================================================
# SAYFA YAPILANDIRMASI
# ==============================================================================

# Tarayıcı sekmesi başlığını, ikonunu ve geniş (wide) sayfa düzenini belirler (ilk Streamlit çağrısı olmalıdır).
st.set_page_config(
    page_title="Akademik Literatür ve Boşluk Analizi",
    page_icon="📚",
    layout="wide",
)

# Kanıt doğrulama durumlarının ekranda ve indirilen raporda görünecek rozet ve açıklamaları.
DURUM_ROZETLERI = {
    "dogrulandi": ("✅", "Özetle doğrulandı"),
    "no_duzeltildi": ("🔁", "Alıntı başka bir makalede bulundu"),
    "dogrulanamadi": ("⚠️", "Alıntı özette bulunamadı"),
    "belirsiz": ("ℹ️", "Özetten anlaşılmıyor"),
}


# ==============================================================================
# OTURUM (SESSION STATE) BAŞLANGIÇ DEĞERLERİ
# ==============================================================================

# Her kullanıcı oturumunda yalnızca bellek (RAM) üzerinde izole tutulan değişkenlerin başlangıç değerleri.
VARSAYILANLAR = {
    "user_api_key": "",       # Kullanıcının kendi Google Gemini API anahtarı (asla diske kaydedilmez).
    "use_vertexai": False,    # Doğrulanan uç nokta (False: Gemini Developer API, True: Vertex AI Express).
    "secili_model": VARSAYILAN_MODEL if VARSAYILAN_MODEL in MODEL_SECENEKLERI else "gemini-flash-latest",
    "canli_modeller": {},     # Kullanıcının kendi API anahtarından canlı çekilen erişilebilir modeller.
    "ozel_model_adi": "",     # İsteğe bağlı: kullanıcının elle yazabileceği özel model kimliği.
    "analiz_paketi": None,    # Tamamlanmış son analizin tüm verileri.
    "sohbet_gecmisi": [],     # Rapor ve özetler üzerine yapılan sohbet geçmişi.
    "sorgu_taslagi": None,    # Üretilen ve kullanıcı onayını bekleyen İngilizce sorgular.
    "taslak_surum": 0,        # Her yeni sorgu üretiminde artar; metin kutularının eski değeri tutmasını önler.
    "aday_makaleler": None,   # OpenAlex'ten çekilen ve kullanıcının analiz öncesi seçip eleyeceği makale listesi.
    "aday_surum": 0,          # Her yeni makale çekiminde artar; onay kutularının (checkbox) taze kalmasını sağlar.
    "analiz_baslat": False,   # "Onay adımını atla" seçiliyse analizin tek tıkla başlamasını sağlar.
    "giris_yapildi": False,   # İsteğe bağlı APP_PASSWORD kullanılıyorsa doğrulama durumunu tutar.
    "yanlis_deneme": 0,       # Hatalı erişim şifresi deneme sayacı.
}
# Henüz oturumda yer almayan her anahtarı varsayılan değeriyle başlatır.
for anahtar, deger in VARSAYILANLAR.items():
    if anahtar not in st.session_state:
        st.session_state[anahtar] = deger


# ==============================================================================
# METİN VE MARKDOWN GÜVENLİK YARDIMCILARI
# ==============================================================================

# Markdown'da özel anlam taşıyan tüm karakterleri kaçırır (ham alıntı ve başlık gösteriminde kullanılır).
def md_kacis(metin):
    return re.sub(r"([\\`*_[\]$<>#~|])", r"\\\1", metin or "")


# Anlatı metinlerinde kalın/italik biçimi korurken yalnızca "$" işaretini kaçırarak hatalı LaTeX çizimini önler.
def anlati_guvenli_md(metin):
    # Zaten kaçırılmış "\$" varsa çift ters eğik çizgi oluşmaması için önce tek biçime indirger.
    return (metin or "").replace(r"\$", "$").replace("$", r"\$")


# Markdown bağlantı sözdizimini bozabilecek parantez ve boşlukları URL kodlamasına çevirir.
def guvenli_link(link):
    return (link or "").replace("(", "%28").replace(")", "%29").replace(" ", "%20")


# Orijinal özet metni içinde kanıt alıntısını bulup kalın (**) vurgular; noktalama ve tire farklarını tolere eder.
def ozeti_vurgula(ozet, alinti):
    # Üç nokta ile ayrılmış parça varsa böler.
    parcalar = [p.strip() for p in re.split(r"\.\.\.|…", alinti or "") if p and p.strip()]
    araliklar = []
    # Her alıntı parçasını orijinal özet içinde arar.
    for parca in parcalar:
        kelimeler = re.findall(r"\w+", parca)
        # Tek kelimelik rastgele eşleşmeleri vurgulamaz.
        if len(kelimeler) < 2:
            continue
        # Kelimeler arasında herhangi bir noktalama/boşluk bulunmasına izin veren desen kurar.
        desen = r"\W+".join(re.escape(k) for k in kelimeler)
        eslesme = re.search(desen, ozet, flags=re.IGNORECASE)
        if eslesme:
            araliklar.append(eslesme.span())

    # Eşleşen karakter aralıklarını sırayla işleyip kalın biçime sarar.
    araliklar.sort()
    sonuc, son = "", 0
    for bas, bit in araliklar:
        if bas < son:
            continue
        sonuc += md_kacis(ozet[son:bas]) + "**" + md_kacis(ozet[bas:bit]) + "**"
        son = bit
    return sonuc + md_kacis(ozet[son:])


# Metindeki [3] veya [1, 5] biçimindeki makale atıflarını ilgili makalelerin tıklanabilir bağlantılarına çevirir.
def atiflari_linkle(metin, linkler):
    # Önce anlatıdaki "$" işaretlerini güvenli hale getirir.
    guvenli_metin = anlati_guvenli_md(metin)

    # Bulunan her atıf grubundaki sayıları tek tek bağlantıya dönüştüren iç fonksiyon.
    def degistir(eslesme):
        parcalar = []
        for n in [int(x) for x in re.findall(r"\d+", eslesme.group(1))]:
            link = linkler.get(n)
            # Bağlantı varsa tıklanabilir [n] linki, yoksa düz [n] metni oluşturur.
            parcalar.append(f"[\\[{n}\\]]({link})" if link else f"\\[{n}\\]")
        return ", ".join(parcalar)

    # Metindeki tüm köşeli parantezli sayı gruplarını dönüştürür.
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", degistir, guvenli_metin)


# Markdown tablo hücresini bozabilecek dik çizgi (|) ve satır sonu karakterlerini temizler.
def md_hucre(metin):
    return str(metin if metin is not None else "").replace("|", "\\|").replace("\n", " ")


# DataFrame tablolarını hem yeni hem eski Streamlit sürümlerinde tam genişlikte sorunsuz gösterir.
def tablo_goster(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


# Analizdeki boşluk kaydını hem yeni {baslik, aciklama} hem eski düz metin biçiminden güvenle okur.
def bosluk_ayristir(madde):
    if isinstance(madde, dict):
        return str(madde.get("baslik", "")).strip(), str(madde.get("aciklama", "")).strip()
    return "", str(madde or "").strip()


# Makale matrisi listesini hem ekrandaki tablo hem indirilen rapor için okunaklı satır sözlüklerine dönüştürür.
def matris_satirlari_olustur(matris, bilgiler):
    satirlar = []
    for k in matris:
        no = k.get("no")
        baslik = bilgiler[no - 1]["baslik"] if isinstance(no, int) and 1 <= no <= len(bilgiler) else ""
        rozet = DURUM_ROZETLERI.get(k.get("kanit_durumu"), ("", ""))[0]
        satirlar.append(
            {
                "No": f"[{no}]",
                "Makale Başlığı": baslik,
                "Yöntem": k.get("yontem", "özetten anlaşılmıyor"),
                "Tasarım": TASARIM_ETIKETLERI.get(k.get("tasarim"), "belirsiz"),
                "Örneklem": k.get("orneklem", "özetten anlaşılmıyor"),
                "Ana Bulgu": k.get("ana_bulgu", "özetten anlaşılmıyor"),
                "Sınırlılık": k.get("sinirlilik", "özetten anlaşılmıyor"),
                "Kanıt": rozet,
            }
        )
    return satirlar


# ==============================================================================
# İSTEĞE BAĞLI ERİŞİM ŞİFRESİ (YÖNETİCİ TANIMLADIYSA)
# ==============================================================================

# İsteğe bağlı uygulama şifresini ortam değişkeninden veya Streamlit Secrets'tan okur.
def erisim_sifresi_al():
    sifre = os.environ.get("APP_PASSWORD", "").strip()
    if sifre:
        return sifre
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"]).strip()
    except Exception:
        pass
    return ""


# APP_PASSWORD tanımlıysa doğru şifre girilmeden uygulamayı açmaz; tanımlı değilse doğrudan BYOK ekranına geçer.
def erisim_kontrolu():
    sifre = erisim_sifresi_al()
    if not sifre or st.session_state.giris_yapildi:
        return

    st.title("🔒 Akademik Literatür ve Boşluk Analizi")
    # Kaba kuvvet denemelerine karşı 5 hatalı girişte kilitler.
    if st.session_state.yanlis_deneme >= 5:
        st.error("Çok fazla hatalı deneme yapıldı. Lütfen sayfayı yenileyin.")
        st.stop()

    girilen = st.text_input("Uygulama erişim şifresi:", type="password")
    if girilen:
        if hmac.compare_digest(girilen.encode("utf-8"), sifre.encode("utf-8")):
            st.session_state.giris_yapildi = True
            st.rerun()
        else:
            st.session_state.yanlis_deneme += 1
            st.error("Şifre hatalı.")
    st.stop()


# ==============================================================================
# ADIM ADIM GEMINI API ANAHTARI ALMA REHBERİ VE BYOK GİRİŞ EKRANI
# ==============================================================================

# Her kullanıcıya ücretsiz Google Gemini API anahtarını nasıl alacağını 1 dakikalık adımlarla anlatan rehber bileşeni.
def api_anahtari_rehberi_goster(acik_baslasin=True):
    with st.expander("📖 Ücretsiz Google Gemini API Anahtarı Nasıl Alınır? (1 Dakikalık Rehber)", expanded=acik_baslasin):
        st.markdown(
            """
            Bu uygulama **her kullanıcının kendi ücretsiz Google Gemini API anahtarı** ile çalışır.
            Böylece başkalarının kotasına takılmazsınız ve kendi araştırmalarınızı güvenle yürütebilirsiniz.

            **Adım Adım Anahtar Alma:**
            1. **[Google AI Studio — API Keys](https://aistudio.google.com/apikey)** sayfasını yeni sekmede açın ve Google (Gmail) hesabınızla giriş yapın.
            2. Sağ üst veya orta kısımdaki **"Create API key"** (API anahtarı oluştur) butonuna tıklayın.
            3. Açılan pencerede bir proje seçin (yoksa **"Create API key in new project"** seçeneğine tıklayın).
            4. Ekranda oluşan **`AIza...`** ile başlayan uzun anahtarı **"Copy"** butonuyla kopyalayın.
            5. Kopyaladığınız anahtarı buradaki **Google Gemini API Anahtarınız** kutusuna yapıştırıp giriş yapın.

            **💡 Hız, Anahtar Türü ve Ücretsiz Kota İpuçları:**
            - **Google AI Studio (`AIza...`)** anahtarları ücretsiz bireysel kullanım için en kolay seçenektir; **Vertex AI Express (`AQ....`)** anahtarları da sistem tarafından otomatik algılanır.
            - **"Doğrula, Modelleri Bul ve Giriş Yap"** butonuna bastığınızda sistem anahtarınızın erişebildiği gerçek modelleri Google'dan canlı olarak çeker ve çalışan en hızlı modeli otomatik seçer.
            - **🔒 Gizlilik Güvencesi:** Girdiğiniz API anahtarı hiçbir veritabanına veya dosyaya kaydedilmez; yalnızca tarayıcı oturumunuz açık kaldığı sürece geçici bellekte (RAM) tutulur ve sayfa kapandığında silinir.
            """
        )


# Kullanıcı henüz kendi API anahtarını girmediyse temiz ve rehberli giriş ekranını gösterir.
def api_anahtari_giris_ekrani():
    # Eğer oturumda geçerli bir API anahtarı zaten varsa giriş ekranını atlar.
    if st.session_state.user_api_key:
        return

    st.title("📚 Akademik Literatür ve Araştırma Boşluğu (Research Gap) Analizi")
    st.write(
        "Küresel akademik literatürü OpenAlex üzerinden saniyeler içinde tarayın, bulguları kanıt alıntılarıyla "
        "doğrulayın, metodolojik boşlukları keşfedin ve özgün tez önerileri geliştirin."
    )

    # API anahtarı giriş kartını çerçeveli kutu içinde sunar.
    with st.container(border=True):
        st.subheader("🔑 Başlamak için Kendi Gemini API Anahtarınızı Girin")
        st.caption(
            "Anahtarınız yalnızca bu oturumda kullanılır; kaydedilmez ve başkalarıyla paylaşılmaz. "
            "Doğrulama sırasında anahtarınıza açık modeller otomatik keşfedilir."
        )

        # API anahtarı giriş kutusunu sunar (AIza... veya AQ.... otomatik algılanır).
        girilen_anahtar = st.text_input(
            "Google Gemini API Anahtarı",
            type="password",
            placeholder="AIzaSy... (Google AI Studio) veya AQ.... (Vertex Express)",
            help="Google AI Studio'dan ücretsiz aldığınız API anahtarını buraya yapıştırın.",
        )

        # Doğrulamalı (otomatik model bulucu) giriş ve hızlı giriş butonlarını yan yana gösterir.
        btn_dogrula, btn_hizli, _ = st.columns([1.6, 1.2, 1.6])
        with btn_dogrula:
            dogrula_basildi = st.button("🚀 Doğrula, Modelleri Bul ve Giriş Yap", type="primary", use_container_width=True)
        with btn_hizli:
            hizli_basildi = st.button("⚡ Doğrulamadan Devam Et", use_container_width=True)

        # "Doğrula, Modelleri Bul ve Giriş Yap" tıklandıysa gerçek 1-tokenlik test ile çalışan modeli otomatik bulur.
        if dogrula_basildi:
            temiz_anahtar = (girilen_anahtar or "").strip()
            if not temiz_anahtar:
                st.warning("Lütfen önce Gemini API anahtarınızı yapıştırın.")
            else:
                with st.spinner("API anahtarı doğrulanıyor ve erişilebilir modeller keşfediliyor..."):
                    basarili, mesaj, calisan_model, canli_modeller, use_vertexai = api_anahtari_dogrula(temiz_anahtar)
                if basarili:
                    st.session_state.user_api_key = temiz_anahtar
                    st.session_state.use_vertexai = use_vertexai
                    st.session_state.canli_modeller = canli_modeller
                    st.session_state.secili_model = calisan_model
                    st.success(mesaj)
                    st.rerun()
                else:
                    st.error(f"Doğrulama başarısız: {mesaj}")

        # "Doğrulamadan Devam Et" tıklandıysa bile arka planda canlı model listesini çekmeye çalışır.
        if hizli_basildi:
            temiz_anahtar = (girilen_anahtar or "").strip()
            if not temiz_anahtar:
                st.warning("Lütfen önce Gemini API anahtarınızı yapıştırın.")
            else:
                with st.spinner("Anahtar kaydediliyor ve erişilebilir modeller kontrol ediliyor..."):
                    canli_modeller = kullanilabilir_modelleri_getir(temiz_anahtar, vertexai=False)
                st.session_state.user_api_key = temiz_anahtar
                st.session_state.use_vertexai = False
                st.session_state.canli_modeller = canli_modeller
                if canli_modeller:
                    st.session_state.secili_model = next(iter(canli_modeller.keys()))
                st.rerun()

    # İlk kez gelen kullanıcıların kolayca anahtar alabilmesi için rehberi açık olarak gösterir.
    api_anahtari_rehberi_goster(acik_baslasin=True)
    # Anahtar girilmeden uygulamanın geri kalanını çalıştırmaz.
    st.stop()


# ==============================================================================
# RAPOR VE DIŞA AKTARMA FONKSİYONLARI
# ==============================================================================

# Analiz paketindeki tüm bölümleri indirilebilir ve sohbet bağlamına uygun eksiksiz Markdown raporuna çevirir.
def tam_rapor_olustur(p):
    a = p["analiz_veri"]
    satir = []
    ekle = satir.append

    ekle(f"# Araştırma Konusu: {p['konu']}")
    ekle(
        "\n> Bu rapor yalnızca OpenAlex makale özetlerine dayanır; tam metinler taranmamıştır. "
        "Araştırma boşluğu iddiaları kesin yargı değil, başlangıç hipotezidir."
    )

    ekle("\n## Arama ve Analiz Ayarları")
    ekle("- İngilizce sorgular: " + " | ".join(p["sorgular"]))
    ekle("- Filtreler: " + p["filtre_ozeti"])
    ekle(f"- Kullanılan model / mod: {p['model_adi']} ({p['analiz_modu']})")
    ekle(f"- İncelenen özgün makale sayısı: {len(p['sonuclar'])}")

    if p["istatistik_metni"]:
        ekle("\n## Sayısal Özet\n" + p["istatistik_metni"])

    # 1. Genel Özet (giriş + tematik kalın maddeler)
    ekle("\n## 1. Genel Özet: Literatürde Şu Ana Kadar Neler Bulunmuş?")
    ekle(str(a.get("genel_ozet_giris", "")))
    for t in a.get("tematik_ozet_maddeleri") or []:
        if isinstance(t, dict):
            ekle(f"- **{t.get('tema', '')}:** {t.get('aciklama', '')}")

    # 2. Literatür Boşlukları (kalın başlıklı numaralı liste)
    ekle("\n## 2. Literatür Boşlukları (Research Gaps)")
    for idx, madde in enumerate(a.get("literatur_bosluklari") or [], 1):
        baslik, aciklama = bosluk_ayristir(madde)
        if baslik:
            ekle(f"{idx}. **{baslik}:** {aciklama}")
        else:
            ekle(f"{idx}. {aciklama}")

    # 3. Tez / Araştırma Önerileri
    ekle("\n## 3. Tez / Araştırma Önerileri")
    for idx, o in enumerate(a.get("tez_onerileri") or [], 1):
        ekle(f"\n### Öneri {idx}: {o.get('baslik', '')}")
        ekle(f"- **Temel Hipotez:** {o.get('hipotez', '')}")
        ekle(f"- **Metodoloji ve Örneklem:** {o.get('metodoloji_ve_orneklem', '')}")
        ekle(f"- **Zorluklar ve Sınırlılıklar:** {o.get('zorluklar_ve_sinirliliklar', '')}")
        benzerler = p["ozgunluk"].get(idx - 1) or []
        if benzerler:
            ekle("- **OpenAlex Benzer Yayın Kontrolü:**")
            for bz in benzerler:
                ekle(f"  - {bz['baslik']} ({bz['yil']}), atıf: {bz['atif']} {bz['link']}")

    # 4. Kanıt Alıntısıyla Sınanan Bulgular
    ekle("\n## 4. Kanıt Alıntısıyla Sınanan Bulgular")
    for b in p["bulgular"]:
        rozet, aciklama = DURUM_ROZETLERI[b["durum"]]
        no = b.get("dogru_makale_no") or b.get("makale_no")
        ekle(f"- **[{no}]** {b.get('bulgu_ozeti', '')} — {rozet} {aciklama}")
        if b.get("kanit_alintisi"):
            ekle(f'  - Kanıt alıntısı: "{b.get("kanit_alintisi")}"')

    # 5. Çelişkiler ve Ayrışmalar
    ekle("\n## 5. Çelişen ve Farklılaşan Noktalar\n" + str(a.get("celiskiler_ve_ayrismalar", "")))

    # 6. Kıyaslamalı Makale Matrisi
    if p["matris"]:
        ekle("\n## 6. Kıyaslamalı Makale Matrisi")
        basliklar = ["No", "Makale Başlığı", "Yöntem", "Tasarım", "Örneklem", "Ana Bulgu", "Sınırlılık", "Kanıt"]
        ekle("| " + " | ".join(basliklar) + " |")
        ekle("|" + "---|" * len(basliklar))
        for s in matris_satirlari_olustur(p["matris"], p["bilgiler"]):
            ekle("| " + " | ".join(md_hucre(s[b]) for b in basliklar) + " |")

    ekle("\n## Kaynakça (APA 7)\n" + p["apa_metni"])
    ekle("\n## Kaynakça (IEEE)\n" + p["ieee_metni"])
    return "\n".join(satir)


# Tüm analiz paketini makine tarafından okunabilir UTF-8 JSON baytlarına dönüştürür.
def json_disari_aktar(p):
    veri = {
        "konu": p["konu"],
        "sorgular": p["sorgular"],
        "filtreler": p["filtre_ozeti"],
        "model": p["model_adi"],
        "analiz_modu": p["analiz_modu"],
        "sure_saniye": p["sure_saniye"],
        "istatistik": p["istatistik"],
        "analiz": p["analiz_veri"],
        "bulgular_dogrulanmis": p["bulgular"],
        "matris_dogrulanmis": p["matris"],
        "ozgunluk_kontrolu": p["ozgunluk"],
        "kaynaklar": p["bilgiler"],
    }
    return json.dumps(veri, ensure_ascii=False, indent=2).encode("utf-8")


# ==============================================================================
# ANALİZ AKIŞI (BALANCED / HIZLI VE İKİ AŞAMALI DERİN MOD)
# ==============================================================================

# Onaylanan sorgularla (veya kullanıcının eleyip seçtiği makalelerle) analiz, kanıt doğrulama ve benzerlik kontrolünü yürütür.
def analizi_calistir(
    konu,
    sorgular,
    api_key,
    model_adi,
    analiz_modu,
    baslangic_yili,
    bitis_yili,
    makale_sayisi,
    siralama_alani,
    sadece_acik_erisim,
    min_atif,
    ulke_kodu,
    secili_ulke_adi,
    dil_kodu,
    secili_dil_adi,
    openalex_mailto,
    secili_makaleler=None,
):
    # Toplam süreyi kullanıcıya göstermek için kronometreyi başlatır.
    baslangic_zamani = time.perf_counter()
    # Tek istekli hızlı mod mu yoksa iki aşamalı derin mod mu seçildiğini belirler.
    tek_istek_modu = analiz_modu.startswith("⚡")
    # Makaleler önceden seçildiyse OpenAlex çekim adımını atladığı için toplam adım 1 azalır.
    hazir_makale_var = bool(secili_makaleler)
    toplam_adim = (2 if tek_istek_modu else 3) if hazir_makale_var else (3 if tek_istek_modu else 4)
    adim = 0

    with st.status("Analiz çalışıyor...", expanded=True) as akis:
        # ADIM 1: Makaleler önceden seçilmediyse OpenAlex'ten PARALEL ve hafif JSON ile çeker.
        if hazir_makale_var:
            sonuclar = list(secili_makaleler)
        else:
            adim += 1
            st.write(f"{adim}/{toplam_adim}: OpenAlex'ten makaleler paralel olarak çekiliyor...")
            sonuclar = makaleleri_getir(
                sorgular,
                baslangic_yili,
                bitis_yili,
                makale_sayisi,
                siralama_alani,
                sadece_acik_erisim=sadece_acik_erisim,
                min_atif=min_atif,
                ulke_kodu=ulke_kodu,
                dil_kodu=dil_kodu,
                mailto=openalex_mailto,
            )
        if not sonuclar:
            akis.update(label="Makale bulunamadı", state="error")
            st.error("Seçtiğiniz filtrelere uygun makale bulunamadı. Yıl aralığını veya filtreleri esnetip tekrar deneyin.")
            return False

        # Gemini'ye gönderilecek kompakt makale özetlerini, ham özet sözlüğünü ve yıl/ülke istatistiklerini hazırlar.
        makale_metinleri = ""
        ham_ozetler = {}
        yil_sayaclari = {}
        for i, m in enumerate(sonuclar, 1):
            ozet = ozeti_duzelt(m.get("abstract_inverted_index"))
            ham_ozetler[i] = ozet
            yil = m.get("publication_year")
            if yil:
                yil_sayaclari[yil] = yil_sayaclari.get(yil, 0) + 1
            makale_metinleri += (
                f"[{i}] {m.get('title')} ({yil}, Atıf: {m.get('cited_by_count', 0)})\n"
                f"Özet: {ozet}\n\n"
            )
        bilgiler = [makale_bilgisi_cikar(m) for m in sonuclar]
        ulke_dagilimi = ulke_dagilimi_hesapla(sonuclar)

        # ADIM 2: Seçilen moda göre yapay zeka analizini yürütür.
        if tek_istek_modu:
            # HIZLI DENGELİ MOD: Özetleri yalnızca BİR kez gönderip matris + analizi tek istekte alır.
            adim += 1
            st.write(f"{adim}/{toplam_adim}: Seçilen {len(sonuclar)} makale için matris, literatür sentezi ve boşluk analizi üretiliyor...")
            on_istatistik = on_istatistik_metni_olustur(len(sonuclar), yil_sayaclari, ulke_dagilimi)
            ham_matris, analiz_veri, analiz_hata = birlesik_analiz_uret(
                makale_metinleri,
                len(sonuclar),
                on_istatistik,
                api_key=api_key,
                model_adi=model_adi,
            )
            if analiz_hata or not analiz_veri:
                akis.update(label="Analiz üretilemedi", state="error")
                st.error(f"Analiz üretilemedi: {analiz_hata or 'Bilinmeyen hata'}")
                return False

            # Matrisi ve bulguları Python koduyla orijinal özetlere karşı doğrular; kesin tasarım sayımlarını hesaplar.
            matris = matrisi_dogrula(ham_matris, ham_ozetler)
            istatistik = bosluk_istatistikleri_hesapla(matris, ulke_dagilimi, toplam_makale=len(sonuclar))
            istatistik_metni = istatistik_metni_olustur(istatistik) if matris else ""
            bulgular = bulgulari_dogrula(analiz_veri.get("bulgular"), ham_ozetler)
        else:
            # İKİ AŞAMALI DERİN MOD: Önce matris + kesin Python tasarım sayımı, ardından bu sayımlarla derin analiz.
            adim += 1
            st.write(f"{adim}/{toplam_adim}: Makale matrisi çıkarılıyor ve tasarım sayımları hesaplanıyor...")
            ham_matris, matris_hata = makale_matrisi_uret(
                makale_metinleri,
                len(sonuclar),
                api_key=api_key,
                model_adi=model_adi,
            )
            if matris_hata:
                st.warning(f"Makale matrisi kısmen/tamamen üretilemedi ({matris_hata}); analiz devam ediyor.")
            matris = matrisi_dogrula(ham_matris, ham_ozetler)
            istatistik = bosluk_istatistikleri_hesapla(matris, ulke_dagilimi, toplam_makale=len(sonuclar))
            istatistik_metni = istatistik_metni_olustur(istatistik) if matris else ""

            adim += 1
            st.write(f"{adim}/{toplam_adim}: Kesin sayımlara dayalı literatür analizi ve tez önerileri hazırlanıyor...")
            analiz_veri, analiz_hata = yapilandirilmis_analiz_uret(
                makale_metinleri,
                len(sonuclar),
                istatistik_metni,
                api_key=api_key,
                model_adi=model_adi,
            )
            if analiz_hata or not analiz_veri:
                akis.update(label="Analiz üretilemedi", state="error")
                st.error(f"Analiz üretilemedi: {analiz_hata or 'Bilinmeyen hata'}")
                return False
            bulgular = bulgulari_dogrula(analiz_veri.get("bulgular"), ham_ozetler)

        # SON ADIM: Tüm tez önerilerinin özgünlük (benzer yayın) kontrolünü PARALEL yapar, kaynakça ve PDF/Word/BibTeX üretir.
        adim += 1
        st.write(f"{adim}/{toplam_adim}: Özgünlük kontrolü (paralel), kaynakça ve PDF/Word dışa aktarma hazırlanıyor...")
        kaynak_idleri = tuple(m.get("id") for m in sonuclar if m.get("id"))
        oneri_sorgulari = [
            oner.get("benzerlik_aramasi_ingilizce") or oner.get("baslik", "")
            for oner in (analiz_veri.get("tez_onerileri") or [])
        ]
        ozgunluk = toplu_benzer_calismalari_ara(
            oneri_sorgulari,
            adet=3,
            haric_idler=kaynak_idleri,
            mailto=openalex_mailto,
        )
        apa_metni, ieee_metni = kaynakca_olustur(sonuclar)

        # Atıf linkleme için makale numarası -> güvenli bağlantı sözlüğünü kurar.
        linkler = {i: guvenli_link(b["link"]) for i, b in enumerate(bilgiler, 1) if b["link"]}
        # Rapora eklenecek okunur filtre özetini hazırlar.
        filtre_ozeti = (
            f"{baslangic_yili}-{bitis_yili}; sıralama: {siralama_alani}; min. atıf: {min_atif}; "
            f"ülke: {secili_ulke_adi}; dil: {secili_dil_adi}; "
            f"açık erişim: {'evet' if sadece_acik_erisim else 'hayır'}"
        )
        # Geçen toplam süreyi saniye cinsinden hesaplar.
        sure_saniye = round(time.perf_counter() - baslangic_zamani, 1)

        # Tüm sonuçları tek oturum paketinde toplar.
        paket_yeni = {
            "konu": konu,
            "sorgular": list(sorgular),
            "filtre_ozeti": filtre_ozeti,
            "model_adi": model_adi,
            "analiz_modu": analiz_modu,
            "sure_saniye": sure_saniye,
            "sonuclar": sonuclar,
            "bilgiler": bilgiler,
            "makale_metinleri": makale_metinleri,
            "ham_ozetler": ham_ozetler,
            "yil_sayaclari": yil_sayaclari,
            "ulke_dagilimi": ulke_dagilimi,
            "istatistik": istatistik,
            "istatistik_metni": istatistik_metni,
            "analiz_veri": analiz_veri,
            "bulgular": bulgular,
            "matris": matris,
            "ozgunluk": ozgunluk,
            "apa_metni": apa_metni,
            "ieee_metni": ieee_metni,
            "linkler": linkler,
        }
        # Tam Markdown raporunu ve indirilebilir PDF, Word (.docx) ve BibTeX (.bib) dosyalarını bir kez üretip pakete kaydeder.
        paket_yeni["tam_rapor"] = tam_rapor_olustur(paket_yeni)
        paket_yeni["bib_bayt"] = bibtex_olustur(bilgiler)
        paket_yeni["docx_bayt"] = word_rapor_olustur(paket_yeni, DURUM_ROZETLERI, TASARIM_ETIKETLERI)
        if FPDF_MEVCUT:
            try:
                paket_yeni["pdf_bayt"] = pdf_rapor_olustur(paket_yeni, TASARIM_ETIKETLERI)
                paket_yeni["pdf_hata"] = None
            except Exception as pdf_exc:
                paket_yeni["pdf_bayt"] = None
                paket_yeni["pdf_hata"] = str(pdf_exc)
        else:
            paket_yeni["pdf_bayt"] = None
            paket_yeni["pdf_hata"] = "PDF için 'pip install fpdf2' kurulumu gereklidir."

        st.session_state.analiz_paketi = paket_yeni
        st.session_state.sohbet_gecmisi = []
        akis.update(label=f"Analiz {sure_saniye} saniyede tamamlandı!", state="complete", expanded=False)
        return True


# ==============================================================================
# GİRİŞ KAPILARI (ÖNCE İSTEĞE BAĞLI ŞİFRE, SONRA BYOK API ANAHTARI)
# ==============================================================================

# Tanımlıysa uygulama erişim şifresini kontrol eder.
erisim_kontrolu()
# Kullanıcının kendi Gemini API anahtarını girip doğrulamasını sağlar.
api_anahtari_giris_ekrani()


# ==============================================================================
# SOL MENÜ (SADE TEMEL AYARLAR + GELİŞMİŞ FİLTRELER)
# ==============================================================================

# Kullanıcının aktif API oturumu ve çıkış/anahtar değiştirme alanı.
st.sidebar.header("🔑 API ve Model Ayarları")
anahtar_etiketi = "Vertex AI Express" if st.session_state.use_vertexai else "Gemini API"
st.sidebar.success(f"✅ API anahtarınız aktif ({anahtar_etiketi}).")

# Kullanıcının anahtarından canlı çekilen modeller varsa onları, yoksa yedek listeyi gösterir.
aktif_model_haritasi = st.session_state.canli_modeller or MODEL_SECENEKLERI
model_listesi = list(aktif_model_haritasi.keys())
mevcut_model_idx = model_listesi.index(st.session_state.secili_model) if st.session_state.secili_model in model_listesi else 0
secili_model = st.sidebar.selectbox(
    "Gemini Modeli",
    options=model_listesi,
    index=mevcut_model_idx,
    format_func=lambda m: aktif_model_haritasi.get(m, m),
)

# İsteğe bağlı: listede olmayan özel bir model ID'si elle yazılabilir.
with st.sidebar.expander("✍️ Özel Model ID (İsteğe Bağlı)", expanded=False):
    ozel_model = st.text_input(
        "Model kimliği",
        value=st.session_state.ozel_model_adi,
        placeholder="Örn: gemini-2.5-flash",
        help="Boş bırakırsanız yukarıdaki listeden seçili model kullanılır.",
    ).strip()
    st.session_state.ozel_model_adi = ozel_model

# Özel model girildiyse onu, değilse listeden seçileni aktif model yapar.
etkin_model = st.session_state.ozel_model_adi or secili_model
st.session_state.secili_model = secili_model

# Tek tıkla anahtardaki canlı modelleri çekip gerçekten çalışan modeli test eden buton.
if st.sidebar.button("🔄 Modelleri Anahtarımdan Çek & Test Et", use_container_width=True):
    with st.spinner("Anahtarınıza açık modeller test ediliyor..."):
        basarili, mesaj, calisan_model, canli_modeller, use_vertexai = api_anahtari_dogrula(
            st.session_state.user_api_key,
            tercih_edilen_model=etkin_model,
        )
    if canli_modeller:
        st.session_state.canli_modeller = canli_modeller
    if basarili:
        st.session_state.use_vertexai = use_vertexai
        st.session_state.secili_model = calisan_model
        st.session_state.ozel_model_adi = ""
        st.sidebar.success(mesaj)
        st.rerun()
    else:
        st.sidebar.error(mesaj)

# Hız/derinlik dengesini belirleyen istek akışı seçici.
analiz_modu = st.sidebar.radio(
    "Analiz Akış Modu",
    options=[
        "⚡ Hızlı Dengeli Mod (Tek İstek — Önerilen)",
        "🔬 İki Aşamalı Derin Mod (Matris → Analiz)",
    ],
    index=0,
    help="Hızlı mod özetleri tek seferde göndererek ~2 kat hızlı çalışır ve API kotasını yarı yarıya korur.",
)

# API anahtarını oturumdan temizlemek veya başka bir anahtarla giriş yapmak için buton.
if st.sidebar.button("🚪 API Anahtarını Değiştir / Çıkış", use_container_width=True):
    st.session_state.user_api_key = ""
    st.session_state.use_vertexai = False
    st.session_state.canli_modeller = {}
    st.session_state.ozel_model_adi = ""
    st.session_state.analiz_paketi = None
    st.session_state.sorgu_taslagi = None
    st.session_state.aday_makaleler = None
    st.session_state.sohbet_gecmisi = []
    st.rerun()

# İhtiyaç duyulduğunda sol menüden de açılabilen API alma rehberi.
with st.sidebar:
    api_anahtari_rehberi_goster(acik_baslasin=False)

st.sidebar.divider()

# Temel arama ayarları (ilk taslaktaki gibi sade ve hemen ulaşılabilir).
st.sidebar.header("⚙️ Arama Ayarları")
baslangic_yili = st.sidebar.slider("Başlangıç Yılı", min_value=2010, max_value=2026, value=2019)
bitis_yili = st.sidebar.slider("Bitiş Yılı", min_value=2010, max_value=2026, value=2026)
makale_sayisi = st.sidebar.slider("Taranacak Makale Sayısı", min_value=3, max_value=15, value=9)

# Gelişmiş filtreler sol menüyü kalabalıklaştırmamak için düzenli bir açılır kutuda toplanır.
with st.sidebar.expander("🎛️ Gelişmiş Filtreler", expanded=False):
    siralama_alani = st.selectbox("Sıralama Ölçütü", options=["Alaka Düzeyi", "Atıf Sayısı"], index=0)
    min_atif = st.number_input("Minimum Atıf Sayısı", min_value=0, max_value=5000, value=0, step=5)

    # Yazar kurumu ülkesi seçenekleri (Türkçe ad -> ISO ülke kodu).
    ulke_secenekleri = {
        "Tüm Ülkeler": "ALL",
        "Türkiye": "TR",
        "Amerika Birleşik Devletleri (ABD)": "US",
        "Birleşik Krallık (İngiltere)": "GB",
        "Almanya": "DE",
        "Kanada": "CA",
        "Avustralya": "AU",
        "Hindistan": "IN",
        "Çin": "CN",
    }
    secili_ulke_adi = st.selectbox("Yazar Kurumu Ülkesi", options=list(ulke_secenekleri.keys()), index=0)
    ulke_kodu = ulke_secenekleri[secili_ulke_adi]

    # Yayın dili seçenekleri (Türkçe ad -> OpenAlex dil kodu).
    dil_secenekleri = {
        "Tüm Diller": "ALL",
        "İngilizce": "en",
        "Türkçe": "tr",
        "Almanca": "de",
        "Fransızca": "fr",
        "İspanyolca": "es",
    }
    secili_dil_adi = st.selectbox("Yayın Dili", options=list(dil_secenekleri.keys()), index=1)
    dil_kodu = dil_secenekleri[secili_dil_adi]

    # Açık erişim ve tek tıkla onay atlama seçenekleri.
    sadece_acik_erisim = st.checkbox("Yalnızca Açık Erişim (Open Access)", value=False)
    onayi_atla = st.checkbox("Sorgu onay adımını atla (tek tık analiz)", value=False)

    # İsteğe bağlı OpenAlex Polite Pool e-posta alanı (daha hızlı OpenAlex yanıtı için).
    openalex_mailto = st.text_input(
        "OpenAlex E-posta (İsteğe Bağlı)",
        value="",
        placeholder="ornek@universite.edu.tr",
        help="OpenAlex'in hızlı kuyruğunu (Polite Pool) kullanmak için isteğe bağlı e-posta girebilirsiniz.",
    )

st.sidebar.info(
    "📌 Seçtiğiniz yıl aralığındaki en alakalı özetler paralel olarak taranır; her bulgu orijinal özet alıntısıyla kodla doğrulanır."
)


# ==============================================================================
# ANA EKRAN: KONU GİRİŞİ VE SORGU ONAYI
# ==============================================================================

st.title("📚 Akademik Literatür ve Araştırma Boşluğu (Research Gap) Analizi")
st.write(
    "Türkçe araştırma konunuzu girin; sistem küresel literatürü tarasın, tematik özeti ve boşlukları çıkarsın, "
    "bulguları özet alıntılarıyla sınasın ve özgün tez önerileri üretsin."
)

# Akademik şeffaflık ve özet sınırlılığı uyarısı.
st.warning(
    "⚠️ **Akademik Metodoloji Uyarısı:** Bu sistemdeki analizler yalnızca OpenAlex'ten çekilen **makale özetlerine** "
    "dayanmaktadır; tam metinler taranmamıştır. Nihai akademik çalışmalarda tam metinlerin incelenmesi zorunludur."
)

# Kullanıcının araştırma konusunu yazdığı ana metin kutusu.
turkce_konu = st.text_input(
    "Araştırmak istediğiniz konu:",
    placeholder="Örn: Sosyal medyanın üniversite öğrencilerinin uyku düzenine etkisi",
)

# --- ADIM 1: İNGİLİZCE SORGU ÜRETİMİ ---
if st.button("🔎 Sorguları Üret", type="primary"):
    if baslangic_yili > bitis_yili:
        st.error("Başlangıç yılı, bitiş yılından büyük olamaz.")
    elif not turkce_konu.strip():
        st.warning("Lütfen önce bir araştırma konusu yazın.")
    else:
        with st.spinner("İngilizce akademik arama sorguları üretiliyor..."):
            ayrintilar, sorgu_hata = sorgu_ayrintilari_uret(
                turkce_konu,
                api_key=st.session_state.user_api_key,
                model_adi=etkin_model,
            )
        if sorgu_hata:
            st.error(f"Sorgu üretilemedi, konu doğrudan aratılacak (sonuçlar zayıf olabilir): {sorgu_hata}")
        # Yeni taslağı oturuma kaydeder; önceki analiz ve sohbeti sıfırlar.
        st.session_state.taslak_surum += 1
        st.session_state.sorgu_taslagi = {
            "konu": turkce_konu.strip(),
            "sorgular": ayrintilar,
            "surum": st.session_state.taslak_surum,
        }
        st.session_state.analiz_paketi = None
        st.session_state.aday_makaleler = None
        st.session_state.sohbet_gecmisi = []
        # Kullanıcı "Onayı atla" dediyse hemen analizi tetikler.
        st.session_state.analiz_baslat = bool(onayi_atla)

# --- ADIM 2: SORGU KONTROL VE ONAY EKRANI ---
duzenlenmis_sorgular = []
makale_getir_basildi = False
dogrudan_analiz_basildi = False
taslak = st.session_state.sorgu_taslagi

if taslak:
    # Makaleler henüz çekilmediyse ve analiz yoksa açık, sonraki aşamalarda otomatik kapalı görünür.
    sorgu_kutusu_acik = st.session_state.aday_makaleler is None and st.session_state.analiz_paketi is None
    with st.expander("🔎 1. Adım: Arama Sorgularını Kontrol Et ve Onayla", expanded=sorgu_kutusu_acik):
        st.caption(f"Sorgular şu konu için üretildi: **{taslak['konu']}**")
        st.caption("Konunun anlamı kaymışsa sorguları doğrudan düzenleyebilirsiniz; arama düzenlenmiş halleriyle yapılır.")
        for i, oge in enumerate(taslak["sorgular"]):
            deger = st.text_input(
                f"İngilizce Sorgu {i + 1}",
                value=oge["sorgu"],
                key=f"sorgu_{taslak['surum']}_{i}",
            )
            st.caption(f"↳ Türkçe karşılığı: {oge['turkce_karsiligi']}")
            duzenlenmis_sorgular.append(deger)
        btn_col1, btn_col2, _ = st.columns([1.6, 1.5, 1.5])
        with btn_col1:
            makale_getir_basildi = st.button("📥 Makaleleri Getir ve Seç (Önerilen)", type="primary", use_container_width=True)
        with btn_col2:
            dogrudan_analiz_basildi = st.button("⚡ Elemeden Doğrudan Analiz Et", use_container_width=True)

# Düzenlenen sorguları temizleyip tekrarsız listeye dönüştüren yardımcı.
def _temiz_sorgulari_al(ham_liste):
    temiz, gorulen = [], set()
    for s in ham_liste:
        s = " ".join(s.split())
        if s and s.lower() not in gorulen:
            gorulen.add(s.lower())
            temiz.append(s)
    return temiz

# --- ADIM 2A: MAKALELERİ ÖN İNCELEME VE ELEME İÇİN ÇEKME ---
if taslak and makale_getir_basildi:
    temiz_sorgular = _temiz_sorgulari_al(duzenlenmis_sorgular)
    if not temiz_sorgular:
        st.warning("En az bir arama sorgusu gerekli.")
    elif baslangic_yili > bitis_yili:
        st.error("Başlangıç yılı, bitiş yılından büyük olamaz.")
    else:
        with st.spinner("OpenAlex'ten makaleler paralel olarak getiriliyor..."):
            bulunanlar = makaleleri_getir(
                temiz_sorgular,
                baslangic_yili,
                bitis_yili,
                makale_sayisi,
                siralama_alani,
                sadece_acik_erisim=sadece_acik_erisim,
                min_atif=min_atif,
                ulke_kodu=ulke_kodu,
                dil_kodu=dil_kodu,
                mailto=openalex_mailto,
            )
        if not bulunanlar:
            st.error("Seçtiğiniz filtrelere uygun makale bulunamadı. Yıl aralığını veya filtreleri esnetip tekrar deneyin.")
        else:
            st.session_state.aday_surum += 1
            st.session_state.aday_makaleler = {
                "konu": taslak["konu"],
                "sorgular": temiz_sorgular,
                "makaleler": bulunanlar,
                "bilgiler": [makale_bilgisi_cikar(m) for m in bulunanlar],
                "ozetler": [ozeti_duzelt(m.get("abstract_inverted_index"), max_karakter=280) for m in bulunanlar],
                "surum": st.session_state.aday_surum,
            }
            st.session_state.analiz_paketi = None
            st.rerun()

# --- ADIM 2B: MAKALE SEÇME / ELEME EKRANI (HUMAN-IN-THE-LOOP) ---
aday_paketi = st.session_state.aday_makaleler
secili_analiz_basildi = False
secilen_makale_listesi = []

if aday_paketi:
    eleme_acik = st.session_state.analiz_paketi is None
    toplam_aday = len(aday_paketi["makaleler"])
    surum = aday_paketi["surum"]

    with st.expander(f"📋 2. Adım: Bulunan {toplam_aday} Makaleyi İncele ve İstemediklerini Ele", expanded=eleme_acik):
        st.caption(
            "Alakasız veya kapsam dışı bulduğun makalelerin tikini kaldırabilirsin. "
            "Yalnızca işaretli makaleler Gemini analizine gönderilir (kota ve doğruluk kazancı sağlar)."
        )

        # Tümünü seç / tümünü temizle hızlı kontrol butonları.
        hizli_1, hizli_2, hizli_3 = st.columns([1, 1, 2.5])
        with hizli_1:
            if st.button("✅ Tümünü Seç", key=f"tum_sec_{surum}", use_container_width=True):
                for idx in range(toplam_aday):
                    st.session_state[f"aday_sec_{surum}_{idx}"] = True
                st.rerun()
        with hizli_2:
            if st.button("⬜ Tümünü Temizle", key=f"tum_temizle_{surum}", use_container_width=True):
                for idx in range(toplam_aday):
                    st.session_state[f"aday_sec_{surum}_{idx}"] = False
                st.rerun()

        # Her aday makale için onay kutusu, temel üst veri ve kısa özet önizlemesi gösterir.
        for idx, (ham_m, bilgi, kisa_ozet) in enumerate(
            zip(aday_paketi["makaleler"], aday_paketi["bilgiler"], aday_paketi["ozetler"])
        ):
            anahtar = f"aday_sec_{surum}_{idx}"
            if anahtar not in st.session_state:
                st.session_state[anahtar] = True

            col_chk, col_detay = st.columns([0.06, 0.94])
            with col_chk:
                secili_mi = st.checkbox("Seç", key=anahtar, label_visibility="collapsed")
            with col_detay:
                baslik_md = md_kacis(bilgi["baslik"])
                baslik_link = f"[{baslik_md}]({guvenli_link(bilgi['link'])})" if bilgi["link"] else baslik_md
                pdf_rozet = f" · [📄 PDF]({guvenli_link(bilgi['pdf_link'])})" if bilgi.get("pdf_link") else ""
                oa_rozet = "🔓 Açık Erişim" if bilgi["acik_erisim"] else "🔒 Kapalı"
                yazarlar_kisa = ", ".join(bilgi["yazarlar"][:2]) + (" ve ark." if len(bilgi["yazarlar"]) > 2 else "")
                st.markdown(f"**[{idx + 1}]** {baslik_link} ({bilgi['yil'] or '-'}) — *Atıf: {bilgi['atif']}*{pdf_rozet}")
                st.caption(f"{yazarlar_kisa or 'Yazar yok'} · {bilgi['dergi']} · {oa_rozet}")
                st.caption(f"Özet önizleme: {anlati_guvenli_md(kisa_ozet)}")

            if secili_mi:
                secilen_makale_listesi.append(ham_m)

        with hizli_3:
            st.info(f"Seçili makale: **{len(secilen_makale_listesi)} / {toplam_aday}**")

        secili_analiz_basildi = st.button(
            f"🚀 Seçili {len(secilen_makale_listesi)} Makale ile Analizi Başlat",
            type="primary",
            disabled=len(secilen_makale_listesi) == 0,
        )

# --- ADIM 3: ANALİZİ ÇALIŞTIRMA (ELEMELİ VEYA DOĞRUDAN) ---
if aday_paketi and secili_analiz_basildi:
    if not secilen_makale_listesi:
        st.warning("Analiz için en az 1 makale seçmelisiniz.")
    else:
        tamamlandi = analizi_calistir(
            konu=aday_paketi["konu"],
            sorgular=aday_paketi["sorgular"],
            api_key=st.session_state.user_api_key,
            model_adi=etkin_model,
            analiz_modu=analiz_modu,
            baslangic_yili=baslangic_yili,
            bitis_yili=bitis_yili,
            makale_sayisi=len(secilen_makale_listesi),
            siralama_alani=siralama_alani,
            sadece_acik_erisim=sadece_acik_erisim,
            min_atif=min_atif,
            ulke_kodu=ulke_kodu,
            secili_ulke_adi=secili_ulke_adi,
            dil_kodu=dil_kodu,
            secili_dil_adi=secili_dil_adi,
            openalex_mailto=openalex_mailto,
            secili_makaleler=secilen_makale_listesi,
        )
        if tamamlandi:
            st.rerun()

elif taslak and (dogrudan_analiz_basildi or st.session_state.analiz_baslat):
    st.session_state.analiz_baslat = False
    temiz_sorgular = _temiz_sorgulari_al(duzenlenmis_sorgular)

    if not temiz_sorgular:
        st.warning("En az bir arama sorgusu gerekli.")
    elif baslangic_yili > bitis_yili:
        st.error("Başlangıç yılı, bitiş yılından büyük olamaz.")
    else:
        tamamlandi = analizi_calistir(
            konu=taslak["konu"],
            sorgular=temiz_sorgular,
            api_key=st.session_state.user_api_key,
            model_adi=etkin_model,
            analiz_modu=analiz_modu,
            baslangic_yili=baslangic_yili,
            bitis_yili=bitis_yili,
            makale_sayisi=makale_sayisi,
            siralama_alani=siralama_alani,
            sadece_acik_erisim=sadece_acik_erisim,
            min_atif=min_atif,
            ulke_kodu=ulke_kodu,
            secili_ulke_adi=secili_ulke_adi,
            dil_kodu=dil_kodu,
            secili_dil_adi=secili_dil_adi,
            openalex_mailto=openalex_mailto,
        )
        # Analiz başarıyla bittiyse sayfayı temiz biçimde yenileyerek sorgu onay kutusunu otomatik kapatır.
        if tamamlandi:
            st.rerun()


# ==============================================================================
# ANALİZ ÇIKTILARINI GÖSTERİM BÖLÜMÜ (İLK TASLAK ESTETİĞİ + DOĞRULAMA)
# ==============================================================================

paket = st.session_state.analiz_paketi
if paket:
    sonuclar = paket["sonuclar"]
    bilgiler = paket["bilgiler"]
    ham_ozetler = paket["ham_ozetler"]
    analiz_veri = paket["analiz_veri"]
    linkler = paket["linkler"]
    istatistik = paket["istatistik"]

    st.success(
        f"✅ Analiz **{paket['sure_saniye']} saniyede** tamamlandı! "
        f"({len(sonuclar)} özgün makale incelendi · Model: `{paket['model_adi']}`)"
    )

    # 1. GENEL ÖZET (İlk taslaktaki gibi giriş paragrafı + kalın tematik maddeler ve tıklanabilir [1, 5] atıfları)
    st.subheader("1. Genel Özet: Literatürde Şu Ana Kadar Neler Bulunmuş?")
    st.markdown(atiflari_linkle(str(analiz_veri.get("genel_ozet_giris", "özetten anlaşılmıyor")), linkler))
    for tema_oge in analiz_veri.get("tematik_ozet_maddeleri") or []:
        if isinstance(tema_oge, dict):
            tema_baslik = md_kacis(str(tema_oge.get("tema", "")).strip())
            tema_aciklama = atiflari_linkle(str(tema_oge.get("aciklama", "")).strip(), linkler)
            st.markdown(f"- **{tema_baslik}:** {tema_aciklama}")

    st.divider()

    # 2. LİTERATÜR BOŞLUKLARI (İlk taslaktaki gibi kalın başlıklı numaralı maddeler + Python sayım özeti)
    st.subheader("2. Literatür Boşlukları (Research Gaps)")
    if paket["istatistik_metni"]:
        st.info("📐 **Boşluk analizinin dayandığı sayımlar**\n\n" + paket["istatistik_metni"])
        st.caption("Sayımlar Python koduyla yapılır; çalışma tasarımı etiketlerini aşağıdaki Kıyaslamalı Makale Matrisinden kontrol edebilirsiniz.")
    for idx, madde in enumerate(analiz_veri.get("literatur_bosluklari") or [], 1):
        baslik, aciklama = bosluk_ayristir(madde)
        if baslik:
            st.markdown(f"{idx}. **{md_kacis(baslik)}:** {atiflari_linkle(aciklama, linkler)}")
        else:
            st.markdown(f"{idx}. {atiflari_linkle(aciklama, linkler)}")

    st.divider()

    # 3. TEZ / ARAŞTIRMA ÖNERİLERİ VE OPENALEX ÖZGÜNLÜK KONTROLÜ
    st.subheader("3. Yeni Tez / Araştırma Önerileri ve Özgünlük Kontrolü")
    for idx, oner in enumerate(analiz_veri.get("tez_onerileri") or []):
        with st.container(border=True):
            st.markdown(f"### Öneri {idx + 1}: {md_kacis(str(oner.get('baslik', '')))}")
            st.markdown(f"- **Temel Hipotez:** {atiflari_linkle(str(oner.get('hipotez', '')), linkler)}")
            st.markdown(f"- **Metodoloji ve Örneklem:** {atiflari_linkle(str(oner.get('metodoloji_ve_orneklem', '')), linkler)}")
            st.markdown(f"- **Olası Zorluklar ve Sınırlılıklar:** {atiflari_linkle(str(oner.get('zorluklar_ve_sinirliliklar', '')), linkler)}")

            benzer_liste = paket["ozgunluk"].get(idx) or []
            if benzer_liste:
                st.markdown("**🔎 OpenAlex Benzer Yayın Kontrolü (Boşluk Sınaması):**")
                for bz in benzer_liste:
                    baslik_md = md_kacis(bz["baslik"])
                    link_str = f"[{baslik_md}]({guvenli_link(bz['link'])})" if bz["link"] else baslik_md
                    st.markdown(f"  - {link_str} ({bz['yil']}) — *Atıf: {bz['atif']}*")
                st.caption("Benzer yayın çıkması önerinin özgün olmadığı anlamına gelmez; kapsam farkını başlık/özet üzerinden değerlendirin.")
            else:
                st.caption("Bu sorguyla OpenAlex'te yakın benzer yayın bulunamadı (bu durum özgünlüğü tek başına garanti etmez).")

    st.divider()

    # 4. KANIT ALINTISIYLA SINANAN BULGULAR (Halüsinasyon Savunması)
    st.subheader("4. Kanıt Alıntısıyla Sınanan Bulgular")
    bulgular = paket["bulgular"]
    sayac = Counter(b["durum"] for b in bulgular)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("✅ Doğrulandı", sayac.get("dogrulandi", 0))
    k2.metric("🔁 No Düzeltildi", sayac.get("no_duzeltildi", 0))
    k3.metric("⚠️ Doğrulanamadı", sayac.get("dogrulanamadi", 0))
    k4.metric("ℹ️ Belirsiz", sayac.get("belirsiz", 0))
    st.caption(
        "Otomatik doğrulama, alıntının ilgili makale özetinde birebir geçip geçmediğini denetler; "
        "yorum doğruluğunu '📖 Kaynakla kıyasla' bölümünden gözle inceleyebilirsiniz."
    )

    for b in bulgular:
        rozet, aciklama = DURUM_ROZETLERI[b["durum"]]
        yazilan_no = b.get("makale_no")
        no = b.get("dogru_makale_no") if b["durum"] == "no_duzeltildi" else yazilan_no
        etiket_no = atiflari_linkle(f"[{no}]", linkler)
        ek_not = f" (model [{yazilan_no}] yazmıştı)" if b["durum"] == "no_duzeltildi" else ""
        st.markdown(f"**{etiket_no}** {anlati_guvenli_md(b.get('bulgu_ozeti', ''))} — *{rozet} {aciklama}{ek_not}*")
        if b.get("kanit_alintisi"):
            st.caption(f'Kanıt alıntısı: "{md_kacis(b.get("kanit_alintisi"))}"')
        if isinstance(no, int) and 1 <= no <= len(sonuclar):
            with st.expander(f"📖 Kaynakla kıyasla: [{no}] {bilgiler[no - 1]['baslik'][:85]}"):
                baglantilar = []
                if bilgiler[no - 1]["link"]:
                    baglantilar.append(f"[🔗 Makale sayfasını aç]({guvenli_link(bilgiler[no - 1]['link'])})")
                if bilgiler[no - 1].get("pdf_link"):
                    baglantilar.append(f"[📄 Açık Erişim PDF]({guvenli_link(bilgiler[no - 1]['pdf_link'])})")
                if baglantilar:
                    st.markdown(" · ".join(baglantilar))
                st.markdown(ozeti_vurgula(ham_ozetler.get(no, ""), b.get("kanit_alintisi", "")))

    # 5. ÇELİŞEN VE AYRIŞAN NOKTALAR
    st.subheader("5. Çelişen ve Farklılaşan Noktalar")
    st.markdown(atiflari_linkle(str(analiz_veri.get("celiskiler_ve_ayrismalar", "özetten anlaşılmıyor")), linkler))

    # 6. DAĞILIM GRAFİKLERİ
    st.subheader("📊 Literatür Dağılım Grafikleri")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Yayın Yılına Göre**")
        yillar = sorted(paket["yil_sayaclari"].keys())
        if yillar:
            st.bar_chart(pd.DataFrame({"Makale": [paket["yil_sayaclari"][y] for y in yillar]}, index=[str(y) for y in yillar]))
    with g2:
        st.markdown("**Yazar Kurumu Ülkesine Göre**")
        ulkeler = list(paket["ulke_dagilimi"].items())[:10]
        if ulkeler:
            st.bar_chart(pd.DataFrame({"Makale": [s for _, s in ulkeler]}, index=[ULKE_ADLARI.get(k, k) for k, _ in ulkeler]))
    with g3:
        st.markdown("**Çalışma Tasarımına Göre**")
        if paket["matris"]:
            tasarimlar = istatistik["tasarimlar"]
            st.bar_chart(pd.DataFrame({"Makale": list(tasarimlar.values())}, index=[TASARIM_ETIKETLERI[t] for t in tasarimlar]))

    # 7. KIYASLAMALI MAKALE MATRİSİ
    if paket["matris"]:
        st.subheader("📋 Kıyaslamalı Makale Matrisi")
        tablo_goster(pd.DataFrame(matris_satirlari_olustur(paket["matris"], bilgiler)))

    # 8. OTOMATİK KAYNAKÇA VE DIŞA AKTARMA
    st.subheader("📚 Otomatik Kaynakça (APA 7 & IEEE)")
    sekme_apa, sekme_ieee = st.tabs(["APA 7", "IEEE"])
    with sekme_apa:
        st.markdown(paket["apa_metni"])
    with sekme_ieee:
        st.markdown(paket["ieee_metni"])
    st.caption("Kaynakça OpenAlex üst verisinden otomatik üretilmiştir; kullanmadan önce yazar adlarını ve biçimi gözden geçirin.")

    # İndirme Paneli (PDF, Word, BibTeX, Markdown ve JSON)
    st.subheader("📥 Raporu ve Kaynakçayı İndir")
    # İlk satır: Akademik sunum ve tez yazımı için PDF, Word ve BibTeX butonları.
    u1, u2, u3 = st.columns(3)
    with u1:
        if paket.get("pdf_bayt"):
            st.download_button(
                label="📄 PDF Rapor İndir (.pdf)",
                data=paket["pdf_bayt"],
                file_name="literatur_analiz_raporu.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.warning(f"PDF hazırlanamadı: {paket.get('pdf_hata') or 'fpdf2 eksik'}")
    with u2:
        # Eski oturum paketleriyle uyumluluk için gerekirse Word dosyasını anlık üretir.
        docx_veri = paket.get("docx_bayt") or word_rapor_olustur(paket, DURUM_ROZETLERI, TASARIM_ETIKETLERI)
        st.download_button(
            label="📝 Word Rapor İndir (.docx)",
            data=docx_veri,
            file_name="literatur_analiz_raporu.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with u3:
        bib_veri = paket.get("bib_bayt") or bibtex_olustur(bilgiler)
        st.download_button(
            label="📚 BibTeX Kaynakça (.bib)",
            data=bib_veri,
            file_name="kaynakca.bib",
            mime="application/x-bibtex",
            use_container_width=True,
        )

    # İkinci satır: Geliştirici ve arşivleme için Markdown ve ham JSON indirme butonları.
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Markdown Rapor (.md)",
            data=paket["tam_rapor"].encode("utf-8"),
            file_name="literatur_analiz_raporu.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Ham Veri (.json)",
            data=json_disari_aktar(paket),
            file_name="literatur_analiz_verisi.json",
            mime="application/json",
            use_container_width=True,
        )

    # İncelenen kaynak makalelerin detaylı listesi (varsa doğrudan PDF bağlantısıyla)
    with st.expander(f"📄 İncelenen {len(sonuclar)} Kaynak Makaleyi ve Linklerini Gör"):
        for i, b in enumerate(bilgiler, 1):
            baslik_md = md_kacis(b["baslik"])
            link_str = f"[{baslik_md}]({guvenli_link(b['link'])})" if b["link"] else baslik_md
            pdf_ek = f" · [📄 PDF]({guvenli_link(b['pdf_link'])})" if b.get("pdf_link") else ""
            st.markdown(f"**[{i}]** {link_str} ({b['yil']}) — *Atıf: {b['atif']}*{pdf_ek}")
            yazar_metni = ", ".join(b["yazarlar"][:3]) + (" ve ark." if len(b["yazarlar"]) > 3 else "")
            ulke_metni = ", ".join(ULKE_ADLARI.get(k, k) for k in b["ulkeler"]) or "ülke bilgisi yok"
            acik_erisim = "🔓 Açık erişim" if b["acik_erisim"] else "🔒 Kapalı erişim"
            st.caption(f"{yazar_metni or 'Yazar bilgisi yok'} · {b['dergi']} · {ulke_metni} · {acik_erisim}")

    # Akademik dürüstlük notu
    st.info(
        "📝 **Dürüstlük Notu:** Bu analiz yalnızca makale özetlerine dayanır, tam metinler taranmamıştır; "
        "araştırma boşluğu iddiaları kesin hüküm değil, başlangıç hipotezidir."
    )

    # 9. RAPOR VE ÖZETLER ÜZERİNE SOHBET
    st.subheader("💬 Rapor ve Özetler Üzerine Soru Sor")
    for msg in st.session_state.sohbet_gecmisi:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    kullanici_sorusu = st.chat_input("İncelenen özetler veya rapor hakkında bir soru yazın...")
    if kullanici_sorusu:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": kullanici_sorusu})
        with st.chat_message("user"):
            st.markdown(kullanici_sorusu)

        with st.chat_message("assistant"):
            with st.spinner("Özetler ve rapor inceleniyor..."):
                yanit_metni = rapor_sohbeti_yanitla(
                    soru=kullanici_sorusu,
                    rapor_metni=paket["tam_rapor"],
                    makale_metinleri=paket["makale_metinleri"],
                    gecmis_mesajlar=st.session_state.sohbet_gecmisi,
                    api_key=st.session_state.user_api_key,
                    model_adi=etkin_model,
                )
            st.markdown(yanit_metni)
        st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": yanit_metni})

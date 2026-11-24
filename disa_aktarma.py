# -*- coding: utf-8 -*-
"""
disa_aktarma.py
Analiz raporunu Türkçe karakter uyumlu PDF (.pdf), Microsoft Word (.docx)
ve Zotero/Mendeley/LaTeX uyumlu BibTeX (.bib) formatlarında dışa aktaran modül.
"""

# Bellekte geçici dosya (BytesIO) oluşturarak Word (.docx) arşivini diske yazmadan üretmek için io'yu içe aktarır.
import io
# Sistemde kurulu Türkçe destekli TrueType (.ttf) yazı tiplerini kontrol etmek için os modülünü içe aktarır.
import os
# Metin temizleme, emoji ayıklama ve BibTeX anahtarı üretme için re modülünü içe aktarır.
import re
# Unicode karakterlerini sadeleştirip güvenli BibTeX atıf anahtarları üretmek için unicodedata'yı içe aktarır.
import unicodedata
# Saf Python ile sıfır harici bağımlılıklı Word (.docx) paketi oluşturmak için zipfile'ı içe aktarır.
import zipfile
# Word (.docx) XML metinlerinde özel karakterleri (<, >, &) güvenle kaçırmak için escape'i içe aktarır.
from xml.sax.saxutils import escape

# PDF oluşturmak için hafif ve saf Python tabanlı fpdf2 kütüphanesini güvenle içe aktarır.
try:
    from fpdf import FPDF
    FPDF_MEVCUT = True
except ImportError:
    FPDF = None
    FPDF_MEVCUT = False


# ==============================================================================
# SABİTLER VE ORTAK YARDIMCILAR
# ==============================================================================

# PDF ve Word tasarımlarında kullanılan modern akademik renk paleti (RGB ve Hex).
LACIVERT_RGB = (27, 54, 93)
ORTA_MAVI_RGB = (43, 84, 126)
ACIK_MAVI_RGB = (237, 244, 250)
ACIK_GRI_RGB = (244, 246, 248)
KOYU_METIN_RGB = (30, 41, 59)
IKINCIL_METIN_RGB = (71, 85, 105)

LACIVERT_HEX = "1B365D"
ORTA_MAVI_HEX = "2B547E"
ACIK_MAVI_HEX = "EDF4FA"
ACIK_GRI_HEX = "F4F6F8"
KENARLIK_HEX = "CBD5E1"
KOYU_METIN_HEX = "1E293B"

# PDF'te emoji desteklemeyen standart yazı tiplerinde bozuk kutu (□) çıkmaması için net Türkçe metin rozetleri.
PDF_DURUM_ETIKETLERI = {
    "dogrulandi": "[DOĞRULANDI]",
    "no_duzeltildi": "[NO DÜZELTİLDİ]",
    "dogrulanamadi": "[DOĞRULANAMADI]",
    "belirsiz": "[BELİRSİZ]",
}

# Windows, Linux (Streamlit Cloud / Colab) ve macOS üzerinde Türkçe karakter destekleyen yazı tipi adayları.
FONT_ADAYLARI = [
    # Windows standart Arial ailesi (tam Türkçe UTF-8 desteği).
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\ariali.ttf"),
    # Windows standart Calibri ailesi.
    (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf", r"C:\Windows\Fonts\calibrii.ttf"),
    # Windows standart Segoe UI ailesi.
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\segoeuii.ttf"),
    # Linux / Streamlit Community Cloud / Google Colab DejaVu Sans ailesi.
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ),
    # Linux Liberation Sans ailesi.
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    ),
    # macOS Arial ailesi.
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ),
]


# Boşluk maddesini hem yeni {baslik, aciklama} hem eski düz metin biçiminden güvenle ayırır.
def _bosluk_ayristir(madde):
    if isinstance(madde, dict):
        return str(madde.get("baslik", "")).strip(), str(madde.get("aciklama", "")).strip()
    return "", str(madde or "").strip()


# PDF'te standart yazı tiplerinin çizemediği 4-baytlık emojileri ve kontrol karakterlerini temizler.
def _pdf_metin_temizle(metin, ascii_zorla=False):
    # Metni stringe çevirip satır sonlarını standartlaştırır.
    temiz = str(metin or "").replace("\r\n", "\n").replace("\r", "\n")
    # Temel Markdown yıldızlarını (**) düz metne sadeleştirir.
    temiz = temiz.replace("**", "")
    # BMP dışı emojileri (ord > 65535) ve yaygın sembolleri sadeleştirerek boş kutu (□) oluşmasını engeller.
    temiz = "".join(ch for ch in temiz if ord(ch) <= 0xFFFF)
    # Bazı Unicode ok/madde karakterlerini güvenli eşdeğerlere çevirir.
    temiz = temiz.replace("•", "-").replace("·", "|").replace("—", "-").replace("–", "-")
    # Eğer sistemde hiçbir TrueType font bulunamadıysa (yalnızca acil yedek) Türkçe harfleri ASCII'ye indirger.
    if ascii_zorla:
        cevir = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        temiz = temiz.translate(cevir).encode("latin-1", "replace").decode("latin-1")
    return temiz.strip()


# PDF'te çok uzun kesintisiz URL veya kelimelerin sayfa kenarından taşmasını engellemek için kırılım ekler.
def _uzun_kelimeleri_kir(metin, max_parca=65):
    parcalar = []
    for kelime in metin.split(" "):
        if len(kelime) <= max_parca:
            parcalar.append(kelime)
        else:
            # Uzun DOI/URL dizilerini belirli aralıklarla bölerek taşmayı önler.
            parcalar.append(" ".join(kelime[i : i + max_parca] for i in range(0, len(kelime), max_parca)))
    return " ".join(parcalar)


# Sistemde Türkçe karakter destekleyen ilk geçerli (normal, kalın, italik) TTF yazı tipi setini bulur.
def _turkce_font_bul():
    for normal, kalin, italik in FONT_ADAYLARI:
        if os.path.isfile(normal):
            return (
                normal,
                kalin if os.path.isfile(kalin) else normal,
                italik if os.path.isfile(italik) else normal,
            )
    return None


# ==============================================================================
# 1) PDF RAPOR OLUŞTURUCU (.pdf)
# ==============================================================================

# Analiz paketinden Türkçe karakter uyumlu, çok sayfalı ve akademik görünümlü PDF baytları üretir.
def pdf_rapor_olustur(paket, tasarim_etiketleri):
    # fpdf2 kurulu değilse anlaşılır bir hata fırlatır (app.py bunu nazikçe kullanıcıya gösterir).
    if not FPDF_MEVCUT:
        raise RuntimeError("PDF dışa aktarma için 'fpdf2' paketi gerekli. 'pip install fpdf2' komutuyla kurabilirsiniz.")

    # Sistemde Türkçe destekli TrueType font arar.
    font_yollari = _turkce_font_bul()
    unicode_font_var = font_yollari is not None
    font_adi = "TurkceFont" if unicode_font_var else "Helvetica"

    # Her sayfanın altında "Sayfa X / Y" bilgisi gösteren özel A4 PDF sınıfını tanımlar.
    class AkademikPDF(FPDF):
        def footer(self):
            # Sayfa altından 12 mm yukarı konumlanır.
            self.set_y(-12)
            self.set_font(font_adi, "I", 8)
            self.set_text_color(*IKINCIL_METIN_RGB)
            alt_bilgi = _pdf_metin_temizle(
                f"Akademik Literatür ve Araştırma Boşluğu Analizi  |  Sayfa {self.page_no()}/{{nb}}",
                ascii_zorla=not unicode_font_var,
            )
            self.cell(0, 6, alt_bilgi, align="C")

    # A4 dikey PDF nesnesini oluşturur ve sayfa kenar boşluklarını ayarlar.
    pdf = AkademikPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(left=15, top=15, right=15)

    # Türkçe TrueType yazı tiplerini PDF'e gömer (normal, kalın ve italik).
    if unicode_font_var:
        normal_yol, kalin_yol, italik_yol = font_yollari
        pdf.add_font(font_adi, "", normal_yol)
        pdf.add_font(font_adi, "B", kalin_yol)
        pdf.add_font(font_adi, "I", italik_yol)

    pdf.add_page()
    icerik_genisligi = pdf.w - pdf.l_margin - pdf.r_margin

    # Metni temizleyip güvenli biçimde çok satırlı yazan yardımcı fonksiyonlar.
    def t(metin):
        return _uzun_kelimeleri_kir(_pdf_metin_temizle(metin, ascii_zorla=not unicode_font_var))

    def bolum_basligi(baslik):
        pdf.ln(3)
        pdf.set_font(font_adi, "B", 12)
        pdf.set_text_color(*LACIVERT_RGB)
        pdf.multi_cell(icerik_genisligi, 7, t(baslik), new_x="LMARGIN", new_y="NEXT")
        # Başlığın altına ince lacivert ayırıcı çizgi çizer.
        y = pdf.get_y()
        pdf.set_draw_color(*ORTA_MAVI_RGB)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(2.5)

    def paragraf(metin, kalin=False, italik=False, boyut=9.5, renk=KOYU_METIN_RGB, bosluk=2.0):
        stil = "B" if kalin else ("I" if italik else "")
        pdf.set_font(font_adi, stil, boyut)
        pdf.set_text_color(*renk)
        pdf.multi_cell(icerik_genisligi, 5.2, t(metin), new_x="LMARGIN", new_y="NEXT")
        if bosluk:
            pdf.ln(bosluk)

    def madde(baslik, aciklama):
        pdf.set_font(font_adi, "B", 9.5)
        pdf.set_text_color(*LACIVERT_RGB)
        if baslik:
            pdf.multi_cell(icerik_genisligi, 5.2, t(f"- {baslik}:"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_adi, "", 9.3)
            pdf.set_text_color(*KOYU_METIN_RGB)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(icerik_genisligi - 3, 5.0, t(aciklama), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font(font_adi, "", 9.3)
            pdf.set_text_color(*KOYU_METIN_RGB)
            pdf.multi_cell(icerik_genisligi, 5.0, t(f"- {aciklama}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

    # --- ÜST LACİVERT BAŞLIK BANNERI ---
    pdf.set_fill_color(*LACIVERT_RGB)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font_adi, "B", 13.5)
    pdf.multi_cell(
        icerik_genisligi,
        7.5,
        t("AKADEMİK LİTERATÜR VE ARAŞTIRMA BOŞLUĞU ANALİZ RAPORU"),
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(font_adi, "I", 9.5)
    pdf.multi_cell(
        icerik_genisligi,
        6,
        t(f"Araştırma Konusu: {paket['konu']}"),
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    # --- METODOLOJİ VE DÜRÜSTLÜK UYARI KUTUSU ---
    pdf.set_fill_color(*ACIK_MAVI_RGB)
    pdf.set_text_color(*KOYU_METIN_RGB)
    pdf.set_font(font_adi, "I", 8.8)
    pdf.multi_cell(
        icerik_genisligi,
        4.8,
        t(
            "Metodoloji Notu: Bu rapor yalnızca OpenAlex'ten çekilen makale özetlerine dayanmaktadır; "
            "tam metinler taranmamıştır. Boşluk iddiaları kesin yargı değil, başlangıç hipotezidir."
        ),
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2.5)

    # --- ARAMA ÖZETİ ---
    paragraf(f"İngilizce Sorgular: {' | '.join(paket['sorgular'])}", boyut=8.8, renk=IKINCIL_METIN_RGB, bosluk=1)
    paragraf(
        f"Filtreler: {paket['filtre_ozeti']}  |  İncelenen Makale: {len(paket['sonuclar'])}  |  Model: {paket['model_adi']}",
        boyut=8.8,
        renk=IKINCIL_METIN_RGB,
        bosluk=2,
    )

    analiz = paket["analiz_veri"]

    # 1. GENEL ÖZET
    bolum_basligi("1. Genel Özet: Literatürde Şu Ana Kadar Neler Bulunmuş?")
    paragraf(analiz.get("genel_ozet_giris", "özetten anlaşılmıyor"))
    for tema_oge in analiz.get("tematik_ozet_maddeleri") or []:
        if isinstance(tema_oge, dict):
            madde(tema_oge.get("tema", ""), tema_oge.get("aciklama", ""))

    # 2. LİTERATÜR BOŞLUKLARI
    bolum_basligi("2. Literatür Boşlukları (Research Gaps)")
    if paket.get("istatistik_metni"):
        pdf.set_fill_color(*ACIK_GRI_RGB)
        pdf.set_font(font_adi, "", 8.8)
        pdf.set_text_color(*KOYU_METIN_RGB)
        pdf.multi_cell(
            icerik_genisligi,
            4.8,
            t("Sayısal Dağılım (Python Sayımı):\n" + paket["istatistik_metni"]),
            fill=True,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)
    for idx, b_madde in enumerate(analiz.get("literatur_bosluklari") or [], 1):
        baslik, aciklama = _bosluk_ayristir(b_madde)
        madde(f"{idx}. {baslik}" if baslik else f"{idx}. Boşluk", aciklama)

    # 3. TEZ / ARAŞTIRMA ÖNERİLERİ
    bolum_basligi("3. Yeni Tez / Araştırma Önerileri ve Özgünlük Kontrolü")
    for idx, oner in enumerate(analiz.get("tez_onerileri") or [], 1):
        paragraf(f"Öneri {idx}: {oner.get('baslik', '')}", kalin=True, boyut=10, renk=ORTA_MAVI_RGB, bosluk=1)
        madde("Temel Hipotez", oner.get("hipotez", ""))
        madde("Metodoloji ve Örneklem", oner.get("metodoloji_ve_orneklem", ""))
        madde("Olası Zorluklar ve Sınırlılıklar", oner.get("zorluklar_ve_sinirliliklar", ""))
        benzerler = paket.get("ozgunluk", {}).get(idx - 1) or []
        if benzerler:
            benzer_metin = "; ".join(f"{bz['baslik']} ({bz['yil']}, atıf: {bz['atif']})" for bz in benzerler)
            madde("OpenAlex Benzer Yayın Kontrolü", benzer_metin)
        pdf.ln(1.5)

    # 4. KANIT ALINTISIYLA SINANAN BULGULAR
    bolum_basligi("4. Kanıt Alıntısıyla Sınanan Bulgular")
    for b in paket.get("bulgular") or []:
        durum_etiket = PDF_DURUM_ETIKETLERI.get(b.get("durum"), "[BELİRSİZ]")
        no = b.get("dogru_makale_no") or b.get("makale_no")
        madde(f"[{no}] {durum_etiket}", b.get("bulgu_ozeti", ""))
        if b.get("kanit_alintisi"):
            paragraf(f'   Kanıt alıntısı: "{b.get("kanit_alintisi")}"', italik=True, boyut=8.5, renk=IKINCIL_METIN_RGB, bosluk=1.5)

    # 5. ÇELİŞEN VE AYRIŞAN NOKTALAR
    bolum_basligi("5. Çelişen ve Farklılaşan Noktalar")
    paragraf(analiz.get("celiskiler_ve_ayrismalar", "özetten anlaşılmıyor"))

    # 6. KIYASLAMALI MAKALE MATRİSİ (Her makale okunaklı kart satırı halinde)
    if paket.get("matris"):
        bolum_basligi("6. Kıyaslamalı Makale Matrisi")
        bilgiler = paket.get("bilgiler") or []
        for satir in paket["matris"]:
            no = satir.get("no")
            m_baslik = bilgiler[no - 1]["baslik"] if isinstance(no, int) and 1 <= no <= len(bilgiler) else ""
            tasarim = tasarim_etiketleri.get(satir.get("tasarim"), "belirsiz")
            durum_etiket = PDF_DURUM_ETIKETLERI.get(satir.get("kanit_durumu"), "")
            pdf.set_fill_color(*ACIK_GRI_RGB)
            pdf.set_font(font_adi, "B", 9.0)
            pdf.set_text_color(*LACIVERT_RGB)
            pdf.multi_cell(icerik_genisligi, 5.0, t(f"[{no}] {m_baslik}  {durum_etiket}"), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_adi, "", 8.6)
            pdf.set_text_color(*KOYU_METIN_RGB)
            detay = (
                f"Yöntem: {satir.get('yontem', '-')}  |  Tasarım: {tasarim}  |  Örneklem: {satir.get('orneklem', '-')}\n"
                f"Ana Bulgu: {satir.get('ana_bulgu', '-')}\n"
                f"Sınırlılık: {satir.get('sinirlilik', '-')}"
            )
            pdf.multi_cell(icerik_genisligi, 4.6, t(detay), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.8)

    # 7. KAYNAKÇA (APA 7 ve IEEE)
    bolum_basligi("7. Kaynakça (APA 7)")
    for satir_apa in (paket.get("apa_metni") or "").split("\n\n"):
        if satir_apa.strip():
            paragraf(satir_apa.replace("*", ""), boyut=8.6, bosluk=1.2)

    bolum_basligi("8. Kaynakça (IEEE)")
    for satir_ieee in (paket.get("ieee_metni") or "").split("\n\n"):
        if satir_ieee.strip():
            paragraf(satir_ieee.replace("*", ""), boyut=8.6, bosluk=1.2)

    # PDF çıktısını bayt dizisi olarak döndürür.
    cikti = pdf.output()
    return bytes(cikti)


# ==============================================================================
# 2) MICROSOFT WORD (.docx) RAPOR OLUŞTURUCU (SIFIR HARİCİ BAĞIMLILIK)
# ==============================================================================

# Word XML metin parçası (run) üreten yardımcı fonksiyon.
def _w_run(text, bold=False, italic=False, color=KOYU_METIN_HEX, size_pt=10.5):
    rpr = [
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>',
        f'<w:color w:val="{color}"/>',
        f'<w:sz w:val="{int(size_pt * 2)}"/>',
        f'<w:szCs w:val="{int(size_pt * 2)}"/>',
    ]
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    satirlar = str(text or "").split("\n")
    parcalar = []
    for idx, satir in enumerate(satirlar):
        if idx > 0:
            parcalar.append("<w:br/>")
        parcalar.append(f'<w:t xml:space="preserve">{escape(satir)}</w:t>')
    return f"<w:r><w:rPr>{''.join(rpr)}</w:rPr>{''.join(parcalar)}</w:r>"


# Word paragrafı oluşturan yardımcı fonksiyon.
def _w_p(runs, space_before=0, space_after=100, indent_left=0):
    ppr = [f'<w:spacing w:before="{space_before}" w:after="{space_after}" w:line="264" w:lineRule="auto"/>']
    if indent_left:
        ppr.append(f'<w:ind w:left="{indent_left}"/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{''.join(runs)}</w:p>"


# Word bölüm başlığı (altı çizgili lacivert) oluşturan yardımcı fonksiyon.
def _w_h1(baslik):
    ppr = (
        '<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="100"/>'
        f'<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="{ORTA_MAVI_HEX}"/></w:pBdr></w:pPr>'
    )
    return f"<w:p>{ppr}{_w_run(baslik, bold=True, color=LACIVERT_HEX, size_pt=13)}</w:p>"


# Word tablo hücresi oluşturan yardımcı fonksiyon.
def _w_tablo(basliklar, satirlar, genislikler):
    grid = "".join(f'<w:gridCol w:w="{g}"/>' for g in genislikler)
    ust = "".join(
        f'<w:tc><w:tcPr><w:tcW w:w="{g}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{LACIVERT_HEX}"/></w:tcPr>'
        f'{_w_p([_w_run(b, bold=True, color="FFFFFF", size_pt=9.0)], space_after=0)}</w:tc>'
        for b, g in zip(basliklar, genislikler)
    )
    tr_ler = [f"<w:tr><w:trPr><w:tblHeader/></w:trPr>{ust}</w:tr>"]
    for r_idx, satir in enumerate(satirlar):
        dolgu = ACIK_GRI_HEX if r_idx % 2 == 1 else "FFFFFF"
        hucreler = "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{g}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{dolgu}"/></w:tcPr>'
            f'{_w_p([_w_run(val, size_pt=8.5)], space_after=0)}</w:tc>'
            for val, g in zip(satir, genislikler)
        )
        tr_ler.append(f"<w:tr>{hucreler}</w:tr>")
    return (
        f'<w:tbl><w:tblPr><w:tblW w:w="{sum(genislikler)}" w:type="dxa"/>'
        f'<w:tblBorders><w:top w:val="single" w:sz="6" w:color="{LACIVERT_HEX}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:color="{LACIVERT_HEX}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:color="{KENARLIK_HEX}"/>'
        f'<w:insideV w:val="single" w:sz="4" w:color="{KENARLIK_HEX}"/></w:tblBorders>'
        f'<w:tblCellMar><w:top w:w="70" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
        f'<w:bottom w:w="70" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(tr_ler)}</w:tbl>"
    )


# Analiz paketinden eksiksiz ve şık bir Microsoft Word (.docx) dosyasını bellekte bayt olarak üretir.
def word_rapor_olustur(paket, durum_rozetleri, tasarim_etiketleri):
    analiz = paket["analiz_veri"]
    parcalar = []

    # Üst lacivert kapak kutusu.
    konu_alt_baslik = f"Konu: {paket['konu']}"
    parcalar.append(
        f'<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
        f'<w:tblCellMar><w:top w:w="160" w:type="dxa"/><w:left w:w="200" w:type="dxa"/>'
        f'<w:bottom w:w="160" w:type="dxa"/><w:right w:w="200" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f'<w:tblGrid><w:gridCol w:w="9360"/></w:tblGrid><w:tr><w:tc>'
        f'<w:tcPr><w:tcW w:w="9360" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{LACIVERT_HEX}"/></w:tcPr>'
        f'{_w_p([_w_run("AKADEMİK LİTERATÜR VE ARAŞTIRMA BOŞLUĞU ANALİZ RAPORU", bold=True, color="FFFFFF", size_pt=14)], space_after=60)}'
        f'{_w_p([_w_run(konu_alt_baslik, italic=True, color="DCE7F5", size_pt=10.5)], space_after=0)}'
        f"</w:tc></w:tr></w:tbl>"
    )
    parcalar.append(
        _w_p(
            [
                _w_run(
                    "Metodoloji Notu: Bu rapor yalnızca OpenAlex makale özetlerine dayanır; tam metinler taranmamıştır. "
                    "Boşluk iddiaları kesin yargı değil, başlangıç hipotezidir.",
                    italic=True,
                    color="475569",
                    size_pt=9.5,
                )
            ],
            space_before=120,
        )
    )
    parcalar.append(
        _w_p(
            [
                _w_run("Sorgular: ", bold=True, color=LACIVERT_HEX, size_pt=9.5),
                _w_run(" | ".join(paket["sorgular"]), size_pt=9.5),
                _w_run(f"\nFiltreler: {paket['filtre_ozeti']} | Makale Sayısı: {len(paket['sonuclar'])} | Model: {paket['model_adi']}", size_pt=9.0, color="475569"),
            ]
        )
    )

    # 1. Genel Özet
    parcalar.append(_w_h1("1. Genel Özet: Literatürde Şu Ana Kadar Neler Bulunmuş?"))
    parcalar.append(_w_p([_w_run(analiz.get("genel_ozet_giris", "özetten anlaşılmıyor"))]))
    for tema_oge in analiz.get("tematik_ozet_maddeleri") or []:
        if isinstance(tema_oge, dict):
            parcalar.append(
                _w_p(
                    [
                        _w_run(f"• {tema_oge.get('tema', '')}: ", bold=True, color=LACIVERT_HEX),
                        _w_run(tema_oge.get("aciklama", "")),
                    ],
                    indent_left=260,
                )
            )

    # 2. Literatür Boşlukları
    parcalar.append(_w_h1("2. Literatür Boşlukları (Research Gaps)"))
    if paket.get("istatistik_metni"):
        parcalar.append(_w_p([_w_run("Sayısal Özet (Python Sayımı):\n" + paket["istatistik_metni"], italic=True, size_pt=9.5, color="334155")]))
    for idx, b_madde in enumerate(analiz.get("literatur_bosluklari") or [], 1):
        baslik, aciklama = _bosluk_ayristir(b_madde)
        if baslik:
            parcalar.append(_w_p([_w_run(f"{idx}. {baslik}: ", bold=True, color=LACIVERT_HEX), _w_run(aciklama)]))
        else:
            parcalar.append(_w_p([_w_run(f"{idx}. {aciklama}")]))

    # 3. Tez Önerileri
    parcalar.append(_w_h1("3. Yeni Tez / Araştırma Önerileri ve Özgünlük Kontrolü"))
    for idx, oner in enumerate(analiz.get("tez_onerileri") or [], 1):
        parcalar.append(_w_p([_w_run(f"Öneri {idx}: {oner.get('baslik', '')}", bold=True, color=ORTA_MAVI_HEX, size_pt=11)], space_before=100))
        parcalar.append(_w_p([_w_run("• Temel Hipotez: ", bold=True), _w_run(oner.get("hipotez", ""))], indent_left=240))
        parcalar.append(_w_p([_w_run("• Metodoloji ve Örneklem: ", bold=True), _w_run(oner.get("metodoloji_ve_orneklem", ""))], indent_left=240))
        parcalar.append(_w_p([_w_run("• Zorluklar ve Sınırlılıklar: ", bold=True), _w_run(oner.get("zorluklar_ve_sinirliliklar", ""))], indent_left=240))
        benzerler = paket.get("ozgunluk", {}).get(idx - 1) or []
        if benzerler:
            benzer_satir = "\n".join(f"  - {bz['baslik']} ({bz['yil']}, atıf: {bz['atif']}) {bz['link']}" for bz in benzerler)
            parcalar.append(_w_p([_w_run("• OpenAlex Benzer Yayınlar:\n", bold=True), _w_run(benzer_satir, size_pt=9.2)], indent_left=240))

    # 4. Bulgular ve Kanıtlar
    parcalar.append(_w_h1("4. Kanıt Alıntısıyla Sınanan Bulgular"))
    for b in paket.get("bulgular") or []:
        rozet, aciklama = durum_rozetleri.get(b.get("durum"), ("", ""))
        no = b.get("dogru_makale_no") or b.get("makale_no")
        parcalar.append(
            _w_p(
                [
                    _w_run(f"[{no}] ", bold=True, color=LACIVERT_HEX),
                    _w_run(b.get("bulgu_ozeti", "")),
                    _w_run(f" — {rozet} {aciklama}", italic=True, color="475569", size_pt=9.5),
                ]
            )
        )
        if b.get("kanit_alintisi"):
            parcalar.append(_w_p([_w_run(f'Kanıt alıntısı: "{b.get("kanit_alintisi")}"', italic=True, color="475569", size_pt=9.0)], indent_left=280))

    # 5. Çelişkiler
    parcalar.append(_w_h1("5. Çelişen ve Farklılaşan Noktalar"))
    parcalar.append(_w_p([_w_run(analiz.get("celiskiler_ve_ayrismalar", "özetten anlaşılmıyor"))]))

    # 6. Kıyaslamalı Makale Matrisi Tablosu
    if paket.get("matris"):
        parcalar.append(_w_h1("6. Kıyaslamalı Makale Matrisi"))
        bilgiler = paket.get("bilgiler") or []
        tablo_satirlari = []
        for k in paket["matris"]:
            no = k.get("no")
            baslik = bilgiler[no - 1]["baslik"] if isinstance(no, int) and 1 <= no <= len(bilgiler) else ""
            rozet = durum_rozetleri.get(k.get("kanit_durumu"), ("", ""))[0]
            tablo_satirlari.append(
                [
                    f"[{no}]",
                    baslik,
                    k.get("yontem", "-"),
                    tasarim_etiketleri.get(k.get("tasarim"), "belirsiz"),
                    k.get("orneklem", "-"),
                    k.get("ana_bulgu", "-"),
                    rozet,
                ]
            )
        parcalar.append(
            _w_tablo(
                ["No", "Başlık", "Yöntem", "Tasarım", "Örneklem", "Ana Bulgu", "Kanıt"],
                tablo_satirlari,
                [560, 2100, 1250, 1050, 1200, 2550, 650],
            )
        )

    # 7. Kaynakçalar
    parcalar.append(_w_h1("7. Kaynakça (APA 7)"))
    for satir in (paket.get("apa_metni") or "").split("\n\n"):
        if satir.strip():
            parcalar.append(_w_p([_w_run(satir.replace("*", ""), size_pt=9.5)]))

    parcalar.append(_w_h1("8. Kaynakça (IEEE)"))
    for satir in (paket.get("ieee_metni") or "").split("\n\n"):
        if satir.strip():
            parcalar.append(_w_p([_w_run(satir.replace("*", ""), size_pt=9.5)]))

    # OpenXML Word paket dosyalarını bellekte (BytesIO) birleştirir.
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(parcalar)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1080" w:right="1260" w:bottom="1080" w:left="1260"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )

    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        zf.writestr("_rels/.rels", rels.encode("utf-8"))
        zf.writestr("word/document.xml", document_xml.encode("utf-8"))
    return tampon.getvalue()


# ==============================================================================
# 3) BIBTEX (.bib) KAYNAKÇA OLUŞTURUCU (ZOTERO / MENDELEY / LATEX UYUMLU)
# ==============================================================================

# Bir metni BibTeX atıf anahtarına uygun yalın küçük ASCII harflere dönüştürür (Türkçe ı/İ/ş/ğ/ü/ö/ç harflerini koruyarak çevirir).
def _bibtex_anahtar_temizle(metin):
    turkce_cevir = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    sade = str(metin or "").translate(turkce_cevir)
    norm = unicodedata.normalize("NFKD", sade).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]", "", norm).lower()


# İncelenen makalelerden standart BibTeX (.bib) kaynakça dosyasını UTF-8 bayt olarak üretir.
def bibtex_olustur(bilgiler):
    kayitlar = []
    kullanilan_anahtarlar = set()

    for idx, b in enumerate(bilgiler or [], 1):
        yazarlar = b.get("yazarlar") or []
        yil = str(b.get("yil") or "nd")
        baslik = (b.get("baslik") or "Baslik yok").replace("{", "").replace("}", "").strip()
        dergi = b.get("dergi") or ""
        link = b.get("link") or ""

        # İlk yazarın soyadı + yıl + başlıktaki ilk anlamlı kelimeden benzersiz BibTeX anahtarı üretir.
        ilk_soyad = _bibtex_anahtar_temizle(yazarlar[0].split()[-1]) if yazarlar else f"kaynak{idx}"
        baslik_kelime = next((_bibtex_anahtar_temizle(k) for k in baslik.split() if len(_bibtex_anahtar_temizle(k)) >= 4), "makale")
        temel_anahtar = f"{ilk_soyad}{yil}{baslik_kelime}"
        anahtar = temel_anahtar
        ek = 2
        while anahtar in kullanilan_anahtarlar:
            anahtar = f"{temel_anahtar}_{ek}"
            ek += 1
        kullanilan_anahtarlar.add(anahtar)

        # DOI bağlantısından saf DOI kimliğini (10.xxxx/...) çıkarır.
        doi_degeri = ""
        if "doi.org/" in link:
            doi_degeri = link.split("doi.org/", 1)[1].strip()

        # BibTeX alanlarını oluşturur.
        alanlar = [
            f"  author = {{{' and '.join(yazarlar) if yazarlar else 'Unknown'}}}",
            f"  title = {{{{{baslik}}}}}",
            f"  year = {{{yil}}}",
        ]
        if dergi and dergi != "Belirtilmemiş":
            alanlar.append(f"  journal = {{{dergi}}}")
        if doi_degeri:
            alanlar.append(f"  doi = {{{doi_degeri}}}")
        if link:
            alanlar.append(f"  url = {{{link}}}")

        kayitlar.append(f"@article{{{anahtar},\n" + ",\n".join(alanlar) + "\n}")

    return ("\n\n".join(kayitlar) + "\n").encode("utf-8")

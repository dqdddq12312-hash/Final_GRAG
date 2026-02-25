import re
import unicodedata
from typing import Optional, Union
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image

# TEXT CLEANING

def remove_space(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)

def is_in_header_zone(bbox, page_height, header_ratio, footer_ratio):
    origin = str(bbox.get("coord_origin", "BOTTOMLEFT")).upper()
    t = float(bbox.get("t", 0))
    b = float(bbox.get("b", 0))

    if origin == "BOTTOMLEFT":
        is_header = t >= page_height * (1.0 - header_ratio)
        is_footer = b <= page_height * footer_ratio
    else: 
        is_header = t <= page_height * header_ratio
        is_footer = b >= page_height * (1.0 - footer_ratio)

    return is_header, is_footer

def remove_header_footer(text, bbox = None, page_height = None, header_ratio = 0.08, footer_ratio = 0.08):
    if bbox is None:
        return text
    if page_height is None or page_height <= 0:
        return text

    if isinstance(bbox, list):
        bboxes = bbox
    else:
        bboxes = [bbox]

    for bb in bboxes:
        is_header, is_footer = is_in_header_zone(bb, page_height, header_ratio, footer_ratio)
        if is_header or is_footer:
            return ""

    return text

def remove_non_informative_chunks(text):
    # Remove các chunks quá ngắn
    words = text.split()
    if len(words) < 5:
        return None
    
    # Remove các chunks có tỷ lệ chữ cái thấp (< 30%)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return None

    # Remove các chunks chứa các ký tự lặp lại
    if len(set(text.replace(' ', ''))) < 3:
        return None

    return text

def clean_text_chunk(text, bbox = None, page_height = None, header_ratio = 0.08, footer_ratio = 0.08):
    if not text:
        return text

    text = normalize_unicode(text)
    text = remove_header_footer(
        text,
        bbox=bbox,
        page_height=page_height,
        header_ratio=header_ratio,
        footer_ratio=footer_ratio,
    )
    text = remove_space(text)

    return text


#  TABLE CLEANING

def clean_table_cell(value):
    if pd.isna(value):
        return np.nan
    
    text = str(value)
    text = remove_space(text)
    
    text = re.sub(r'^[\-–—]+$', '', text)  # Loại bỏ các dòng chỉ chứa dấu "-"
    text = re.sub(r'^\s*[\*†‡§¶#]+\s*$', '', text)  # Loại bỏ các ký hiệu chú thích
    
    return text

def clean_numeric_value(value):
    if pd.isna(value) or value == '':
        return None
    
    text = str(value).strip()
    
    # Xóa các ký hiệu tiền tệ
    text = re.sub(r'[€$£¥₹]', '', text)
    
    # Xóa các đơn vị tiền tệ
    text = re.sub(r'\s*(VND|USD|EUR|GBP|JPY)\s*', '', text, flags=re.IGNORECASE)
    
    # Xử lý phần trăm
    if '%' in text:
        text = text.replace('%', '').strip()
        try:
            return float(text.replace(',', '')) / 100 
        except ValueError:
            return None
    
    # Xóa các đơn vị khác
    text = re.sub(
        r'([0-9.,]+)\s*'
        r'(tCO2e?|kg|tons?|GJ|MWh|kWh|m[²³]|ha|hectares?|'
        r'liters?|gallons?|persons?|employees?|people|'
        r'million|billion|trillion|thousand)',
        r'\1', text, flags=re.IGNORECASE
    )
    
    text = text.replace(' ', '')  # 1 234.56 → 1234.56
    text = text.replace(',', '')  # 1,234.56 → 1234.56
    
    # Chuyển đổi thành float
    try:
        return float(text)
    except ValueError:
        return None

def normalize_numeric_columns(df):
    df = df.copy()

    new_cols = {}
    for i, col in enumerate(df.columns):
        col_values = df.iloc[:, i]
        if col_values.dtype == 'object':
            # Object --> numeric
            cleaned = col_values.apply(clean_numeric_value)

            num_valid = cleaned.notna().sum()
            total = len(cleaned)

            if total > 0:
                numeric_ratio = num_valid / total
            else:
                numeric_ratio = 0

            if numeric_ratio > 0.7:
                new_col_name = col + '_numeric'
                # Avoid duplicate keys
                if new_col_name in new_cols:
                    new_col_name = f"{col}_{i}_numeric"
                new_cols[new_col_name] = cleaned

    for col_name, col_values in new_cols.items():
        df[col_name] = col_values

    return df

def extract_units_from_column(df, col):
    unit_pattern = (
        r'(tCO2e?|kg|tons?|GJ|MWh|kWh|'
        r'VND|USD|EUR|'
        r'm[²³]|ha|hectares?|'
        r'liters?|gallons?|'
        r'persons?|employees?|people|'
        r'%|€|\$|£|'
        r'million|billion|trillion|thousand)'
    )
    
    units = []
    
    # Check column header 
    col_match = re.search(unit_pattern, str(col), re.IGNORECASE)
    if col_match:
        units.append(col_match.group(1))
    
    # Check cell values
    for val in df[col].dropna().head(20): 
        match = re.search(unit_pattern, str(val), re.IGNORECASE)
        if match:
            units.append(match.group(1))
    
    if units:
        most_common = Counter(units).most_common(1)[0][0]
        return most_common
    return None

def compress_markdown_table(md) :
    lines = md.split("\n")
    result = []
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            # Loại bỏ khoảng trắng thừa
            stripped = [p.strip() for p in parts]
            # Lấy phần tử bên trong bảng
            inner = stripped[1:-1]
            line = "| " + " | ".join(inner) + " |"
        result.append(line)
    return "\n".join(result)

def clean_table_dataframe(df):
    df = df.copy()
    
    # Chuyển về chuỗi, loại bỏ khoảng trắng thừa ở đầu/cuối và giữa
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)            
        col_str = remove_space(col_str) 
        col_str = col_str.strip()     
        cleaned_columns.append(col_str)
    df.columns = cleaned_columns
    
    # Xử lý tên cột rỗng hoặc NaN
    new_cols = []
    for col in df.columns:
        if col in ('', 'nan', 'none', 'NaN', 'None') or (isinstance(col, float) and pd.isna(col)):
            new_cols.append('Unnamed')
        else:
            new_cols.append(col)
    df.columns = new_cols

    # Deduplicate column names
    seen: dict = {}
    deduped_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            deduped_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped_cols.append(col)
    df.columns = deduped_cols

    df = df.dropna(how='all')  
    df = df.dropna(axis=1, how='all') 

    # Clean text cells
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(clean_table_cell)
        except Exception:
            continue
    
    df = normalize_numeric_columns(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    return df

def is_informative_image(pil_image, min_width, min_height = 150, min_area = 20000):
    w, h = pil_image.size
    
    # Filter theo size ảnh (quá nhỏ sẽ coi là logo)
    if w < min_width or h < min_height:
        return False
    if w * h < min_area:
        return False

    # Filter theo tỷ lệ (quá dài sẽ coi là ảnh trang trí)
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > 15:
        return False

    return True

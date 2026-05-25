def build_page_pack(disclosure_row, all_chunks_for_report):
    """Tạo page pack: các chunk báo cáo có trang trùng với page_list trong GCI.

    Trả [] khi:
    - disclosure được tham chiếu từ tài liệu ngoài (external_document_reference).
    - page_list rỗng (không khai báo trang nào).
    """
    if (disclosure_row.get("location_type") or "") == "external_document_reference":
        return []

    page_list = disclosure_row.get("page_list") or []
    if not page_list:
        return []

    target_pages = {int(p) for p in page_list}

    result = []
    for chunk in all_chunks_for_report:
        pages = chunk.get("page_numbers") or []
        if not pages:
            continue
        chunk_pages = {int(p) for p in pages}
        on_target = bool(chunk_pages & target_pages)
        if on_target:
            result.append(chunk)

    return result

__all__ = ["build_page_pack"]

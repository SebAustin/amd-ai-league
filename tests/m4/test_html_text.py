from amdleague.m4.html_text import parse_page


def _read(site_dir, rel):
    return (site_dir / rel).read_text()


def test_drops_scripts_and_styles(site_dir):
    page = parse_page(_read(site_dir, "index.html"), "https://acme.example/")
    joined = " ".join(page.passages)
    assert "ignore me" not in joined
    assert "color:red" not in joined
    assert page.title == "Acme Tools — Home"


def test_links_are_absolute(site_dir):
    page = parse_page(_read(site_dir, "index.html"), "https://acme.example/")
    urls = {link.url for link in page.links}
    assert "https://acme.example/about/contact.html" in urls
    texts = {link.text for link in page.links}
    assert "Contact us" in texts


def test_table_rows_keep_headers(site_dir):
    page = parse_page(_read(site_dir, "products/index.html"), "https://acme.example/products/")
    assert any("Product: Cordless Drill Pro" in p and "Price: $149.00" in p for p in page.passages)


def test_json_ld_becomes_passages(site_dir):
    page = parse_page(_read(site_dir, "products/drill-pro.html"), "https://acme.example/p")
    assert any("sku: CDP-2041" in p for p in page.passages)
    assert any("price: 149.00" in p for p in page.passages)


def test_handles_garbage_html():
    page = parse_page("<<<not html", "https://x.example/")
    assert page.url == "https://x.example/"
